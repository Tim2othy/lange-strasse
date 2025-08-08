"""Simplified state representation for neural network training"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from collections import Counter

@dataclass
class NNState:
    """Simplified state for neural network - only essential information"""
    # 1. Available dice and kept dice info
    available_dice: List[int]  # 6 elements, 0 if kept
    kept_dice_counts: List[int]  # Count of each value 1-6 currently kept
    kept_1s_grouped: int  # How many 1s are in groups (vs individual)
    kept_5s_grouped: int  # How many 5s are in groups (vs individual)

    # 2. Scoring
    turn_score: int  # Total accumulated this turn
    current_set_score: int  # Score from current dice set only

    # 3. Player scores and strich status
    player_scores: List[int]  # Total scores for all 3 players
    player_strich: List[bool]  # Whether each player has gotten a strich

    # 4. Turn information
    turn_number: int

    # 5. Player order
    starting_player: int  # 0, 1, or 2
    current_player: int   # 0, 1, or 2

class NNStateExtractor:
    """Extract simplified state for neural network"""

    @staticmethod
    def extract_state(game) -> NNState:
        """Extract essential state information for NN"""
        dice_set = game.dice_set

        # Count available dice by value
        available_dice_counts = [0] * 6
        avail_d_values = dice_set.get_available_dice_values()
        for i in avail_d_values:
            available_dice_counts[i - 1] += 1

        # Count kept dice by value
        kept_dice_counts = [0] * 6  # Index 0-5 for values 1-6
        kept_1s_grouped = 0
        kept_5s_grouped = 0

        for group in dice_set.kept_groups:
            group_counts = Counter(group)
            for value, count in group_counts.items():
                kept_dice_counts[value - 1] += count

                # Track if 1s/5s are in groups (3+ of same value)
                if value == 1 and count >= 3:
                    kept_1s_grouped += count
                elif value == 5 and count >= 3:
                    kept_5s_grouped += count

        # 2. Scoring
        turn_score = dice_set.get_current_total_score()
        current_set_score = dice_set.get_current_score()

        # 3. Player information
        player_scores = [player.total_score for player in game.players]
        player_strich = [getattr(player, 'has_strich', False) for player in game.players]

        # 4. Turn number (we'll need to add this to game)
        turn_number = getattr(game, 'turn_number', 1)

        # 5. Player order
        starting_player = getattr(game, 'starting_player_idx', 0)
        current_player = game.current_player_idx

        return NNState(
            available_dice=available_dice,
            kept_dice_counts=kept_dice_counts,
            kept_1s_grouped=kept_1s_grouped,
            kept_5s_grouped=kept_5s_grouped,
            turn_score=turn_score,
            current_set_score=current_set_score,
            player_scores=player_scores,
            player_strich=player_strich,
            turn_number=turn_number,
            starting_player=starting_player,
            current_player=current_player
        )

    @staticmethod
    def to_vector(state: NNState) -> np.ndarray:
        """Convert state to normalized vector for neural network"""
        vector = []

        # 1. Available dice (6 elements, normalize 0-6 to 0-1)
        for die in state.available_dice:
            vector.append(die / 6.0)

        # Kept dice counts (6 elements, normalize by max possible)
        for count in state.kept_dice_counts:
            vector.append(count / 6.0)

        # Grouped 1s and 5s (normalize by max possible)
        vector.append(state.kept_1s_grouped / 6.0)
        vector.append(state.kept_5s_grouped / 6.0)

        # 2. Scores (normalize by reasonable maximums)
        vector.append(state.turn_score / 10000.0)  # Max reasonable turn score
        vector.append(state.current_set_score / 2000.0)  # Max reasonable set score

        # 3. Player scores (normalize by win condition 10,000)
        for score in state.player_scores:
            vector.append(score / 10000.0)

        # Player strich status (binary)
        for has_strich in state.player_strich:
            vector.append(1.0 if has_strich else 0.0)

        # 4. Turn number (normalize by expected max ~20 turns)
        vector.append(state.turn_number / 20.0)

        # 5. Player indices (normalize 0,1,2 to 0,0.5,1)
        vector.append(state.starting_player / 2.0)
        vector.append(state.current_player / 2.0)

        return np.array(vector, dtype=np.float32)

    @staticmethod
    def get_vector_size() -> int:
        """Get the size of the state vector"""
        # 6 (available dice) + 6 (kept counts) + 2 (grouped) + 2 (scores) +
        # 3 (player scores) + 3 (strich) + 1 (turn) + 2 (player indices) = 25
        return 25

class ActionEncoder:
    """Encode/decode actions for neural network"""

    def __init__(self):
        # We'll create a mapping of all possible actions to indices
        self.action_to_index = {}
        self.index_to_action = {}
        self._build_action_space()

    def _build_action_space(self):
        """Build discrete action space"""
        index = 0

        # For each possible combination of dice values to keep
        # We'll use a more compact representation:
        # Each action is encoded as: (dice_mask, stop_flag)
        # dice_mask: 6-bit representation of which dice positions to keep
        # stop_flag: whether to end turn

        for mask in range(1, 64):  # 1 to 63 (can't keep 0 dice)
            # Without stopping
            action = (mask, False)
            self.action_to_index[action] = index
            self.index_to_action[index] = action
            index += 1

            # With stopping
            action = (mask, True)
            self.action_to_index[action] = index
            self.index_to_action[index] = action
            index += 1

    def encode_action(self, dice_positions: List[int], stop: bool) -> int:
        """Encode dice positions and stop flag to action index"""
        # Convert dice positions to mask
        mask = 0
        for pos in dice_positions:
            mask |= (1 << pos)

        action = (mask, stop)
        return self.action_to_index.get(action, -1)

    def decode_action(self, action_index: int) -> Tuple[List[int], bool]:
        """Decode action index to dice positions and stop flag"""
        if action_index not in self.index_to_action:
            return [], False

        mask, stop = self.index_to_action[action_index]

        # Convert mask to dice positions
        positions = []
        for i in range(6):
            if mask & (1 << i):
                positions.append(i)

        return positions, stop

    def get_action_space_size(self) -> int:
        """Get total number of possible actions"""
        return len(self.action_to_index)

    def get_valid_actions(self, game) -> List[int]:
        """Get valid action indices for current game state"""
        valid_indices = []

        # Get available dice positions
        available_positions = game.dice_set.get_available_dice()
        if not available_positions:
            return valid_indices

        # Test each possible action
        for action_index in range(self.get_action_space_size()):
            positions, stop = self.decode_action(action_index)

            # Check if positions are available
            if not all(pos in available_positions for pos in positions):
                continue

            # Convert positions to dice values for validation
            dice_values = [game.dice_set.dice[pos] for pos in positions]

            # Check if this is a valid move
            if self._is_valid_move(game, dice_values, stop):
                valid_indices.append(action_index)

        return valid_indices

    def _is_valid_move(self, game, dice_values: List[int], stop: bool) -> bool:
        """Check if a move is valid (simplified validation)"""
        try:
            # Use the existing validation from the game
            dice_set = game.dice_set

            # Quick validation - can't stop with all 6 dice
            total_kept_after = sum(dice_set.kept_dice) + len(dice_values)
            if stop and total_kept_after == 6:
                return False

            # Use the game's validation logic
            success, _ = dice_set.keep_dice_by_value(dice_values, stop_after=False)
            return success
        except:
            return False

# Example usage and testing
if __name__ == "__main__":
    # Test the action encoder
    encoder = ActionEncoder()
    print(f"Action space size: {encoder.get_action_space_size()}")

    # Test encoding/decoding
    positions = [0, 1, 2]  # Keep first 3 dice
    stop = True
    action_idx = encoder.encode_action(positions, stop)
    decoded_pos, decoded_stop = encoder.decode_action(action_idx)

    print(f"Original: positions={positions}, stop={stop}")
    print(f"Encoded: {action_idx}")
    print(f"Decoded: positions={decoded_pos}, stop={decoded_stop}")

    # Test state vector size
    print(f"State vector size: {NNStateExtractor.get_vector_size()}")
