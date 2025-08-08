"""AI state representation for Lange Strasse"""
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import Counter

@dataclass
class PlayerState:
    """State information for a single player"""
    total_score: int
    has_strich: bool
    money: int

@dataclass
class GameState:
    """Complete game state for AI decision making"""
    # Dice information
    available_dice: List[int]  # Values of available dice (0 means not available)
    kept_groups: List[List[int]]  # Groups of kept dice

    # Scoring
    turn_score: int  # Total accumulated this turn
    current_set_score: int  # Score from current dice set only

    # Player information
    players: List[PlayerState]
    current_player_idx: int
    starting_player_idx: int

    # Game context
    turn_number: int
    is_final_round: bool

    # Special flags
    lange_strasse_achieved: bool
    can_complete_lange_strasse: bool
    can_complete_talheim: bool
    is_totale_risk: bool

class StateExtractor:
    """Extract AI state from game objects"""

    @staticmethod
    def extract_state(game) -> GameState:
        """Extract complete game state for AI"""
        dice_set = game.dice_set

        # Extract available dice (0 for kept positions)
        available_dice = [0] * 6
        available_indices = dice_set.get_available_dice()
        for idx in available_indices:
            available_dice[idx] = dice_set.dice[idx]

        # Extract kept groups
        kept_groups = dice_set.kept_groups.copy()

        # Extract scoring information
        turn_score = dice_set.get_current_total_score()
        current_set_score = dice_set.get_current_score()

        # Extract player information
        players = []
        for player in game.players:
            players.append(PlayerState(
                total_score=player.total_score,
                has_strich=getattr(player, 'has_strich', False),  # Will implement this
                money=player.money
            ))

        # Check special combinations
        can_complete_lange_strasse = StateExtractor._can_complete_lange_strasse(dice_set)
        can_complete_talheim = StateExtractor._can_complete_talheim(dice_set)
        is_totale_risk = not dice_set.can_keep_any_dice()

        return GameState(
            available_dice=available_dice,
            kept_groups=kept_groups,
            turn_score=turn_score,
            current_set_score=current_set_score,
            players=players,
            current_player_idx=game.current_player_idx,
            starting_player_idx=game.starting_player_idx,
            turn_number=getattr(game, 'turn_number', 1),  # Will implement this
            is_final_round=game.final_turn,
            lange_strasse_achieved=dice_set.lange_strasse_achieved,
            can_complete_lange_strasse=can_complete_lange_strasse,
            can_complete_talheim=can_complete_talheim,
            is_totale_risk=is_totale_risk
        )

    @staticmethod
    def _can_complete_lange_strasse(dice_set) -> bool:
        """Check if Lange Strasse can be completed with available dice"""
        if dice_set.lange_strasse_achieved:
            return False

        # Get all kept values
        all_kept = []
        for group in dice_set.kept_groups:
            all_kept.extend(group)

        # Get available dice values
        available_values = dice_set.get_available_dice_values()

        # Check if we can complete 1-2-3-4-5-6
        all_values = set(all_kept + available_values)
        return all_values.issuperset({1, 2, 3, 4, 5, 6})

    @staticmethod
    def _can_complete_talheim(dice_set) -> bool:
        """Check if Talheim can be completed with current dice"""
        all_kept = []
        for group in dice_set.kept_groups:
            all_kept.extend(group)

        available_values = dice_set.get_available_dice_values()
        all_values = all_kept + available_values

        if len(all_values) != 6:
            return False

        counts = Counter(all_values)
        return len(counts) == 3 and all(count == 2 for count in counts.values())

    @staticmethod
    def to_vector(state: GameState) -> List[float]:
        """Convert state to normalized vector for ML models"""
        vector = []

        # Available dice (6 elements, 0-6, normalize to 0-1)
        for die in state.available_dice:
            vector.append(die / 6.0)

        # Kept dice counts by value (6 elements)
        all_kept = []
        for group in state.kept_groups:
            all_kept.extend(group)
        kept_counts = Counter(all_kept)
        for i in range(1, 7):
            vector.append(kept_counts.get(i, 0) / 6.0)  # Normalize by max possible

        # Scores (normalize by reasonable maximums)
        vector.append(state.turn_score / 10000.0)  # Max expected turn score
        vector.append(state.current_set_score / 2000.0)  # Max reasonable set score

        # Player scores (normalize by win condition)
        for player in state.players:
            vector.append(player.total_score / 10000.0)
            vector.append(1.0 if player.has_strich else 0.0)

        # Game context
        vector.append(state.current_player_idx / 2.0)  # 0, 1, or 2 -> 0, 0.5, 1
        vector.append(state.starting_player_idx / 2.0)
        vector.append(state.turn_number / 20.0)  # Normalize by expected max turns

        # Boolean flags
        vector.extend([
            1.0 if state.is_final_round else 0.0,
            1.0 if state.lange_strasse_achieved else 0.0,
            1.0 if state.can_complete_lange_strasse else 0.0,
            1.0 if state.can_complete_talheim else 0.0,
            1.0 if state.is_totale_risk else 0.0
        ])

        return vector
