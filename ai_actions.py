"""AI action representation and generation for Lange Strasse"""
from dataclasses import dataclass
from typing import List, Tuple, Set
from collections import Counter
import itertools

@dataclass
class Action:
    """Represents a possible action the AI can take"""
    dice_to_keep: List[int]  # Values of dice to keep
    stop_after: bool  # Whether to end turn after keeping these dice

    def __str__(self):
        dice_str = ' '.join(map(str, self.dice_to_keep))
        return f"keep {dice_str}" + (" stop" if self.stop_after else "")

class ActionGenerator:
    """Generate valid actions from a game state"""

    @staticmethod
    def get_valid_actions(game) -> List[Action]:
        """Get all valid actions from current game state"""
        dice_set = game.dice_set
        actions = []

        if dice_set.game_over:
            return actions

        available_dice = dice_set.get_available_dice_values()
        if not available_dice:
            return actions

        # Get all currently kept values for validation
        all_kept_values = []
        for group in dice_set.kept_groups:
            all_kept_values.extend(group)

        # Generate all possible combinations of available dice
        valid_combinations = ActionGenerator._get_valid_combinations(
            available_dice, all_kept_values
        )

        for combo in valid_combinations:
            # Check if we can stop with this combination
            can_stop = ActionGenerator._can_stop_with_combo(dice_set, combo)

            # Add action without stopping
            actions.append(Action(dice_to_keep=combo, stop_after=False))

            # Add action with stopping if valid
            if can_stop:
                actions.append(Action(dice_to_keep=combo, stop_after=True))

        return actions

    @staticmethod
    def _get_valid_combinations(available_dice: List[int], kept_values: List[int]) -> List[List[int]]:
        """Generate all valid combinations of dice to keep"""
        valid_combos = []
        available_counts = Counter(available_dice)

        # Generate all possible subsets of available dice
        for r in range(1, len(available_dice) + 1):
            for combo in itertools.combinations(available_dice, r):
                combo_list = list(combo)
                combo_counts = Counter(combo_list)

                # Check if we have enough dice of each type
                valid = True
                for value, count in combo_counts.items():
                    if available_counts[value] < count:
                        valid = False
                        break

                if valid and ActionGenerator._is_valid_keep(combo_list, kept_values):
                    valid_combos.append(combo_list)

        # Remove duplicates (order doesn't matter)
        unique_combos = []
        seen = set()
        for combo in valid_combos:
            sorted_combo = tuple(sorted(combo))
            if sorted_combo not in seen:
                seen.add(sorted_combo)
                unique_combos.append(combo)

        return unique_combos

    @staticmethod
    def _is_valid_keep(dice_values: List[int], kept_values: List[int]) -> bool:
        """Check if a combination of dice can be legally kept"""
        # Check for special combinations first
        temp_kept_values = kept_values + dice_values

        # Check if this would be Talheim (3 pairs with exactly 6 dice)
        if len(temp_kept_values) == 6:
            temp_counts = Counter(temp_kept_values)
            if len(temp_counts) == 3 and all(count == 2 for count in temp_counts.values()):
                return True  # Talheim is always valid

        # Check if this would complete Lange Strasse
        if set(temp_kept_values).issuperset({1, 2, 3, 4, 5, 6}):
            return True  # Lange Strasse is always valid

        # Regular validation rules
        dice_counts = Counter(dice_values)
        kept_counts = Counter(kept_values)

        for value, count in dice_counts.items():
            # 1s and 5s can always be kept individually
            if value == 1 or value == 5:
                continue

            # Other values need 3+ total (kept + new)
            total_count = kept_counts.get(value, 0) + count
            if total_count < 3 and count > 0:
                return False

        return True

    @staticmethod
    def _can_stop_with_combo(dice_set, combo: List[int]) -> bool:
        """Check if player can stop after keeping this combination"""
        # Can't stop if keeping all 6 dice
        total_kept_after = sum(dice_set.kept_dice) + len(combo)
        if total_kept_after == 6:
            return False

        # Must have at least 300 points from current set
        # Simulate the score after keeping this combo
        temp_groups = dice_set.kept_groups.copy()
        temp_groups.append(combo)

        # Calculate what the score would be
        from scoring import ScoreCalculator
        projected_score = ScoreCalculator.calculate_score_from_groups(temp_groups)

        return projected_score >= 300
