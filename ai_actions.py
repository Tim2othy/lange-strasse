"""AI action representation and generation for Lange Strasse"""
import itertools
from dataclasses import dataclass
from typing import List

from scoring import ScoreCalculator, ScoreValidator, flatten


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

        kept_values = flatten(dice_set.kept_groups)

        for combo in ActionGenerator._valid_combinations(available_dice, kept_values):
            actions.append(Action(dice_to_keep=combo, stop_after=False))
            if ActionGenerator._can_stop_with_combo(dice_set, combo):
                actions.append(Action(dice_to_keep=combo, stop_after=True))

        return actions

    @staticmethod
    def _valid_combinations(available_dice: List[int], kept_values: List[int]) -> List[List[int]]:
        """All distinct, legally-keepable subsets of the available dice."""
        combos = []
        seen = set()
        for r in range(1, len(available_dice) + 1):
            for combo in itertools.combinations(available_dice, r):
                key = tuple(sorted(combo))  # equivalent keeps differ only in order
                if key in seen:
                    continue
                seen.add(key)
                if ScoreValidator.is_valid_keep(list(combo), kept_values)[0]:
                    combos.append(list(combo))
        return combos

    @staticmethod
    def _can_stop_with_combo(dice_set, combo: List[int]) -> bool:
        """Check if the player can stop after keeping this combination."""
        # Can't stop while keeping all 6 dice.
        if sum(dice_set.kept_dice) + len(combo) == 6:
            return False

        # Must have at least 300 points from the current set.
        projected = ScoreCalculator.calculate_score_from_groups(dice_set.kept_groups + [combo])
        return projected >= 300
