"""Player and AIPlayer Classes"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState


import itertools
from dataclasses import dataclass
from typing import List
from log import log
from algorithms.heuristic import MoveEvaluator
from algorithms.td import TD_ALGORITHMS, td_action_score
from algorithms.dp import action_value
from game_state import GameState

from game.rules import ScoreCalculator, ScoreValidator, flatten


@dataclass
class Action:
    """Represents a possible action the AI can take"""

    dice_to_keep: List[int]  # Values of dice to keep
    stop_after: bool  # Whether to end turn after keeping these dice

    def __str__(self):
        dice_str = " ".join(map(str, self.dice_to_keep))
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
    def _valid_combinations(
        available_dice: List[int], kept_values: List[int]
    ) -> List[List[int]]:
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
        projected = ScoreCalculator.calculate_score_from_groups(
            dice_set.kept_groups + [combo]
        )
        return projected >= 300


"""AI move evaluation for Lange Strasse"""


def random_action(state: GameState, actions: list[Action]) -> Action:
    """Choose a random valid action"""
    import random

    return random.choice(actions)


def best_action(state: GameState, actions: list[Action], algorithm: str) -> Action:
    """Choose action with highest value based on some algorithm"""

    if not actions:
        exception = "No valid actions available for the current state."
        raise ValueError(exception)

    # Evaluate all actions
    best_action = None
    best_score = float("-inf")

    for action in actions:
        if algorithm == "simple":
            score = MoveEvaluator().evaluate_action(state, action)
        elif algorithm == "dp":
            score = action_value(state, action)
        elif algorithm in TD_ALGORITHMS:
            score = td_action_score(state, action, algorithm)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if score > best_score:
            best_score = score
            best_action = action

    text = "No valid action found despite valid_actions being non-empty."
    assert best_action is not None, text
    return best_action


class Player:
    """Represents a player in the game"""

    def __init__(self, name):
        self.name = name
        self.total_score = 0
        self.money = 0  # Money in cents
        self.has_strich = False

    def add_money(self, cents):
        """Add money (can be negative for losses)"""
        self.money += cents
        if cents > 0:
            log(f"💰 {self.name} gains {cents}¢!")
        else:
            log(f"💸 {self.name} loses {abs(cents)}¢!")


class AIPlayer(Player):
    """AI-controlled player"""

    def __init__(self, name: str, ai_type: str = "simple"):
        super().__init__(name)

        self.ai_type = ai_type

    def choose_action(self, state: GameState, actions: list[Action]) -> Action:
        """Let the AI choose an action"""

        match self.ai_type:
            case "simple":
                return best_action(state, actions, "simple")
            case "random":
                return random_action(state, actions)
            case "dp":
                return best_action(state, actions, "dp")
            case name if name in TD_ALGORITHMS:
                return best_action(state, actions, name)
            case _:
                raise ValueError(f"Unknown AI type: {self.ai_type}")

    def get_explanation(self, game, action: Action):
        """Get explanation for the AI's choice"""
        return
