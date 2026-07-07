"""AI-controlled player: picks the legal action with the highest value under
its configured algorithm."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from algorithms.dp import action_value as dp_action_value
from algorithms.heuristic import MoveEvaluator
from algorithms.nn import NN_ALGORITHMS, nn_action_score
from algorithms.td import TD_ALGORITHMS, td_action_score
from game.game import Player
from game.rules import Action

if TYPE_CHECKING:
    from game_state import GameState

_HEURISTIC = MoveEvaluator()


class AIPlayer(Player):
    """A player whose decisions come from one of the algorithms in src/algorithms."""

    def __init__(self, name: str, ai_type: str = "simple"):
        super().__init__(name)
        self.ai_type = ai_type

    def describe(self) -> str:
        return f"{self.name} ({self.ai_type})"

    def choose_action(self, state: GameState, actions: list[Action]) -> Action:
        """Pick the best action (ties go to the first); random plays uniformly."""
        if not actions:
            raise ValueError("No valid actions available for the current state.")
        if self.ai_type == "random":
            return random.choice(actions)
        return max(actions, key=lambda action: self._action_value(state, action))

    def _action_value(self, state: GameState, action: Action) -> float:
        if self.ai_type == "simple":
            return _HEURISTIC.evaluate_action(state, action)
        if self.ai_type == "dp":
            return dp_action_value(state, action)
        if self.ai_type in TD_ALGORITHMS:
            return td_action_score(state, action, self.ai_type)
        if self.ai_type in NN_ALGORITHMS:
            return nn_action_score(state, action)
        raise ValueError(f"Unknown AI type: {self.ai_type}")
