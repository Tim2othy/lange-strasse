"""Player and AIPlayer Classes"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game_state import GameState


from ai_actions import Action
from ai_evaluator import random_action, best_action
from log import log


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
            case "td":
                return best_action(state, actions, "td")
            case _:
                raise ValueError(f"Unknown AI type: {self.ai_type}")

    def get_explanation(self, game, action: Action):
        """Get explanation for the AI's choice"""
        return
