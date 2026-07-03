"""Player and AIPlayer Classes"""

from ai_actions import Action
from ai_evaluator import random_action, simple_action
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

    def choose_action(self, game_state, actions) -> Action:
        """Let the AI choose an action"""

        match self.ai_type:
            case "simple":
                return simple_action(game_state, actions)
            case "random":
                return random_action(game_state, actions)
            case _:
                raise ValueError(f"Unknown AI type: {self.ai_type}")

    def get_explanation(self, game, action: Action):
        """Get explanation for the AI's choice"""
        return
