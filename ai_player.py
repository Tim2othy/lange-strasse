"""Player and AIPlayer Classes"""

from ai_actions import Action
from ai_evaluator import SimpleAI


class Player:
    """Represents a player in the game"""

    def __init__(self, name):
        self.name = name
        self.total_score = 0
        self.money = 0  # Money in cents
        self.has_strich = False
        self.is_ai = False

    def add_money(self, cents):
        """Add money (can be negative for losses)"""
        self.money += cents
        if cents > 0:
            print(f"💰 {self.name} gains {cents}¢!")
        else:
            print(f"💸 {self.name} loses {abs(cents)}¢!")


class AIPlayer(Player):
    """AI-controlled player"""

    def __init__(self, name: str, ai_type: str = "simple"):
        super().__init__(name)
        self.is_ai = True
        # Ensure AIPlayer has all the same attributes as Player
        self.has_strich = False

        if ai_type == "simple":
            self.ai = SimpleAI()
        else:
            raise ValueError(f"Unknown AI type: {ai_type}")

    def choose_action(self, game) -> Action:
        """Let the AI choose an action"""
        return self.ai.choose_action(game)

    def get_explanation(self, game, action: Action) -> str:
        """Get explanation for the AI's choice"""
        return self.ai.get_action_explanation(game, action)
