"""AI player integration for Lange Strasse"""
from ai_actions import Action
from ai_evaluator import SimpleAI
from game import Player


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
