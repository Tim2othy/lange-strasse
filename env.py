"""Headless RL environment wrapping TheGame.

This is a *library*, not a runnable script -- run everything through main.py.
It exists for the one case that differs from a normal game: an external learner
that supplies actions one step at a time and reads rewards back. (A normal
AI-vs-AI game, where each player decides for itself, is driven by main.py and
does not need this.)

The game's players are plain score/money holders here -- their ``choose_action``
is never called, because actions arrive through ``step``. Because the game is
multi-player and turn-based, consecutive steps may belong to different seats, so
each ``step`` reports which seat acted (``info["actor"]``) for reward attribution.
Forced turn-endings (dead roll / Totale) are auto-resolved so every observation
handed back is a real decision point.
"""

from ai_actions import Action, ActionGenerator
from game import TheGame
from game_state import GameState, StateExtractor


class LangeStrasseEnv:
    """One ``step`` == one player's single keep/stop decision."""

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self) -> GameState:
        """Start a fresh game and return the first decision-point observation."""
        self.game: TheGame = TheGame(choice=1)  # plain players; actions come via step()
        self.game.start_new_turn()
        self.game.advance_to_decision()
        return self.observe()

    def observe(self) -> GameState:
        return StateExtractor.extract_state(self.game)

    def legal_actions(self) -> list[Action]:
        return ActionGenerator.get_valid_actions(self.game)

    @property
    def current_player_idx(self) -> int:
        return self.game.current_player_idx

    @property
    def done(self) -> bool:
        return self.game.game_over

    def step(self, action: Action):
        """Apply ``action`` for the current player.

        Returns ``(obs, reward, done, info)`` where ``obs`` is None at game end,
        ``reward`` is the acting player's own money change over this step, and
        ``info`` carries the actor and (at game end) the final standings.
        """
        actor = self.game.current_player_idx
        money_before = self.game.players[actor].money

        success, result = self.game.process_dice_action(
            action.dice_to_keep, action.stop_after
        )
        if not success:
            raise ValueError(f"Illegal action {action} was passed to step(): {result}")

        self.game.advance_to_decision()

        reward = self.game.players[actor].money - money_before
        info = {"actor": actor, "result": result}
        if self.done:
            info["standings"] = self.standings()
            return None, reward, True, info
        return self.observe(), reward, False, info

    def standings(self) -> list[dict]:
        """Final per-player results, only meaningful once the game is over."""
        return [
            {
                "name": p.name,
                "score": p.total_score,
                "money": p.money,
                "is_winner": p is self.game.winner,
            }
            for p in self.game.players
        ]
