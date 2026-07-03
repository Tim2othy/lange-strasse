"""Headless environment wrapper around TheGame, for simulation and RL.

It exposes the usual ``reset`` / ``legal_actions`` / ``step`` loop and drives the
game with no console I/O. Because Lange Strasse is multi-player and turn-based,
consecutive steps may belong to *different* players: each ``step`` reports which
seat acted (``info["actor"]``) so a self-play trainer can attribute reward
correctly. Forced turn-endings (a dead roll / Totale, where the player has no
decision to make) are auto-resolved inside ``reset``/``step`` so every observation
handed back is a real decision point.
"""

import time
from collections import Counter

import log
from ai_actions import Action, ActionGenerator
from ai_evaluator import random_action, simple_action
from config import N_GAMES, VERBOSE
from game import TheGame
from game_state import GameState, StateExtractor


class LangeStrasseEnv:
    """One ``step`` == one player's single keep/stop decision."""

    def __init__(self):
        self.game: TheGame | None = None
        self.reset()

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self) -> GameState:
        """Start a fresh game and return the first decision-point observation."""
        log.set_verbose(VERBOSE)
        self.game = TheGame(choice=1)  # plain players; the env supplies the actions
        self.game.start_new_turn()
        self._advance_to_decision()
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

        self._advance_to_decision()

        reward = self.game.players[actor].money - money_before
        info = {"actor": actor, "result": result}
        if self.done:
            info["standings"] = self.standings()
            return None, reward, True, info
        return self.observe(), reward, False, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _advance_to_decision(self) -> None:
        """Fast-forward through forced turn-endings until a real decision or game end.

        Mirrors the guard at the top of the interactive loop: a player with no
        keepable dice ends their turn (paying the Totale penalty if it happened
        on the first roll), and we skip on to the next player.
        """
        while not self.game.game_over:
            dice_set = self.game.dice_set
            if dice_set.game_over or not dice_set.can_keep_any_dice():
                if dice_set.check_totale():
                    self.game.handle_totale()
                self.game.end_turn(0)
                continue
            break

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


def play_game(policies) -> list[dict]:
    """Play one full game headlessly.

    ``policies`` is one callable per seat with signature ``(state, actions) -> action``
    (e.g. ``random_action`` / ``simple_action``). Returns the final standings.
    """
    env = LangeStrasseEnv()
    state = env.observe()
    while not env.done:
        actions = env.legal_actions()
        action = policies[env.current_player_idx](state, actions)
        state, _reward, _done, info = env.step(action)
    return info["standings"]


if __name__ == "__main__":
    # Quick smoke test / demo: simulate a batch of games silently.

    seats = [simple_action, random_action, random_action]

    start = time.perf_counter()
    wins = Counter()
    for _ in range(N_GAMES):
        standings = play_game(seats)
        winner = next(s["name"] for s in standings if s["is_winner"])
        wins[winner] += 1
    elapsed = time.perf_counter() - start

    print(
        f"Simulated {N_GAMES} games in {elapsed:.2f}s "
        f"({N_GAMES / elapsed:.0f} games/s)"
    )
    print("Wins by seat:", dict(wins))
