"""Small MLP value function over afterstates, learned by TD(0) self-play.

Same setup as the linear models in td.py -- V(afterstate) = the acting player's
expected FINAL MONEY, terminal-only reward, TD(0) bootstrap between a player's
consecutive afterstates -- but the value function is a one-hidden-layer neural
network, so it can represent the nonlinear interactions a linear model provably
cannot (e.g. "opponent already crossed 10K AND this is my last turn -> never
stop short of the lead").

The encoding is the full lossless state plus the money-rule atoms (each mirrors
one way money changes hands; see game_state._atom_features) plus the DP
solver's turn-value, so the net learns a *correction* to DP rather than dice
combinatorics from scratch.

Train with:  python -m algorithms.nn [n_games] [alpha] [epsilon]
"""

import pickle
from pathlib import Path

import numpy as np

from algorithms.td import GAME_KEYS, MONEY_SCALE, _encode
from game_state import GameState

HIDDEN = 64

# fmt: off
NN_KEYS = GAME_KEYS + [
    # money-rule atoms: one per way money changes hands
    "prospective_score", "points_to_win", "crosses_10k",
    "score_best_opp", "is_leading", "gap_p2", "gap_p3",
    "below_5000_me", "points_to_5000", "below_5000_p2", "below_5000_p3",
    "opps_below_5000",
    "round_bonus_alive", "flag_super_strasse",
    # the DP solver's turn-value, as a hint to correct rather than relearn
    "dp_value",
]
# fmt: on

WEIGHTS_PATH = Path(__file__).with_name("nn_weights.pkl")

nn_features = _encode(NN_KEYS)  # (state, action) -> list[float], with bias
DIM = len(NN_KEYS) + 1  # _encode appends a constant bias input


class MLP:
    """One-hidden-layer tanh network with a linear scalar output, plain SGD."""

    def __init__(self, dim: int = DIM, hidden: int = HIDDEN, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, dim**-0.5, (hidden, dim))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, hidden**-0.5, hidden)
        self.b2 = 0.0
        self.games_trained = 0  # cumulative self-play games behind these weights

    def value(self, features: list[float]) -> float:
        x = np.asarray(features)
        return float(self.w2 @ np.tanh(self.W1 @ x + self.b1) + self.b2)

    def values(self, feature_rows: list[list[float]]) -> np.ndarray:
        """Values of many feature vectors at once (one action per row)."""
        X = np.asarray(feature_rows)
        return np.tanh(X @ self.W1.T + self.b1) @ self.w2 + self.b2

    def update(self, features: list[float], target: float, alpha: float) -> float:
        """One SGD step of all weights toward `target`; returns the TD error."""
        x = np.asarray(features)
        h = np.tanh(self.W1 @ x + self.b1)
        err = target - float(self.w2 @ h + self.b2)
        dh = err * self.w2 * (1.0 - h * h)
        self.w2 += alpha * err * h
        self.b2 += alpha * err
        self.W1 += alpha * np.outer(dh, x)
        self.b1 += alpha * dh
        return err

    def save(self, path: Path = WEIGHTS_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "keys": NN_KEYS,
                    "hidden": len(self.b1),
                    "games": self.games_trained,
                    "W1": self.W1,
                    "b1": self.b1,
                    "w2": self.w2,
                    "b2": self.b2,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> "MLP":
        """Load weights if present and encoded with the same keys, else fresh."""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if data.get("keys") == NN_KEYS:
                    model = cls(hidden=data["hidden"])
                    model.W1, model.b1 = data["W1"], data["b1"]
                    model.w2, model.b2 = data["w2"], data["b2"]
                    model.games_trained = data.get("games", 0)
                    return model
            except (pickle.UnpicklingError, EOFError, OSError, KeyError):
                pass  # missing/corrupt/stale -> start fresh
        return cls()


# --- play-time model (lazily loaded, cached) --------------------------------- #
NN_ALGORITHMS = {"nn"}  # algorithm strings that dispatch here (see ai_player)

_MODEL: MLP | None = None


def _model() -> MLP:
    global _MODEL
    if _MODEL is None:
        _MODEL = MLP.load()
        if _MODEL.games_trained:
            print(
                f"[nn] using weights trained on "
                f"{_MODEL.games_trained} self-play games"
            )
        else:
            print(
                "[nn] no trained weights -- playing untrained "
                "(run: python -m algorithms.nn)"
            )
    return _MODEL


def nn_action_score(state: GameState, action) -> float:
    """Value the AI assigns to `action` (used by AIPlayer's "nn" algorithm)."""
    return _model().value(nn_features(state, action))


# --- training --------------------------------------------------------------- #
def train(
    alpha: float,
    epsilon: float,
    n_games: int,
    seed: "int | None" = None,
    log_every: int = 500,
) -> MLP:
    """Self-play TD(0) for the MLP. Continues from saved weights; saves at end."""
    import random

    import log
    from env import LangeStrasseEnv  # local import to avoid an import cycle

    log.VERBOSE = False  # silence game output during training
    if seed is not None:
        random.seed(seed)

    model = MLP.load()

    for game_i in range(1, n_games + 1):
        env = LangeStrasseEnv()
        last_features: dict[int, list[float]] = {}  # player -> its last chosen afterstate
        info = {}

        while not env.done:
            state = env.observe()
            player = env.current_player_idx
            actions = env.legal_actions()
            feats = [nn_features(state, a) for a in actions]

            if random.random() < epsilon:
                choice = random.randrange(len(actions))
            else:
                choice = int(np.argmax(model.values(feats)))
            chosen = feats[choice]

            # Bootstrap: this player's previous afterstate -> value of the new one.
            if player in last_features:
                model.update(last_features[player], model.value(chosen), alpha)
            last_features[player] = chosen

            _obs, _reward, _done, info = env.step(actions[choice])

        # Terminal: pull each player's last afterstate toward its final money.
        for player, feats in last_features.items():
            target = info["standings"][player]["money"] / MONEY_SCALE
            model.update(feats, target, alpha)

        if log_every and game_i % log_every == 0:
            print(f"  ...{game_i}/{n_games} games")

    model.games_trained += n_games
    model.save()
    global _MODEL
    _MODEL = model  # so same-process evaluation uses the fresh weights
    return model


def _evaluate(games: int = 300) -> None:
    """Quick arena: the net vs the DP solver vs the simple heuristic."""
    import log
    import play

    log.VERBOSE = False

    matchup = ["nn", "dp", "simple"]
    wins, money = play.run_matchup(games, matchup)

    print(f"\nEval over {games} games:")
    for algo in matchup:
        print(
            f"  {algo:8s}: {wins[algo]:3d} wins ({wins[algo] / games:.0%}), "
            f"{money[algo] / games:+.1f}c/game"
        )


def run_training(alpha: float, epsilon: float, n_games: int) -> None:
    import time

    print(f"Training nn by self-play for {n_games} games...")
    start = time.perf_counter()
    model = train(alpha, epsilon, n_games)
    print(
        f"Done in {time.perf_counter() - start:.1f}s. "
        f"Model now trained on {model.games_trained} games total, "
        f"saved to {WEIGHTS_PATH.name}."
    )
    _evaluate()


if __name__ == "__main__":
    import sys

    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    run_training(alpha, epsilon, n_games)
