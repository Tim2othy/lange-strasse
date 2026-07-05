"""Simple linear TD(0) value function over between-turn states, learned by self-play.

Estimates ``V(afterstate) = w . features(afterstate)`` -- the acting player's expected
FINAL MONEY. We use *afterstates* (the position right after a move, before the next
roll), TD-Gammon style: the value implicitly averages over the future dice, so we just
pick the action whose afterstate is worth most. Afterstates also dodge a trap of a
plain linear ``Q(s, a)``: with additive state+action features the state term is the
same for every action and cancels in the argmax, so the state couldn't influence the
choice. An afterstate value can, because each action lands in a different position.

Everything is from the *acting player's* perspective (``to_vector`` orders that player
first), so ``V`` consistently means "my expected final money".

Reward is terminal only: final money / MONEY_SCALE, no discount (episodic). Every
intermediate money flow (Strasse, Totale, end-game settlement) is already part of the
final total, so a terminal reward is complete; TD bootstrapping carries the credit
back through the turns.

Train it with:  python -m algorithms.td [n_games]
"""

import pickle
from pathlib import Path

from algorithms.turn_value import action_value
from game_state import GameState, StateExtractor

MONEY_SCALE = 100.0  # final money (cents) is divided by this to form the TD target
DP_SCALE = 1000.0    # normalizer for the DP turn-value feature

# feature vector = action_features(afterstate) + [bias, dp_value]
FEATURE_DIM = StateExtractor.action_feature_size() + 2

_WEIGHTS_PATH = Path(__file__).with_name("td_weights.pkl")
_WEIGHTS_VERSION = 2  # bumped: DP feature added, so old weights are incompatible


def td_features(state: GameState, action) -> list[float]:
    """Features for the TD model.

    The generic afterstate features come from game_state (the single home for
    position features); here we append only what's specific to this model: a bias
    term and the DP solver's own turn-value estimate of the action. The DP value is
    another algorithm's output, so it's added at this layer rather than in
    game_state (which must not depend on the solver).
    """
    features = StateExtractor.action_features(state, action)
    features.append(1.0)  # bias
    features.append(action_value(state, action) / DP_SCALE)  # DP turn-EV of the action
    return features


class LinearTD:
    """A linear value function trained by TD(0)."""

    def __init__(self, dim: int, weights: list[float] | None = None):
        self.w = list(weights) if weights is not None else [0.0] * dim

    def value(self, features: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, features))

    def update(self, features: list[float], target: float, alpha: float) -> float:
        """One TD step of the weights toward `target`; returns the TD error."""
        step = alpha * (target - self.value(features))
        for i, xi in enumerate(features):
            if xi:
                self.w[i] += step * xi
        return step / alpha if alpha else 0.0

    def save(self, path: Path = _WEIGHTS_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"version": _WEIGHTS_VERSION, "dim": len(self.w), "w": self.w}, f
            )

    @classmethod
    def load(cls, dim: int, path: Path = _WEIGHTS_PATH) -> "LinearTD":
        """Load weights if present and compatible, else a zero-initialized model."""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if data.get("version") == _WEIGHTS_VERSION and data.get("dim") == dim:
                    return cls(dim, data["w"])
            except (
                pickle.UnpicklingError,
                EOFError,
                OSError,
                KeyError,
                AttributeError,
            ):
                pass  # missing/corrupt/stale -> start from zeros
        return cls(dim)


# --- play-time model (lazily loaded) ---------------------------------------- #
_MODEL: "LinearTD | None" = None


def _model() -> LinearTD:
    global _MODEL
    if _MODEL is None:
        _MODEL = LinearTD.load(FEATURE_DIM)
    return _MODEL


def td_action_score(state: GameState, action) -> float:
    """Value the AI assigns to `action` (used by ai_evaluator's "td" algorithm)."""
    return _model().value(td_features(state, action))


# --- training --------------------------------------------------------------- #
def train(
    n_games: int = 5000,
    alpha: float = 0.01,
    epsilon: float = 0.1,
    seed: "int | None" = None,
    log_every: int = 500,
) -> LinearTD:
    """Self-play TD(0). Continues from saved weights if present; saves at the end."""
    import random

    import log
    from env import LangeStrasseEnv  # local import to avoid an import cycle

    log.VERBOSE = False  # silence game output during training
    if seed is not None:
        random.seed(seed)

    model = LinearTD.load(FEATURE_DIM)

    for game_i in range(1, n_games + 1):
        env = LangeStrasseEnv()
        last_features: dict[int, list[float]] = {}  # player -> its last chosen afterstate
        info = {}

        while not env.done:
            state = env.observe()
            player = env.current_player_idx
            actions = env.legal_actions()
            feats = [td_features(state, a) for a in actions]

            if random.random() < epsilon:
                choice = random.randrange(len(actions))
            else:
                choice = max(range(len(actions)), key=lambda j: model.value(feats[j]))
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

    model.save()
    global _MODEL
    _MODEL = model  # so same-process evaluation uses the freshly trained weights
    return model


def _evaluate(games: int = 300) -> None:
    """Quick arena: TD vs simple vs random, seats rotated to cancel first-move bias."""
    from collections import Counter

    import log
    import main

    log.VERBOSE = False
    main.VERBOSE = False

    matchup = ["td", "simple", "random"]
    wins: Counter = Counter()
    for g in range(games):
        seats = [matchup[(i + g) % 3] for i in range(3)]
        game = main.run_ai_game(seats)
        wins[seats[game.players.index(game.winner)]] += 1

    print(f"\nEval over {games} games (seats rotated):")
    for algo in matchup:
        print(f"  {algo:7s}: {wins[algo]:3d} ({wins[algo] / games:.0%})")


if __name__ == "__main__":
    import sys
    import time

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    print(f"Training linear TD by self-play for {n} games...")
    start = time.perf_counter()
    train(n)
    print(
        f"Done in {time.perf_counter() - start:.1f}s. Weights saved to {_WEIGHTS_PATH.name}."
    )
    _evaluate()
