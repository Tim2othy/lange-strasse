"""Linear TD(0) value functions over between-turn states, learned by self-play.

Estimates ``V(afterstate) = w . features(afterstate)`` -- the acting player's expected
FINAL MONEY. We use *afterstates* (the position right after a move, before the next
roll), TD-Gammon style: the value implicitly averages over the future dice, so we just
pick the action whose afterstate is worth most. Afterstates also dodge a trap of a
plain linear ``Q(s, a)``: with additive state+action features the state term is the
same for every action and cancels in the argmax, so the state couldn't influence the
choice. An afterstate value can, because each action lands in a different position.

Everything is from the *acting player's* perspective (the encoders order that player
first), so ``V`` consistently means "my expected final money".

Reward is terminal only: final money / MONEY_SCALE, no discount (episodic). Every
intermediate money flow (Strasse, Totale, end-game settlement) is already part of the
final total, so a terminal reward is complete; TD bootstrapping carries the credit
back through the turns.

Variants share all of this and differ only in which afterstate atoms they encode
(see the TD_*_KEYS lists below and game_state._atom_features):

  * ``td_full``  -- the rich encoding (raw dice counts, derived scores, completion
    flags, all players) plus the DP solver's turn-value as an extra feature.
  * ``td_small`` -- a deliberately raw encoding: triplet sizes, loose 1s/5s, raw
    scalars, no flags, no DP hint. "How far does raw data get?".
  * ``td_min``   -- only atoms that can actually shift a linear afterstate argmax.

Train one with:  python -m algorithms.td [td_full|td_small|td_min] [n_games]
"""

import pickle
from pathlib import Path

from algorithms.dp import action_value
from config import TD_ALPHA, TD_EPSILON, TD_TRAIN_GAMES, WEIGHTS_VERSION
from game_state import GameState, StateExtractor

MONEY_SCALE = 100.0  # final money (cents) is divided by this to form the TD target
DP_SCALE = 1000.0    # normalizer for the DP turn-value feature


# --- feature encodings ------------------------------------------------------ #
# Each model is an ordered list of afterstate atom names; game_state computes every
# atom once (see StateExtractor._atom_features), and a model just picks which atoms
# it reads. Adding a model to compare = a new key list + a VARIANTS entry; no new
# encoder, no duplicated normalization. (fmt: off keeps the visual grouping below.)

# fmt: off
# Full model: byte-for-byte the original rich layout (raw dice counts, grouped
# 1s/5s, derived scores, situation flags, all three players, turn-over flag), so a
# previously trained td_weights.pkl still applies. td_features also appends the DP
# solver's turn-value.
_TD_FULL_KEYS = [
    "avail1", "avail2", "avail3", "avail4", "avail5", "avail6",
    "kept1", "kept2", "kept3", "kept4", "kept5", "kept6",
    "grouped1", "grouped5",
    "current_set_score", "turn_accumulated", "total_turn_score", "roll_count",
    "dice_left",
    "flag_lange_strasse", "flag_talheim", "flag_keep_any",
    "score_me", "strich_me", "money_me",
    "score_p2", "strich_p2", "money_p2",
    "score_p3", "strich_p3", "money_p3",
    "turn_number", "is_final_round",
    "ends_turn",
]

# Raw model: triplet sizes + loose 1s/5s, raw turn scalars, seat offset, all three
# players. No derived scores, no flags, no DP hint.
_TD_SMALL_KEYS = [
    "group1", "group2", "group3", "group4", "group5", "group6",
    "loose1", "loose5",
    "turn_accumulated", "roll_count", "seat_offset", "turn_number", "is_final_round",
    "score_me", "strich_me", "money_me",
    "score_p2", "strich_p2", "money_p2",
    "score_p3", "strich_p3", "money_p3",
    "ends_turn",
]

# Minimal model: only atoms that can actually shift a linear afterstate argmax.
# Anything constant across a turn's actions (e.g. is_final_round) is useless to a
# linear value and left out. Money is a closed zero-sum economy, so the three
# balances span only two dimensions; this model keeps just my own.
TD_MIN_KEYS = [
    "group1", "group2", "group3", "group4", "group5", "group6",
    "loose1", "loose5",
    "turn_accumulated",
    "score_me",
    "money_me",
    "ends_turn",
]
# fmt: on


def _keys_encoder(keys: list[str]):
    """A variant feature function: the named afterstate atoms plus a bias term.
    Defining a hand-picked model is then a one-line VARIANTS entry."""

    def features(state: GameState, action) -> list[float]:
        f = StateExtractor.select_features(state, action, keys)
        f.append(1.0)  # bias
        return f

    return features


def td_features(state: GameState, action) -> list[float]:
    """Full-model features: the _TD_FULL_KEYS atoms, a bias term, and the DP solver's
    own turn-value estimate (an external signal, so it's appended here rather than
    living in game_state)."""
    f = StateExtractor.select_features(state, action, _TD_FULL_KEYS)
    f.append(1.0)  # bias
    f.append(action_value(state, action) / DP_SCALE)  # DP turn-EV of the action
    return f


class LinearTD:
    """A linear value function trained by TD(0)."""

    def __init__(
        self, dim: int, weights: list[float] | None = None, games_trained: int = 0
    ):
        self.w = list(weights) if weights is not None else [0.0] * dim
        self.games_trained = (
            games_trained  # cumulative self-play games behind these weights
        )

    def value(self, features: list[float]) -> float:
        return sum(wi * xi for wi, xi in zip(self.w, features))

    def update(self, features: list[float], target: float, alpha: float) -> float:
        """One TD step of the weights toward `target`; returns the TD error."""
        step = alpha * (target - self.value(features))
        for i, xi in enumerate(features):
            if xi:
                self.w[i] += step * xi
        return step / alpha if alpha else 0.0

    def save(self, path: Path, version: int) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "version": version,
                    "dim": len(self.w),
                    "games": self.games_trained,
                    "w": self.w,
                },
                f,
            )

    @classmethod
    def load(cls, dim: int, path: Path, version: int) -> "LinearTD":
        """Load weights if present and compatible, else a zero-initialized model."""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if data.get("version") == version and data.get("dim") == dim:
                    return cls(dim, data["w"], data.get("games", 0))
            except (
                pickle.UnpicklingError,
                EOFError,
                OSError,
                KeyError,
                AttributeError,
            ):
                pass  # missing/corrupt/stale -> start from zeros
        return cls(dim)


# --- model variants --------------------------------------------------------- #
class _Variant:
    """One TD model: its feature encoder and where its weights live on disk."""

    def __init__(self, name, features, dim, filename, version):
        self.name = name
        self.features = features  # (state, action) -> list[float]
        self.dim = dim
        self.path = Path(__file__).with_name(filename)
        self.version = version

    def load(self) -> LinearTD:
        return LinearTD.load(self.dim, self.path, self.version)


VARIANTS = {
    "td_full": _Variant(
        "td_full",
        td_features,
        len(_TD_FULL_KEYS) + 2,  # + bias + DP value
        "td_weights.pkl",
        WEIGHTS_VERSION,
    ),
    "td_small": _Variant(
        "td_small",
        _keys_encoder(_TD_SMALL_KEYS),
        len(_TD_SMALL_KEYS) + 1,  # + bias
        "td_small_weights.pkl",
        1,
    ),
    "td_min": _Variant(
        "td_min",
        _keys_encoder(TD_MIN_KEYS),
        len(TD_MIN_KEYS) + 1,  # + bias
        "td_min_weights.pkl",
        1,
    ),
}
_DEFAULT_VARIANT = "td_full"

# Every algorithm string that means "use a TD model" -- the variants plus the bare
# "td" alias. ai_evaluator/ai_player dispatch off this, so adding a variant above
# is all it takes to make it playable; no dispatch code needs to change.
TD_ALGORITHMS = {"td", *VARIANTS}

# Kept for callers that predate the variants (e.g. hand_eval, interp): the full dim.
FEATURE_DIM = VARIANTS["td_full"].dim


def _resolve(name: str) -> _Variant:
    """Map an algorithm string to a variant; bare ``"td"`` means the full model."""
    if name == "td":
        name = "td_full"
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown TD variant: {name!r} (choose from {list(VARIANTS)})")


# --- play-time models (lazily loaded, cached per variant) ------------------- #
_MODELS: dict[str, LinearTD] = {}


def _model(variant: str = _DEFAULT_VARIANT) -> LinearTD:
    v = _resolve(variant)
    if v.name not in _MODELS:
        model = v.load()
        if model.games_trained:
            print(
                f"[{v.name}] using weights trained on "
                f"{model.games_trained} self-play games"
            )
        else:
            print(
                f"[{v.name}] no trained weights -- playing untrained "
                f"(run: python -m algorithms.td {v.name})"
            )
        _MODELS[v.name] = model
    return _MODELS[v.name]


def td_action_score(state: GameState, action, variant: str = _DEFAULT_VARIANT) -> float:
    """Value the AI assigns to `action` (used by ai_evaluator's td algorithms)."""
    v = _resolve(variant)
    return _model(v.name).value(v.features(state, action))


# --- training --------------------------------------------------------------- #
def train(
    variant: str = _DEFAULT_VARIANT,
    n_games: int = TD_TRAIN_GAMES,
    alpha: float = TD_ALPHA,
    epsilon: float = TD_EPSILON,
    seed: "int | None" = None,
    log_every: int = 500,
) -> LinearTD:
    """Self-play TD(0) for one variant. Continues from saved weights; saves at end."""
    import random

    import log
    from env import LangeStrasseEnv  # local import to avoid an import cycle

    v = _resolve(variant)
    log.VERBOSE = False  # silence game output during training
    if seed is not None:
        random.seed(seed)

    model = v.load()

    for game_i in range(1, n_games + 1):
        env = LangeStrasseEnv()
        last_features: dict[int, list[float]] = {}  # player -> its last chosen afterstate
        info = {}

        while not env.done:
            state = env.observe()
            player = env.current_player_idx
            actions = env.legal_actions()
            feats = [v.features(state, a) for a in actions]

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

    model.games_trained += n_games
    model.save(v.path, v.version)
    _MODELS[v.name] = model  # so same-process evaluation uses the fresh weights
    return model


def _evaluate(variant: str = _DEFAULT_VARIANT, games: int = 300) -> None:
    """Quick arena: this variant vs simple vs random, seats rotated to cancel first-move bias."""
    from collections import Counter

    import log
    import main

    log.VERBOSE = False
    main.VERBOSE = False

    matchup = [variant, "simple", "random"]
    wins: Counter = Counter()
    for g in range(games):
        seats = [matchup[(i + g) % 3] for i in range(3)]
        game = main.run_ai_game(seats)
        wins[seats[game.players.index(game.winner)]] += 1

    print(f"\nEval over {games} games (seats rotated):")
    for algo in matchup:
        print(f"  {algo:8s}: {wins[algo]:3d} ({wins[algo] / games:.0%})")


if __name__ == "__main__":
    import sys
    import time

    variant = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_VARIANT
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else TD_TRAIN_GAMES
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")

    print(f"Training {variant} by self-play for {n_games} games...")
    start = time.perf_counter()
    model = train(variant, n_games)
    print(
        f"Done in {time.perf_counter() - start:.1f}s. "
        f"Model now trained on {model.games_trained} games total, "
        f"saved to {VARIANTS[variant].path.name}."
    )
    _evaluate(variant)
