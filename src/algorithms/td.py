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
from game_state import GameState, StateExtractor

MONEY_SCALE = 100.0  # final money (cents) is divided by this to form the TD target
DP_SCALE = 1000.0    # normalizer for the DP turn-value feature


# --- feature encodings ------------------------------------------------------ #
# Each model is an ordered list of afterstate atom names; game_state computes every
# atom once (see StateExtractor._atom_features), and a model just picks which atoms
# it reads. Adding a model to compare = a new key list + a VARIANTS entry; no new
# encoder, no duplicated normalization. (fmt: off keeps the visual grouping below.)

# fmt: off
# Full model: the rich encoding -- raw dice counts, grouped 1s/5s, derived scores,
# situation flags, all three players, the turn-over flag -- plus "dp_value", the DP
# solver's turn-value (an extra atom; see _EXTRA_ATOMS). "dp_value" is just a key, so
# add it to / drop it from any model's list to compare with vs. without the solver.
_TD_FULL_KEYS = [
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
    "dp_value",
]

# Minimal model: only atoms that can actually shift a linear afterstate argmax.
# Anything constant across a turn's actions (e.g. is_final_round) is useless to a
# linear value and left out. Money is a closed zero-sum economy, so the three
# balances span only two dimensions; this model keeps just my own.
_TD_MIN_KEYS = [
    "group1", "group2", "group3", "group4", "group5", "group6",
    "loose1", "loose5",
    "turn_accumulated",
    "score_me",
    "money_me",
    "ends_turn",
    "dp_value"
]
# Model with all info from Gamestate in vector representation
_TD_SMALL_KEYS = [
    "group1", "group2", "group3", "group4", "group5", "group6",
    "loose1", "loose5",

    "turn_accumulated",
    "roll_count",
    
    "seat_offset",

    "score_me", "strich_me", "money_me",
    "score_p2", "strich_p2", "money_p2",
    "score_p3", "strich_p3", "money_p3",

    "turn_number",
    "is_final_round",

    "ends_turn",
    "dp_value"
]
_TD_DP_KEYS = ["dp_value"]
# fmt: on


# Extra atoms that need the algorithm layer, so they can't be base atoms in
# game_state (which stays free of the solver, and would otherwise compute these for
# every model on every action). A model requests one just by listing its key, and it
# is computed only when some model does. "dp_value" is the DP solver's turn-value.
_EXTRA_ATOMS = {
    "dp_value": lambda state, action: action_value(state, action) / DP_SCALE,
}


def _encode(keys: list[str]):
    """Feature function for a model defined as an ordered list of atom keys.

    Keys are game_state afterstate atoms, optionally plus the solver-layer atoms in
    _EXTRA_ATOMS (e.g. "dp_value"). A trailing bias term is always added. Extra atoms
    are computed only when the key list asks for them.
    """
    keys = list(keys)
    extra_keys = [k for k in keys if k in _EXTRA_ATOMS]

    def features(state: GameState, action) -> list[float]:
        atoms = StateExtractor.afterstate_atoms(state, action)
        for k in extra_keys:
            atoms[k] = _EXTRA_ATOMS[k](state, action)
        f = [atoms[k] for k in keys]
        f.append(1.0)  # bias
        return f

    return features


# Full-model encoder, also exported as ``td_features`` for hand_eval / interp.
td_features = _encode(_TD_FULL_KEYS)


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

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"dim": len(self.w), "games": self.games_trained, "w": self.w}, f
            )

    @classmethod
    def load(cls, dim: int, path: Path) -> "LinearTD":
        """Load weights if present and compatible, else a zero-initialized model."""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if data.get("dim") == dim:
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
    """One TD model, fully derived from its name and key list: the feature encoder,
    the input dimension, and the on-disk weights file all follow from those two."""

    def __init__(self, name: str, keys: list[str]):
        self.name = name
        self.keys = list(keys)
        self.features = _encode(self.keys)  # (state, action) -> list[float]
        self.dim = len(self.keys) + 1  # + bias
        self.path = Path(__file__).with_name(f"{name}_weights.pkl")

    def load(self) -> LinearTD:
        return LinearTD.load(self.dim, self.path)


# The model registry: name -> ordered key list. This is the ONE place a model is
# defined -- its encoder, dimension, weights file ("<name>_weights.pkl") and its
# dispatch string all follow automatically. Add a line here and the model just works
# everywhere (config ALGOS, ai_player, interp); no other edits.
_MODEL_KEYS = {
    "td_full": _TD_FULL_KEYS,
    "td_small": _TD_SMALL_KEYS,
    "td_min": _TD_MIN_KEYS,
    "td_dp": _TD_DP_KEYS,
}

VARIANTS = {name: _Variant(name, keys) for name, keys in _MODEL_KEYS.items()}

# Every algorithm string that means "use a TD model" -- the variants plus the bare
# "td" alias. ai_player dispatches off this, so adding a variant above
# is all it takes to make it playable; no dispatch code needs to change.
TD_ALGORITHMS = {*VARIANTS}

# Kept for callers that predate the variants (e.g. hand_eval, interp): the full dim.
FEATURE_DIM = VARIANTS["td_full"].dim


# --- play-time models (lazily loaded, cached per variant) ------------------- #
_MODELS: dict[str, LinearTD] = {}


def _model(model_variant) -> LinearTD:
    v = VARIANTS[model_variant]
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


def td_action_score(state: GameState, action, variant: str) -> float:
    """Value the AI assigns to `action` (used by AIPlayer's td algorithms)."""
    v = VARIANTS[variant]
    return _model(variant).value(v.features(state, action))


# --- training --------------------------------------------------------------- #
def train(
    model_variant: str,
    alpha: float,
    epsilon: float,
    n_games: int,
    seed: "int | None" = None,
    log_every: int = 500,
) -> LinearTD:
    """Self-play TD(0) for one variant. Continues from saved weights; saves at end."""
    import random

    import log
    from env import LangeStrasseEnv  # local import to avoid an import cycle

    v = VARIANTS[model_variant]
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
                choice = max(range(len(actions)), key=lambda j: model.value(feats[j]))  # type: ignore
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
    model.save(v.path)
    _MODELS[v.name] = model  # so same-process evaluation uses the fresh weights
    return model


def _evaluate(model, games: int = 300) -> None:
    """Quick arena: this variant vs simple vs random."""
    import log
    import play

    log.VERBOSE = False

    matchup = [model, "simple", "random"]
    wins, _ = play.run_matchup(games, matchup)

    print(f"\nEval over {games} games:")
    for algo in matchup:
        print(f"  {algo:8s}: {wins[algo]:3d} ({wins[algo] / games:.0%})")


def run_training(model_variant, alpha, epsilon, n_games):
    import time

    print(f"Training {model_variant} by self-play for {n_games} games...")
    start = time.perf_counter()
    model = train(model_variant, alpha, epsilon, n_games)
    print(
        f"Done in {time.perf_counter() - start:.1f}s. "
        f"Model now trained on {model.games_trained} games total, "
        f"saved to {VARIANTS[model_variant].path.name}."
    )
    _evaluate(model_variant)
