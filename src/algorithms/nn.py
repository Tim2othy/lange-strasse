"""Multi-head MLP over afterstates: payout-event probabilities -> expected money.

Instead of one output regressing the noisy money total, the net predicts the
probabilities of the payout events themselves -- my final placement, win by
round 10, no strich, under 5000 -- and expected money is assembled from them
analytically with the payout table (exact game knowledge, nothing learned).
Each head is a dense, low-variance target: every game labels every one of
them. The in-game money flows (Strasse, Totale, Super Strasse) are the one
part that stays a regression, as a separate "flow" head.

Money is zero-sum and every payout event moves a FIXED amount, so the acting
player's expected money is *linear* in these event probabilities -- the
analytic combination is exact, not an approximation. Two consequences shape
the head list: my own placement alone fixes my share of the basic stakes
(1st collects 50+70, 2nd pays 50, 3rd pays 70), so no opponent-placement
heads are needed; and the only correlated events whose product matters are
"I win AND an opponent is under 5000", which get their own joint heads.

The DP solver's turn-value enters twice: as an input feature, and as a linear
residual (``w_res * dp_value``) on the final value. The head layer starts at
zero, so every head is initially constant across actions and the net ranks
actions exactly like the "dp" algorithm. The residual is trained against the
TOTAL money error, so as the heads learn to explain the money themselves,
``w_res`` anneals toward zero on its own.

Training is Monte Carlo self-play: every decision of a game is labeled with
that game's final events (cross-entropy) and the realized future in-game
money (squared error).

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
ARCH = "multihead-v1"  # bumped whenever the head layout changes

nn_features = _encode(NN_KEYS)
DIM = len(NN_KEYS) + 1  # _encode appends a constant bias input
DP_IDX = NN_KEYS.index("dp_value")

# --- Head layout: 12 probability outputs + 1 regression output. ------------- #
# All from the acting player's perspective (me / p2 / p3 as in the features).
# The order is load-bearing: PAYOUT below is aligned with it.
HEAD_NAMES = [
    "place1",
    "place2",
    "place3",  # softmax: my final placement
    "by10_me",
    "by10_p2",
    "by10_p3",  # sigmoid: X wins by round 10
    "nostrich_me",
    "nostrich_p2",
    "nostrich_p3",  # sigmoid: X ends without strich
    "below5k_me",  # sigmoid: I end under 5000
    "iwin_p2_below",
    "iwin_p3_below",  # sigmoid: I win AND opp under 5000
    "flow",  # linear: my future in-game money
]
N_PROB = 12
FLOW_IDX = 12
N_OUT = 13

# Expected-money weight of each probability head, in MONEY_SCALE units
# (cents / 100). Mirrors Game.finish_game exactly. Zero-sum is baked in:
# each coefficient is MY side of one fixed transfer.
# fmt: off
PAYOUT = np.array([
    1.2, -0.5, -0.7,    # 1st collects 50 + 70; 2nd pays 50; 3rd pays 70
    1.0, -0.5, -0.5,    # round-10 winner collects 50 from each other player
    1.0, -0.5, -0.5,    # a no-strich player collects 50 from each other player
    -0.5,               # under 5000 at the end -> I pay the winner 50
    0.5, 0.5,           # I'm the winner: collect 50 per under-5000 opponent
])
# fmt: on


class MLP:
    """Shared tanh hidden layer feeding the event heads (softmax placement,
    sigmoid event flags, linear flow) plus a ``w_res * dp_value`` residual.
    The head layer (W2, b2) is zero-initialized, so at first every head is
    constant across actions and ranking falls back to pure DP; the residual's
    weights train against the total money error and fade as the heads take
    over. Plain SGD throughout.
    """

    def __init__(self, dim: int = DIM, hidden: int = HIDDEN, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, dim**-0.5, (hidden, dim))
        self.b1 = np.zeros(hidden)
        self.w2 = np.zeros((N_OUT, hidden))
        self.b2 = np.zeros(N_OUT)
        self.w_res = 1.0  # coefficient on dp_value -- starts as a pure pass-through
        self.b_res = 0.0
        self.games_trained = 0  # cumulative self-play games behind these weights

    def _forward(self, x: np.ndarray):
        """-> (hidden activations, raw head outputs, head probabilities)."""
        h = np.tanh(self.W1 @ x + self.b1)
        z = self.w2 @ h + self.b2
        p = np.empty(N_PROB)
        e = np.exp(z[:3] - z[:3].max())
        p[:3] = e / e.sum()  # softmax over my placement
        p[3:] = 1.0 / (1.0 + np.exp(-z[3:N_PROB]))  # sigmoid event flags
        return h, z, p

    def value(self, features: list[float]) -> float:
        x = np.asarray(features)
        _h, z, p = self._forward(x)
        return float(self.w_res * x[DP_IDX] + self.b_res + PAYOUT @ p + z[FLOW_IDX])

    def values(self, feature_rows: list[list[float]]) -> np.ndarray:
        """Values of many feature vectors at once (one action per row)."""
        X = np.asarray(feature_rows)
        H = np.tanh(X @ self.W1.T + self.b1)
        Z = H @ self.w2.T + self.b2
        P = np.empty((X.shape[0], N_PROB))
        E = np.exp(Z[:, :3] - Z[:, :3].max(axis=1, keepdims=True))
        P[:, :3] = E / E.sum(axis=1, keepdims=True)
        P[:, 3:] = 1.0 / (1.0 + np.exp(-Z[:, 3:N_PROB]))
        return self.w_res * X[:, DP_IDX] + self.b_res + P @ PAYOUT + Z[:, FLOW_IDX]

    def head_probs(self, features: list[float]) -> dict[str, float]:
        """Named head outputs for one afterstate (probabilities + flow)."""
        x = np.asarray(features)
        _h, z, p = self._forward(x)
        return dict(zip(HEAD_NAMES, [*p, z[FLOW_IDX]]))

    def update(
        self,
        features: list[float],
        y_probs: np.ndarray,
        y_flow: float,
        y_total: float,
        alpha: float,
    ) -> float:
        """One SGD step: probability heads toward their event labels (cross-
        entropy), the flow head toward the realized in-game money (squared
        error), and the DP residual toward the total money error. Returns the
        total-value error."""
        x = np.asarray(features)
        h, z, p = self._forward(x)
        value = float(self.w_res * x[DP_IDX] + self.b_res + PAYOUT @ p + z[FLOW_IDX])

        # d(-loss)/dz is "target - prediction" for every head: softmax and
        # sigmoid cross-entropy and linear squared error all share that form.
        err = np.empty(N_OUT)
        err[:N_PROB] = y_probs - p
        err[FLOW_IDX] = y_flow - z[FLOW_IDX]

        dh = (self.w2.T @ err) * (1.0 - h * h)  # taken before W2 moves
        self.w2 += alpha * np.outer(err, h)
        self.b2 += alpha * err
        self.W1 += alpha * np.outer(dh, x)
        self.b1 += alpha * dh

        # The DP prior explains whatever money the heads don't (yet); as they
        # learn, this error turns against the prior and anneals it away.
        err_total = y_total - value
        self.w_res += alpha * err_total * x[DP_IDX]
        self.b_res += alpha * err_total
        return err_total

    def save(self, path: Path = WEIGHTS_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "keys": NN_KEYS,
                    "arch": ARCH,
                    "hidden": len(self.b1),
                    "games": self.games_trained,
                    "W1": self.W1,
                    "b1": self.b1,
                    "W2": self.w2,
                    "b2": self.b2,
                    "w_res": self.w_res,
                    "b_res": self.b_res,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> "MLP":
        """Load weights if present and same keys + head layout, else fresh."""
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if data.get("keys") == NN_KEYS and data.get("arch") == ARCH:
                    model = cls(hidden=data["hidden"])
                    model.W1, model.b1 = data["W1"], data["b1"]
                    model.w2, model.b2 = data["W2"], data["b2"]
                    model.w_res = data["w_res"]
                    model.b_res = data["b_res"]
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


def nn_head_report(state: GameState, action) -> dict[str, float]:
    """Debug view: every head's prediction for one action's afterstate."""
    return _model().head_probs(nn_features(state, action))


# --- game-end labels --------------------------------------------------------- #
def _final_settlement(
    scores: list[int], has_strich: list[bool], turn_number: int
) -> list[int]:
    """End-of-game transfers per seat, in cents -- mirrors Game.finish_game
    (including its tie-breaking: stable sort by score, seat order on ties)."""
    n = len(scores)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    winner = ranked[0]
    settle = [0] * n
    settle[ranked[0]] += 120  # collects 50 from 2nd and 70 from 3rd
    settle[ranked[1]] -= 50
    settle[ranked[2]] -= 70
    if turn_number <= 10:
        for i in range(n):
            settle[i] += 100 if i == winner else -50
    for j in range(n):
        if not has_strich[j]:
            for i in range(n):
                settle[i] += 100 if i == j else -50
    for j in range(n):
        if j != winner and scores[j] < 5000:
            settle[j] -= 50
            settle[winner] += 50
    return settle


def _event_targets(
    seat: int, scores: list[int], has_strich: list[bool], turn_number: int
) -> np.ndarray:
    """Cross-entropy targets for `seat`'s probability heads, in that seat's
    me/p2/p3 perspective (matching the feature encoding)."""
    n = len(scores)
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    winner = ranked[0]
    by10 = turn_number <= 10
    order = [(seat + i) % n for i in range(n)]  # me, p2, p3 as absolute seats

    y = np.zeros(N_PROB)
    y[ranked.index(seat)] = 1.0  # one-hot own placement
    for k, s in enumerate(order):
        y[3 + k] = 1.0 if (by10 and s == winner) else 0.0
        y[6 + k] = 0.0 if has_strich[s] else 1.0
    y[9] = 1.0 if scores[seat] < 5000 else 0.0
    y[10] = 1.0 if (seat == winner and scores[order[1]] < 5000) else 0.0
    y[11] = 1.0 if (seat == winner and scores[order[2]] < 5000) else 0.0
    return y


# --- training --------------------------------------------------------------- #
def train(
    alpha: float,
    epsilon: float,
    n_games: int,
    seed: "int | None" = None,
    log_every: int = 500,
) -> MLP:
    """Monte Carlo self-play: play a game epsilon-greedily, then label every
    decision with the game's final events and realized future in-game money.
    Continues from saved weights; saves at end."""
    import random

    import log
    from env import LangeStrasseEnv  # local import to avoid an import cycle

    log.VERBOSE = False  # silence game output during training
    if seed is not None:
        random.seed(seed)

    model = MLP.load()

    for game_i in range(1, n_games + 1):
        env = LangeStrasseEnv()
        # seat -> [(chosen afterstate features, balance at that decision), ...]
        history: dict[int, list[tuple[list[float], int]]] = {}
        info = {}

        while not env.done:
            state = env.observe()
            player = env.current_player_idx
            money_now = state.players[player].money
            actions = env.legal_actions()
            feats = [nn_features(state, a) for a in actions]

            if random.random() < epsilon:
                choice = random.randrange(len(actions))
            else:
                choice = int(np.argmax(model.values(feats)))
            history.setdefault(player, []).append((feats[choice], money_now))

            _obs, _reward, _done, info = env.step(actions[choice])

        standings = info["standings"]
        turn_number = info["final_turn_number"]
        scores = [s["score"] for s in standings]
        has_strich = [s["has_strich"] for s in standings]
        settle = _final_settlement(scores, has_strich, turn_number)
        assert sum(settle) == 0, "settlement must be zero-sum"

        for seat, decisions in history.items():
            y_probs = _event_targets(seat, scores, has_strich, turn_number)
            final = standings[seat]["money"]
            for feats_row, money_then in decisions:
                # in-game flow still to come = total still to come - settlement
                y_flow = (final - settle[seat] - money_then) / MONEY_SCALE
                y_total = (final - money_then) / MONEY_SCALE
                model.update(feats_row, y_probs, y_flow, y_total, alpha)

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
