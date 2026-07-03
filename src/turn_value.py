"""Dynamic-programming solver for single-turn expected value.

The within-turn decision is a *single-agent stochastic optimal-stopping* problem
(there is no opponent inside a turn; the "branching" is the dice, so we take an
expectation, not a minimax). We compute

    V(n, t) = expected final turn score when you are about to roll `n` dice with
              `t` points already at risk this turn (lost entirely if you bust).

Why the state is (dice_left, total_at_risk) and nothing else:
    When you continue, with some probability you bust and lose *everything* banked
    this turn -- so the decision to stop vs. continue genuinely depends on how much
    is at risk. By carrying the full at-risk total `t` (accumulated previous sets +
    current set), a bust branch returns 0 (you lose all of `t`), and

        stop value     = t            (bank it)
        continue value = V(n, t)      (already accounts for busting away t)

    so max(stop, continue) does the right trade-off automatically. That resolves
    the "but the 1000 I already have must be in the bust probabilities" problem:
    it is, because we pass it in as `t`.

Deliberate approximations (strong-but-simple v1; refine later):
  - The exact kept configuration isn't part of the state, so cross-roll
    triplet-extension and Lange Strasse / Talheim *building across rolls* are not
    modelled. Specials that land within a single roll are still scored.
  - Stop-eligibility uses `t >= STOP_MIN`. That's exact for the first set (where
    total == current-set score) and slightly permissive in later sets. At runtime
    it doesn't matter for legality anyway: the game only offers legal stop actions.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial

from ai_actions import ActionGenerator
from scoring import flatten, score_groups

NUM_DICE = 6
STOP_MIN = 300      # minimum current-set score needed to bank a turn
T_STEP = 50         # every score in the game is a multiple of 50
T_MAX = 4000        # at/above this we assume optimal play banks (keeps recursion finite)


@lru_cache(maxsize=None)
def _roll_distribution(n: int) -> tuple[tuple[tuple[int, ...], float], ...]:
    """All distinct outcomes of rolling `n` fair dice as (roll_multiset, probability)."""
    outcomes = []
    total = 6**n
    for roll in combinations_with_replacement(range(1, 7), n):
        ways = factorial(n)
        for count in Counter(roll).values():
            ways //= factorial(count)
        outcomes.append((roll, ways / total))
    return tuple(outcomes)


@lru_cache(maxsize=None)
def _keep_options(roll: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Legal keeps of `roll` as (gain, dice_used). Empty tuple iff the roll busts.

    Reuses the game's own keep-enumeration so the DP can never disagree with the
    real rules. `kept_values` is empty because this v1 scores each roll on its own
    (see the module docstring on ignored cross-roll building).
    """
    options = []
    for combo in ActionGenerator._valid_combinations(list(roll), []):
        options.append((score_groups([combo]), len(combo)))
    return tuple(options)


def continue_value(n: int, t: int) -> float:
    """Expected final turn score from choosing to roll `n` dice with `t` at risk."""
    if t >= T_MAX:
        # Boundary: with this much at stake, optimal play banks. Returning `t`
        # (rather than recursing further) keeps the state space finite; states this
        # high are reached with negligible probability, so the effect is tiny.
        return float(t)
    return _value(n, t)


@lru_cache(maxsize=None)
def _value(n: int, t: int) -> float:
    """Expected final turn score: about to roll `n` dice with `t` at risk."""
    ev = 0.0
    for roll, prob in _roll_distribution(n):
        options = _keep_options(roll)
        if not options:
            best = 0.0  # bust -> lose everything at risk
        else:
            best = max(_keep_value(n, t, gain, used) for gain, used in options)
        ev += prob * best
    return ev


def _keep_value(n: int, t: int, gain: int, used: int) -> float:
    """Value of keeping `used` dice for `gain`, then playing on optimally."""
    t2 = t + gain
    remaining = n - used
    if remaining == 0:
        # Hot dice: all dice kept -> must roll a fresh 6 (stopping isn't allowed).
        return continue_value(NUM_DICE, t2)
    stop_value = float(t2) if t2 >= STOP_MIN else float("-inf")
    return max(stop_value, continue_value(remaining, t2))


def action_value(state, action) -> float:
    """DP value of taking `action` in `state` (higher is better).

    ``state`` is a GameState, ``action`` an Action. Stopping banks the exact total;
    continuing looks up the DP continuation value of the resulting (dice, total) state.
    """
    new_kept = state.kept_groups + [list(action.dice_to_keep)]
    total_at_risk = state.turn_accumulated_score + score_groups(new_kept)

    if action.stop_after:
        return float(total_at_risk)

    dice_used = len(flatten(new_kept))
    remaining = NUM_DICE - dice_used
    n = NUM_DICE if remaining == 0 else remaining  # all 6 kept -> hot dice, fresh 6
    return continue_value(n, total_at_risk)


def precompute() -> None:
    """Eagerly fill the value cache (optional; otherwise it fills lazily on first use)."""
    for t in range(T_MAX - T_STEP, -1, -T_STEP):
        for n in range(1, NUM_DICE + 1):
            _value(n, t)
