"""Dynamic-programming solver for single-turn expected value.

The within-turn decision is a *single-agent stochastic optimal-stopping* problem
(no opponent inside a turn; the branching is the dice, so we take an expectation,
not a minimax). We compute the expected final turn score under optimal play.

State = (kept_config, prev):
  - ``kept_config``: the groups of dice kept in the *current* set, exactly as the
    game stores them. Carrying the full configuration (not just its score) is what
    lets the DP value the specials correctly:
      * extending a kept triplet -- a 4th of a kind doubles the group (via merge_kept
        + score_groups), an 8th... etc.;
      * completing a Talheim (three pairs, 500 / 1000 consecutive) -- which ends the turn;
      * completing a Lange Strasse (1250) across rolls.
  - ``prev``: points already banked from *previous* sets this turn (hot dice). It's
    at risk too, so it's carried to weigh stopping vs. gambling.

Why prev must be in the state: continuing can bust and lose *everything* at risk this
turn, so the stop/continue choice depends on the whole at-risk total. Passing prev in
means a bust branch returns 0 (you lose prev + current set), and

    stop     = prev + current_set_score
    continue = V(kept_config, prev)   # already accounts for busting it all away

so max(stop, continue) trades off correctly.
"""

import pickle
from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
from math import factorial
from pathlib import Path

from ai_player import ActionGenerator
from game.rules import flatten, merge_kept, score_groups, talheim_score

NUM_DICE = 6
STOP_MIN = 300      # minimum current-set score needed to bank a turn
T_MAX = 4000  # at/above this we assume optimal play banks (keeps the DAG finite)

# --- persisted value cache -------------------------------------------------- #
# Building the table takes a few seconds, so we pickle it and start warm next run.
# The header records everything the values depend on; a mismatch (rules or params
# changed) means the file is ignored and rebuilt rather than silently trusted.
_CACHE: dict[tuple, float] = {}
_CACHE_PATH = Path(__file__).with_name("turn_value_cache.pkl")
_CACHE_VERSION = 1  # bump when a scoring-rule change affects the computed values
_ready = False


def _header() -> tuple:
    return (_CACHE_VERSION, NUM_DICE, STOP_MIN, T_MAX)


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
def _legal_keeps(
    roll: tuple[int, ...], kept_values: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Legal keeps of `roll` (as value-tuples) given the already-kept values.

    Reuses the game's own enumeration so the DP can never disagree with the rules.
    Depends only on the kept *values* (not their grouping), so it caches broadly.
    Empty iff the roll busts.
    """
    combos = ActionGenerator._valid_combinations(list(roll), list(kept_values))
    return tuple(tuple(combo) for combo in combos)


def _canon(kept_groups) -> tuple[tuple[int, ...], ...]:
    """Order-independent, hashable key for a kept configuration."""
    return tuple(sorted(tuple(sorted(group)) for group in kept_groups))


def continue_value(kept_key: tuple[tuple[int, ...], ...], prev: int) -> float:
    """Expected final turn score from rolling on with config `kept_key` and `prev` banked."""
    total = prev + score_groups([list(group) for group in kept_key])
    if total >= T_MAX:
        # Boundary: with this much at stake optimal play banks. Returning the total
        # (instead of recursing) keeps the state space finite; such states are reached
        # with negligible probability, so the effect is tiny.
        return float(total)
    return _value(kept_key, prev)


def _value(kept_key: tuple[tuple[int, ...], ...], prev: int) -> float:
    """Expected final turn score: about to roll the remaining dice from this state.

    Memoized in the module-level _CACHE dict (which is what we pickle). The state
    graph is a DAG ordered by total-at-risk, so there are no cycles and storing the
    result after computing it is safe.
    """
    memo_key = (kept_key, prev)
    cached = _CACHE.get(memo_key)
    if cached is not None:
        return cached

    kept_groups = [list(group) for group in kept_key]
    kept_values = tuple(sorted(flatten(kept_groups)))
    n = NUM_DICE - len(kept_values)

    ev = 0.0
    for roll, prob in _roll_distribution(n):
        keeps = _legal_keeps(roll, kept_values)
        if not keeps:
            best = 0.0  # bust -> lose everything at risk
        else:
            best = max(_keep_value(kept_groups, prev, keep) for keep in keeps)
        ev += prob * best

    _CACHE[memo_key] = ev
    return ev


def _keep_value(kept_groups, prev: int, keep: tuple[int, ...]) -> float:
    """Value of keeping `keep`, then playing on optimally."""
    new_kept = merge_kept(kept_groups, keep)
    values = flatten(new_kept)
    total = prev + score_groups(new_kept)

    # Talheim ends the turn on the spot (banked, no choice to continue).
    if talheim_score(values) > 0:
        return float(total)

    if len(values) == NUM_DICE:
        # Hot dice (includes a Lange Strasse): all six kept -> roll a fresh six.
        return continue_value((), total)

    set_score = total - prev
    stop_value = float(total) if set_score >= STOP_MIN else float("-inf")
    return max(stop_value, continue_value(_canon(new_kept), prev))


def action_value(state, action) -> float:
    """DP value of taking `action` in `state` (higher is better).

    ``state`` is a GameState, ``action`` an Action. Uses merge_kept so keeping a die
    that extends an existing triplet is scored as the doubled group, matching the game.
    """
    ensure_ready()
    new_kept = merge_kept(state.kept_groups, action.dice_to_keep)
    values = flatten(new_kept)
    prev = state.turn_accumulated_score
    total = prev + score_groups(new_kept)

    # Stopping banks the total; a Talheim also ends the turn at the total.
    if action.stop_after or talheim_score(values) > 0:
        return float(total)

    if len(values) == NUM_DICE:
        return continue_value((), total)  # hot dice / Lange Strasse
    return continue_value(_canon(new_kept), prev)


def precompute() -> None:
    """Fill the value cache from the start of a turn (every reachable state)."""
    continue_value((), 0)


def load_cache(path: Path = _CACHE_PATH) -> bool:
    """Populate the in-memory cache from `path`. Returns False if missing or stale."""
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError):
        return False  # corrupt/partial file -> rebuild
    if not isinstance(data, dict) or data.get("header") != _header():
        return False  # rules/params changed -> rebuild
    _CACHE.update(data["values"])
    return True


def save_cache(path: Path = _CACHE_PATH) -> None:
    """Write the current in-memory cache to `path`."""
    with open(path, "wb") as f:
        pickle.dump({"header": _header(), "values": _CACHE}, f)


def ensure_ready() -> None:
    """Ensure the value cache is loaded, once. Load from disk, else build and save.

    Cheap to call on every decision: after the first call it's just a flag check.
    """
    global _ready
    if _ready:
        return
    _ready = True
    if not load_cache():
        precompute()
        save_cache()
