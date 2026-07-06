"""The rules of Lange Strasse: scoring, keep-validity, and action enumeration.

Single source of truth for the rules, as pure functions over dice values and
kept groups -- no game state lives here. The game, the DP solver, and the ML
feature layer all call these same functions, so they can never disagree.
"""

import itertools
from collections import Counter
from dataclasses import dataclass

NUM_DICE = 6
STOP_MIN = 300
# Face values that make up a Lange Strasse (1-2-3-4-5-6).
LANGE_STRASSE = frozenset({1, 2, 3, 4, 5, 6})
LANGE_STRASSE_SCORE = 1250

# Points for a single die kept on its own (only 1s and 5s can be kept individually).
INDIVIDUAL_SCORE = {1: 100, 5: 50}
# Base score for three of a kind (each extra die doubles it). Others: value * 100.
TRIPLET_BASE = {1: 1000, 5: 500}


def flatten(groups):
    """Flatten a list of dice groups into a single list of values."""
    return [value for group in groups for value in group]


def is_lange_strasse(values):
    """True if ``values`` contains at least one of every face 1-6."""
    return LANGE_STRASSE.issubset(values)


def talheim_score(values):
    """Score for a Talheim (three pairs across exactly 6 dice); 0 if it isn't one.

    1000 points if the three pairs are consecutive, otherwise 500.
    """
    if len(values) != NUM_DICE:
        return 0
    counts = Counter(values)
    if len(counts) != 3 or any(count != 2 for count in counts.values()):
        return 0
    low, mid, high = sorted(counts)
    return 1000 if (mid == low + 1 and high == mid + 1) else 500


def merge_kept(kept_groups, new_values):
    """Add ``new_values`` to ``kept_groups`` and return the resulting groups (inputs
    are not mutated).

    A newly kept die extends an existing same-face triplet when there is one -- so a
    4th of a kind lands in the group and doubles its score -- otherwise three-or-more
    of a face form their own group and stray 1s/5s stay individual. This is the single
    source of truth for how kept dice are grouped; the game and the DP solver both use it.
    """
    groups = [list(group) for group in kept_groups]
    for value, count in Counter(new_values).items():
        for group in groups:
            # Only merge into an existing triplet+ of this same value.
            if len(group) >= 3 and len(set(group)) == 1 and group[0] == value:
                group.extend([value] * count)
                break
        else:
            if count >= 3:
                groups.append([value] * count)  # new triplet+
            else:
                groups.extend([value] for _ in range(count))  # individual 1s/5s
    return groups


def score_groups(kept_groups):
    """Total score for kept dice groups, including Talheim / Lange Strasse.

    Both specials need all six dice, so for fewer than six kept dice this is
    the plain triplet/individual score (the one the STOP_MIN check cares about).
    """
    values = flatten(kept_groups)

    talheim = talheim_score(values)
    if talheim:
        return talheim
    if is_lange_strasse(values):
        return LANGE_STRASSE_SCORE

    total = 0
    for group in kept_groups:
        for value, count in Counter(group).items():
            if count >= 3:
                base = TRIPLET_BASE.get(value, value * 100)
                total += base * 2 ** (count - 3)
            else:
                total += count * INDIVIDUAL_SCORE.get(value, 0)
    return total


def is_valid_keep(dice_values, already_kept=None):
    """Return ``(is_valid, error_message)`` for keeping ``dice_values``.

    Rules:
    - Any number of 1s or 5s can be kept.
    - A group of 3+ identical dice can be kept.
    - If 3+ of a value are already kept, more of it can be added.
    - Completing a Lange Strasse or Talheim is always allowed.
    """
    if not dice_values:
        return True, ""

    already_kept = already_kept or []

    # Special combinations are always legal, whatever the individual dice are.
    combined = list(already_kept) + list(dice_values)
    if is_lange_strasse(combined) or talheim_score(combined):
        return True, ""

    kept_counts = Counter(already_kept)
    for value, count in Counter(dice_values).items():
        if value in INDIVIDUAL_SCORE:
            continue
        if count >= 3 or kept_counts.get(value, 0) >= 3:
            continue
        return False, (
            f"Cannot keep {count} die/dice with value {value}. "
            "Need at least 3 of the same value (except 1s and 5s)."
        )
    return True, ""


def can_keep_any(available, kept):
    """True if at least one legal keep exists from ``available`` given ``kept`` values."""
    if not available:
        return True  # Nothing available to keep is not a dead end (all 6 already kept).

    # 1s and 5s can always be kept.
    if any(value in INDIVIDUAL_SCORE for value in available):
        return True

    # Some keep could complete a Lange Strasse (possible iff every face is present).
    if is_lange_strasse(list(kept) + list(available)):
        return True

    # A value already has (or would reach) a keepable triplet.
    kept_counts = Counter(kept)
    if any(
        count >= 3 or kept_counts.get(value, 0) >= 3
        for value, count in Counter(available).items()
    ):
        return True

    # Some keep could complete a Talheim (exactly six dice forming three pairs).
    need = NUM_DICE - len(kept)
    kept_list = list(kept)
    return 0 < need <= len(available) and any(
        talheim_score(kept_list + list(combo))
        for combo in itertools.combinations(available, need)
    )


# --------------------------------------------------------------------------- #
# Actions: what a player can do at a decision point.
# --------------------------------------------------------------------------- #
@dataclass
class Action:
    """One decision: which dice values to keep, and whether to stop afterwards."""

    dice_to_keep: list[int]
    stop_after: bool

    def __str__(self):
        dice_str = " ".join(map(str, self.dice_to_keep))
        return f"keep {dice_str}" + (" stop" if self.stop_after else "")


def valid_keeps(available, kept_values):
    """All distinct, legally-keepable subsets of ``available`` (as value lists)."""
    keeps = []
    seen = set()
    for r in range(1, len(available) + 1):
        for combo in itertools.combinations(available, r):
            key = tuple(sorted(combo))  # equivalent keeps differ only in order
            if key in seen:
                continue
            seen.add(key)
            if is_valid_keep(list(combo), kept_values)[0]:
                keeps.append(list(combo))
    return keeps


def may_stop(kept_groups, new_values):
    """True if a player may stop after also keeping ``new_values``: fewer than
    all six dice kept, and the resulting set worth at least STOP_MIN."""
    merged = merge_kept(kept_groups, new_values)
    return len(flatten(merged)) < NUM_DICE and score_groups(merged) >= STOP_MIN


def legal_actions(available, kept_groups):
    """Every legal Action at a decision point: each valid keep, plus a stopping
    variant where stopping is allowed."""
    kept_values = flatten(kept_groups)
    actions = []
    for keep in valid_keeps(available, kept_values):
        actions.append(Action(dice_to_keep=keep, stop_after=False))
        if may_stop(kept_groups, keep):
            actions.append(Action(dice_to_keep=keep, stop_after=True))
    return actions
