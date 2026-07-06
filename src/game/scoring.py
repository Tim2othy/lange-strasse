"""scoring.py -- the single source of truth for Lange Strasse scoring and rules."""

from collections import Counter

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
    if len(values) != 6:
        return 0
    counts = Counter(values)
    if len(counts) != 3 or any(count != 2 for count in counts.values()):
        return 0
    low, mid, high = sorted(counts)
    return 1000 if (mid == low + 1 and high == mid + 1) else 500


def can_keep_any(available, kept):
    """True if at least one legal keep exists from ``available`` given ``kept`` values."""
    if not available:
        return True  # Nothing available to keep is not a dead end (all 6 already kept).

    # 1s and 5s can always be kept.
    if any(value in INDIVIDUAL_SCORE for value in available):
        return True

    # Keeping everything available could complete a Lange Strasse.
    if is_lange_strasse(list(kept) + list(available)):
        return True

    # A value already has (or would reach) a keepable triplet.
    kept_counts = Counter(kept)
    return any(
        count >= 3 or kept_counts.get(value, 0) >= 3
        for value, count in Counter(available).items()
    )


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
    """Total score for kept dice groups, accounting for Talheim / Lange Strasse."""
    values = flatten(kept_groups)

    talheim = talheim_score(values)
    if talheim:
        return talheim

    if is_lange_strasse(values):
        return LANGE_STRASSE_SCORE

    return ScoreCalculator.calculate_score_from_groups(kept_groups)


class ScoreValidator:
    """Validate which dice combinations can be kept according to the rules."""

    @staticmethod
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


class ScoreCalculator:
    """Calculate points based on dice combinations."""

    @staticmethod
    def calculate_score_from_groups(kept_groups):
        """Total score for kept groups, ignoring Talheim / Lange Strasse specials.

        Used where the caller only cares about the plain triplet/individual score
        (e.g. the 300-point minimum needed to stop). Use :func:`score_groups` for
        the full score that includes special combinations.
        """
        total = 0
        for group in kept_groups:
            for value, count in Counter(group).items():
                if count >= 3:
                    base = TRIPLET_BASE.get(value, value * 100)
                    total += base * 2 ** (count - 3)
                else:
                    total += count * INDIVIDUAL_SCORE.get(value, 0)
        return total
