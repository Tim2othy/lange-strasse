from collections import Counter

class ScoreValidator:
    """Validate which dice combinations can be kept according to Lange Strasse rules"""

    @staticmethod
    def is_valid_keep(dice_values, already_kept_dice=None):
        """
        Check if a combination of dice values is valid to keep.
        Rules:
        - Any number of 1s or 5s can be kept
        - Any group of 3 or more identical dice can be kept
        - If you already have 3+ of a kind kept, you can keep more of that kind
        - Cannot keep dice that don't follow these rules

        Args:
            dice_values: List of dice values to check
            already_kept_dice: List of dice values already kept (for checking existing groups)

        Returns:
            (bool, str): (is_valid, error_message)
        """
        if not dice_values:
            return True, ""

        # Count what's already been kept
        kept_counts = Counter(already_kept_dice) if already_kept_dice else Counter()

        # Count occurrences of each die value being kept now
        counts = Counter(dice_values)

        for value, count in counts.items():
            if value == 1 or value == 5:
                # 1s and 5s can always be kept in any quantity
                continue
            elif count >= 3:
                # Groups of 3+ identical dice can be kept
                continue
            elif kept_counts.get(value, 0) >= 3:
                # We already have 3+ of this value kept, so we can keep more
                continue
            else:
                # This die value appears less than 3 times and we don't have 3+ already kept
                return False, f"Cannot keep {count} die/dice with value {value}. Need at least 3 of the same value (except 1s and 5s)."

        return True, ""

class ScoreCalculator:
    """Calculate points based on dice combinations"""

    @staticmethod
    def calculate_score_from_groups(kept_groups):
        """
        Calculate score for kept dice values.

        Rules:
        - Individual 1s: 100 points each
        - Individual 5s: 50 points each
        - Three of a kind: number * 100 (except 1s = 1000, 5s = 500)
        - Four of a kind: double the three of a kind score
        - Five of a kind: double the four of a kind score, etc.

        Args:
            kept_groups: List of lists, each containing dice values kept together

        Returns:
            int: Total score
        """
        if not kept_groups:
            return 0

        total_score = 0

        for group in kept_groups:
            if not group:
                continue

            counts = Counter(group)

            for value, count in counts.items():
                if count >= 3:
                    # Three or more of a kind kept together - use triplet scoring
                    if value == 1:
                        base_score = 1000
                    elif value == 5:
                        base_score = 500
                    else:
                        base_score = value * 100

                    # Double for each additional die beyond 3
                    score = base_score * (2 ** (count - 3))
                    total_score += score

                else:
                    # Individual scoring for 1s and 5s only
                    if value == 1:
                        total_score += count * 100
                    elif value == 5:
                        total_score += count * 50
                    # Other values can't be kept individually (validation prevents this)

        return total_score

    @staticmethod
    def calculate_score(kept_dice_values):
        """Legacy method - assumes all dice were kept as individual groups"""
        if not kept_dice_values:
            return 0
        groups = [[die] for die in kept_dice_values]
        return ScoreCalculator.calculate_score_from_groups(groups)

    @staticmethod
    def has_scoring_dice(dice):
        """Check if any dice can score points"""
        # For now, assume any dice can potentially score
        # This could be more sophisticated later
        return True
