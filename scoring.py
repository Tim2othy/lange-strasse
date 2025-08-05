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
    def calculate_score(dice):
        # TODO: Implement Lange Strasse scoring rules
        # Return (score, scoring_dice_indices)
        pass

    @staticmethod
    def has_scoring_dice(dice):
        score, _ = ScoreCalculator.calculate_score(dice)
        return score > 0
