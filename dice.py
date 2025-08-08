import random
from scoring import ScoreValidator, ScoreCalculator

class DiceSet:
    """Handle dice rolling and management"""
    def __init__(self, num_dice=6):
        self.dice = [0] * num_dice
        self.kept_dice = [False] * num_dice
        self.kept_groups = []  # Track groups of dice kept together
        self.turn_accumulated_score = 0  # Score accumulated within this turn
        self.game_over = False
        # Auto-roll at start
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def roll(self):
        """Roll all non-kept dice"""
        for i in range(len(self.dice)):
            if not self.kept_dice[i]:
                self.dice[i] = random.randint(1, 6)
        return self.dice

    def check_game_over(self):
        """Central method to check if game should end due to no valid moves"""
        if not self.can_keep_any_dice() and not self.game_over:
            self.game_over = True
            return True
        return False

    def can_keep_any_dice(self):
        """Check if any dice can be kept according to the rules"""
        available_dice = [self.dice[i] for i in self.get_available_dice()]
        if not available_dice:
            return True  # No dice to check

        # Get all currently kept values for validation
        all_kept_values = []
        for group in self.kept_groups:
            all_kept_values.extend(group)

        # Check individual dice first (1s and 5s)
        for die_val in available_dice:
            if die_val == 1 or die_val == 5:
                return True

        # Check if we can keep any individual die based on already kept dice
        dice_counts = {}
        for val in available_dice:
            dice_counts[val] = dice_counts.get(val, 0) + 1

        kept_counts = {}
        for val in all_kept_values:
            kept_counts[val] = kept_counts.get(val, 0) + 1

        for val, count in dice_counts.items():
            # Can keep if we have 3+ of this value, or if we already have 3+ kept
            if count >= 3 or kept_counts.get(val, 0) >= 3:
                return True

        return False

    def keep_dice(self, indices, stop_after=False):
        """Keep dice at specified indices if the combination is valid"""
        # Check if trying to stop with all 6 dice
        if stop_after and len(indices) + sum(self.kept_dice) == 6:
            return False, "Cannot stop when keeping all 6 dice. Must keep 5 or fewer."

        # Get the values of dice we want to keep
        values_to_keep = [self.dice[i] for i in indices if 0 <= i < len(self.dice) and not self.kept_dice[i]]

        # Get all currently kept values for validation
        all_kept_values = []
        for group in self.kept_groups:
            all_kept_values.extend(group)

        # Validate the combination with already kept dice
        is_valid, error_msg = ScoreValidator.is_valid_keep(values_to_keep, all_kept_values)

        if not is_valid:
            return False, error_msg

        # If valid, keep the dice and record the group
        for i in indices:
            if 0 <= i < len(self.dice) and not self.kept_dice[i]:
                self.kept_dice[i] = True

        # Add this group to our kept groups
        self.kept_groups.append(values_to_keep)

        # Check if stopping
        if stop_after:
            self.game_over = True
            turn_score = self.get_current_total_score()
            return True, f"Turn ended! Score this turn: {turn_score} points"

        # Check if all dice are kept
        available = self.get_available_dice()
        if not available:
            # All 6 dice kept - add current round score to accumulated and reset dice
            current_round_score = ScoreCalculator.calculate_score_from_groups(self.kept_groups)
            self.turn_accumulated_score += current_round_score
            print(f"All 6 dice kept! Adding {current_round_score} points. Turn total so far: {self.turn_accumulated_score}")
            self.reset_dice_only()
            return True, f"All dice used! Rolling again with {self.turn_accumulated_score} points accumulated this turn."

        # Auto-roll remaining dice
        self.roll()

        # Check if player can keep any of the new dice
        if self.check_game_over():
            return True, "No valid moves available! Turn ends with 0 points"

        return True, "Dice kept successfully"

    def get_current_score(self):
        """Get the current score for this round only"""
        return ScoreCalculator.calculate_score_from_groups(self.kept_groups)

    def get_current_total_score(self):
        """Get the total score for the entire turn (accumulated + current round)"""
        return self.turn_accumulated_score + self.get_current_score()

    def reset_for_new_turn(self):
        """Reset for a new player's turn"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        self.turn_accumulated_score = 0  # Reset turn accumulation
        self.game_over = False
        # Auto-roll for new turn
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def reset_dice_only(self):
        """Reset dice for continuing the same turn"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []  # Clear current round groups but keep turn_accumulated_score
        # Auto-roll after reset
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def get_available_dice(self):
        """Get indices of dice that can still be rolled"""
        return [i for i, kept in enumerate(self.kept_dice) if not kept]

    def display(self):
        """Display current dice state"""
        if self.game_over:
            print(f"\nTurn over! Score: 0 points (no valid moves)")
            return

        print("\nCurrent dice:")
        for i, (die, kept) in enumerate(zip(self.dice, self.kept_dice)):
            status = "[KEPT]" if kept else ""
            print(f"Die {i+1}: {die} {status}")

        if self.kept_groups:
            print(f"Kept groups this round: {[sorted(group) for group in self.kept_groups]}")
            current_round_score = self.get_current_score()
            print(f"Score this round: {current_round_score} points")

        if self.turn_accumulated_score > 0:
            print(f"Accumulated this turn: {self.turn_accumulated_score} points")

        total_turn_score = self.get_current_total_score()
        print(f"Total turn score: {total_turn_score} points")
