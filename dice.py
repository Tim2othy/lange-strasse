"""dice.py hore the DiceSet class lives"""
import random
from scoring import ScoreValidator, ScoreCalculator
from collections import Counter

class DiceSet:
    """Handle dice rolling and management"""
    def __init__(self, num_dice=6):
        self.dice = [0] * num_dice
        self.kept_dice = [False] * num_dice
        self.kept_groups = []
        self.turn_accumulated_score = 0
        self.game_over = False
        self.lange_strasse_achieved = False
        self.super_strasse_achieved = False  # Track if this is a super strasse
        self.roll_count = 0  # Track number of rolls this turn
        self.debug_next_roll = None  # For testing - force next dice values
        # Auto-roll at start
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def roll(self):
        """Roll all non-kept dice"""
        self.roll_count += 1  # Increment roll count

        if self.debug_next_roll:
            # Use forced values for debugging
            available_indices = self.get_available_dice()
            for i, idx in enumerate(available_indices):
                if i < len(self.debug_next_roll):
                    self.dice[idx] = self.debug_next_roll[i]
                else:
                    self.dice[idx] = random.randint(1, 6)
            self.debug_next_roll = None  # Clear after use
        else:
            # Normal random roll
            for i in range(len(self.dice)):
                if not self.kept_dice[i]:
                    self.dice[i] = random.randint(1, 6)
        return self.dice

    def force_next_roll(self, values):
        """Debug: Force the next roll to have specific values"""
        self.debug_next_roll = values
        self.roll_count -= 1

    def check_game_over(self):
        """Central method to check if game should end due to no valid moves"""
        if not self.can_keep_any_dice() and not self.game_over:
            self.game_over = True
            # Check if this is a Totale (no dice kept at all)
            if len(self.kept_groups) == 0:
                print("💀 TOTALE! No dice can be kept! 💀")
                return "TOTALE"
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

        # Check if keeping all available dice would complete a Lange Strasse
        temp_kept_values = all_kept_values + available_dice
        if set(temp_kept_values).issuperset({1, 2, 3, 4, 5, 6}):
            return True  # This would complete a Lange Strasse

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

    def check_lange_strasse(self):
        """Check if all dice 1-6 have been kept (Lange Strasse)"""
        all_kept_values = []
        for group in self.kept_groups:
            all_kept_values.extend(group)

        # Check if we have at least one of each value 1-6
        unique_values = set(all_kept_values)
        return unique_values == {1, 2, 3, 4, 5, 6}

    def check_talheim(self):
        """Check if we have 3 pairs (Talheim) - must be exactly 6 dice with 3 pairs"""
        all_kept_values = []
        for group in self.kept_groups:
            all_kept_values.extend(group)

        if len(all_kept_values) != 6:
            return False, 0

        # Count occurrences of each value
        counts = Counter(all_kept_values)

        # Check if we have exactly 3 pairs (each value appears exactly twice)
        if len(counts) != 3 or not all(count == 2 for count in counts.values()):
            return False, 0

        # Check if pairs are consecutive
        values = sorted(counts.keys())
        is_consecutive = (values[1] == values[0] + 1 and values[2] == values[1] + 1)

        if is_consecutive:
            return True, 1000  # Consecutive pairs
        else:
            return True, 500   # Non-consecutive pairs

    def check_totale(self):
        """Check if this is a Totale (no dice kept at start of turn)"""
        return len(self.kept_groups) == 0 and not self.can_keep_any_dice()


    def keep_dice_by_value(self, dice_values, stop_after=False):
        """Keep dice by their face values (e.g., keep two 5s, one 1)"""
        # Convert dice_values to counts
        values_to_keep = Counter(dice_values)

        # Check if we can keep all 6 dice with this selection
        total_kept_after = sum(self.kept_dice) + len(dice_values)
        if stop_after and total_kept_after == 6:
            return False, "Cannot stop when keeping all 6 dice. Must keep 5 or fewer."

        # Get available dice and their counts
        available_indices = self.get_available_dice()
        available_dice = [self.dice[i] for i in available_indices]
        available_counts = Counter(available_dice)

        # Check if we have enough of each requested value
        for value, count in values_to_keep.items():
            if available_counts[value] < count:
                return False, f"Only {available_counts[value]} dice with value {value} available, but {count} requested."

        # Get all currently kept values for validation
        all_kept_values = []
        for group in self.kept_groups:
            all_kept_values.extend(group)

        dice_values_list = []
        for value, count in values_to_keep.items():
            dice_values_list.extend([value] * count)

        # Check for special combinations first
        temp_kept_values = all_kept_values + dice_values_list

        # Check if this would be a special combination (always valid)
        if len(temp_kept_values) == 6:
            temp_counts = Counter(temp_kept_values)
            is_potential_talheim = (len(temp_counts) == 3 and
                                  all(count == 2 for count in temp_counts.values()))
            is_potential_lange_strasse = set(temp_kept_values).issuperset({1, 2, 3, 4, 5, 6})

            if is_potential_talheim or is_potential_lange_strasse:
                # Special combinations are always valid
                pass
            else:
                # Regular validation
                is_valid, error_msg = ScoreValidator.is_valid_keep(dice_values_list, all_kept_values)
                if not is_valid:
                    return False, error_msg
        else:
            # Check if this would complete a Lange Strasse
            if set(temp_kept_values).issuperset({1, 2, 3, 4, 5, 6}):
                # This is part of a Lange Strasse - always valid
                pass
            else:
                # Regular validation for non-special combinations
                is_valid, error_msg = ScoreValidator.is_valid_keep(dice_values_list, all_kept_values)
                if not is_valid:
                    return False, error_msg

        # Check minimum score requirement for stopping BEFORE keeping dice
        if stop_after:
            # Calculate what the score would be if we kept these dice
            temp_groups = self.kept_groups.copy()
            temp_groups.append(dice_values_list)
            projected_set_score = ScoreCalculator.calculate_score_from_groups(temp_groups)

            if projected_set_score < 300:
                return False, f"Cannot stop with less than 300 points from current dice set. Would score {projected_set_score} points."

        # Keep the dice (find indices and mark them as kept)
        for i in available_indices:
            die_value = self.dice[i]
            if values_to_keep[die_value] > 0:
                values_to_keep[die_value] -= 1
                self.kept_dice[i] = True

        self._merge_kept_dice(dice_values_list)

        # Check for Talheim first (takes priority)
        is_talheim, talheim_score = self.check_talheim()
        if is_talheim:
            self.game_over = True
            print(f"🎯 TALHEIM! Three pairs worth {talheim_score} points! 🎯")
            return True, f"TALHEIM_{talheim_score}"

        # Check for Lange Strasse
        if self.check_lange_strasse():
            self.lange_strasse_achieved = True

            # Check if this is a Super Strasse (achieved on 3rd roll)
            if self.roll_count >= 3:
                self.super_strasse_achieved = True
                print("🌟 SUPER STRASSE! Lange Strasse on the 3rd roll! 🌟")
            else:
                print("🎉 LANGE STRASSE! 1-2-3-4-5-6 completed! 🎉")
        # Check if stopping
        if stop_after:
            self.game_over = True
            turn_score = self.get_current_total_score()
            return True, f"Turn ended! Score this turn: {turn_score} points"

        # Check if all dice are kept
        available = self.get_available_dice()
        if not available:
            # All 6 dice kept - add current round score to accumulated and reset dice
            current_round_score = self.get_current_score()
            if self.lange_strasse_achieved:
                current_round_score += 1100 #HACK this isn't a good solution, I'm so sorry
            self.turn_accumulated_score += current_round_score
            print(f"All 6 dice kept! Adding {current_round_score} points. Turn total so far: {self.turn_accumulated_score}")
            self.reset_dice_only()
            self.roll_count = 0
            return True, f"All dice used! Rolling again with {self.turn_accumulated_score} points accumulated this turn."

        # Auto-roll remaining dice
        self.roll()

        # Check if player can keep any of the new dice
        if self.check_game_over():
            return True, "No valid moves available! Turn ends with 0 points"

        return True, "Dice kept successfully"

    def get_current_score(self):
        """Get the current score for this round only"""

        # Check for Talheim first
        is_talheim, talheim_score = self.check_talheim()
        if is_talheim:
            return talheim_score

        return ScoreCalculator.calculate_score_from_groups(self.kept_groups)

    def _merge_kept_dice(self, new_dice_values):
        """Add new dice values, only merging when it creates a scoring advantage"""
        new_counts = Counter(new_dice_values)

        for value, new_count in new_counts.items():
            # Check if we should merge with existing groups of the same value
            merged = False

            # Only merge if it would create/extend a triplet or better
            for i, group in enumerate(self.kept_groups):
                if (len(set(group)) == 1 and group[0] == value and
                    len(group) >= 3):  # Only merge with existing triplets+
                    self.kept_groups[i].extend([value] * new_count)
                    merged = True
                    break

            if not merged:
                # Check if the new dice form a triplet by themselves
                if new_count >= 3:
                    # Keep as a group
                    self.kept_groups.append([value] * new_count)
                else:
                    # Keep individually (important for 1s and 5s kept separately)
                    for _ in range(new_count):
                        self.kept_groups.append([value])

    def get_current_total_score(self):
        """Get the total score for the entire turn (accumulated + current round)"""
        return self.turn_accumulated_score + self.get_current_score()

    def reset_for_new_turn(self):
        """Reset for a new player's turn"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        self.turn_accumulated_score = 0
        self.game_over = False
        self.lange_strasse_achieved = False  # Reset for new turn
        self.super_strasse_achieved = False  # Reset for new turn
        self.roll_count = 0  # Reset roll count for new turn
        # Auto-roll for new turn
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def reset_dice_only(self):
        """Reset dice for continuing the same turn"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        # Keep lange_strasse_achieved, super_strasse_achieved, roll_count, and turn_accumulated_score
        # Auto-roll after reset
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def get_available_dice(self):
        """Get indices of dice that can still be rolled"""
        return [i for i, kept in enumerate(self.kept_dice) if not kept]

    def get_available_dice_values(self):
        """Get the values of dice that can still be rolled"""
        available_indices = self.get_available_dice()
        return [self.dice[i] for i in available_indices]

    def display(self):
        """Display current dice state in a cleaner format"""
        if self.game_over:
            print(f"\nTurn over! Score: 0 points (no valid moves)")
            return

        # Show kept dice groups
        if self.kept_groups:
            kept_display = []
            for group in self.kept_groups:
                if len(group) == 1:
                    # Single die - just show the value
                    kept_display.append(str(group[0]))
                elif len(group) >= 3 and len(set(group)) == 1:
                    # Three or more of the same value - show as a group
                    kept_display.append(f"[{', '.join(map(str, sorted(group)))}]")
                else:
                    # Mixed group - show individual values
                    kept_display.extend(map(str, sorted(group)))

            print(f"Kept dice: {', '.join(kept_display)}")

        # Show available dice
        available_values = self.get_available_dice_values()
        if available_values:
            print(f"Available dice: {', '.join(map(str, sorted(available_values)))}")
        else:
            print("All dice are kept!")

        # Show scores
        if self.turn_accumulated_score > 0:
            print(f"Current set score: {self.get_current_score()} points")

        total_turn_score = self.get_current_total_score()
        print(f"🧩Total turn  score: {total_turn_score} points")
