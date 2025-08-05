import random
from scoring import ScoreValidator, ScoreCalculator

class DiceSet:
    """Handle dice rolling and management"""
    def __init__(self, num_dice=6):
        self.dice = [0] * num_dice
        self.kept_dice = [False] * num_dice
        self.kept_groups = []  # Track groups of dice kept together
        self.accumulated_score = 0  # Track score from previous rounds
        self.game_over = False
        self.final_score = 0
        # Auto-roll at start
        self.roll()

    def roll(self):
        """Roll all non-kept dice"""
        for i in range(len(self.dice)):
            if not self.kept_dice[i]:
                self.dice[i] = random.randint(1, 6)
        return self.dice

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
            self.final_score = self.get_current_score()
            return True, f"Game ended! Final score: {self.final_score} points"

        # Check if all dice are kept
        available = self.get_available_dice()
        if not available:
            # All 6 dice kept - add current score to accumulated and reset
            current_round_score = ScoreCalculator.calculate_score_from_groups(self.kept_groups)
            self.accumulated_score += current_round_score
            self.reset_dice_only()
            print(f"All 6 dice kept! Continuing with {self.accumulated_score} points accumulated...")
            return True, f"All dice used! Rolling again with {self.accumulated_score} points accumulated."

        # Auto-roll remaining dice
        self.roll()

        # Check if player can keep any of the new dice
        if not self.can_keep_any_dice():
            self.game_over = True
            self.final_score = 0
            return True, "No valid moves available! You lose - final score: 0 points"

        return True, "Dice kept successfully"

    def get_current_score(self):
        """Get the current total score (accumulated + current round)"""
        current_round_score = ScoreCalculator.calculate_score_from_groups(self.kept_groups)
        return self.accumulated_score + current_round_score

    def reset(self):
        """Reset everything for a new game"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        self.accumulated_score = 0
        self.game_over = False
        self.final_score = 0
        # Auto-roll after reset
        self.roll()

    def reset_dice_only(self):
        """Reset dice for a new round but keep accumulated score"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []  # Clear current round groups
        # Auto-roll after reset
        self.roll()

    def get_available_dice(self):
        """Get indices of dice that can still be rolled"""
        return [i for i, kept in enumerate(self.kept_dice) if not kept]

    def display(self):
        """Display current dice state"""
        if self.game_over:
            print(f"\nGAME OVER! Final score: {self.final_score} points")
            return

        print("\nCurrent dice:")
        for i, (die, kept) in enumerate(zip(self.dice, self.kept_dice)):
            status = "[KEPT]" if kept else ""
            print(f"Die {i+1}: {die} {status}")

        if self.kept_groups:
            print(f"Kept groups: {[sorted(group) for group in self.kept_groups]}")
            print(f"Current score: {self.get_current_score()} points")

def main():
    """Interactive command line dice rolling"""
    dice_set = DiceSet()

    print("Welcome to Lange Strasse Dice Roller!")
    print("Commands: 'keep <numbers>', 'keep <numbers> stop', 'reset', 'quit'")
    print("Example: 'keep 1 3 5' to keep dice 1, 3, and 5")
    print("Example: 'keep 1 3 stop' to keep dice 1, 3 and end the game")
    print("\nScoring rules:")
    print("- Individual 1s: 100 points each")
    print("- Individual 5s: 50 points each")
    print("- Three of a kind: number × 100 (1s = 1000, 5s = 500)")
    print("- Each additional die doubles the score")
    print("\nGame rules:")
    print("- Add 'stop' to end the game and keep your score")
    print("- Cannot stop when keeping all 6 dice")
    print("- If no dice can be kept, you lose (score = 0)")
    print("- If all 6 dice are kept, continue with accumulated score")

    # Show initial roll
    dice_set.display()
    if not dice_set.game_over:
        print(f"Available dice to keep: {[i+1 for i in dice_set.get_available_dice()]}")

    while True:
        if dice_set.game_over:
            print("\nGame is over! Use 'reset' to start a new game or 'quit' to exit.")

        command = input("\nEnter command: ").strip().lower()

        if command == 'quit':
            print("Goodbye!")
            break

        elif command == 'reset':
            dice_set.reset()
            print("New game started!")
            dice_set.display()
            if not dice_set.game_over:
                print(f"Available dice to keep: {[i+1 for i in dice_set.get_available_dice()]}")

        elif command.startswith('keep'):
            if dice_set.game_over:
                print("Game is over! Use 'reset' to start a new game.")
                continue

            try:
                parts = command.split()
                if len(parts) == 1:
                    print("Please specify which dice to keep (e.g., 'keep 1 3 5')")
                    continue

                # Check if 'stop' is in the command
                stop_after = 'stop' in parts
                if stop_after:
                    parts.remove('stop')

                if len(parts) == 1:  # Only 'keep' and 'stop'
                    print("Please specify which dice to keep before 'stop'")
                    continue

                indices = [int(x) - 1 for x in parts[1:]]  # Convert to 0-based

                # Validate indices are in range and not already kept
                valid_indices = []
                for idx in indices:
                    if 0 <= idx < len(dice_set.dice):
                        if not dice_set.kept_dice[idx]:
                            valid_indices.append(idx)
                        else:
                            print(f"Die {idx+1} is already kept!")
                    else:
                        print(f"Invalid die number: {idx+1}")

                if not valid_indices:
                    continue

                # Try to keep the dice
                success, message = dice_set.keep_dice(valid_indices, stop_after)

                if success:
                    action = "Stopped" if stop_after else "Kept"
                    print(f"{action} dice: {[i+1 for i in valid_indices]}")
                    dice_set.display()

                    if not dice_set.game_over:
                        available = dice_set.get_available_dice()
                        if available:
                            print(f"Available dice to keep: {[i+1 for i in available]}")
                        else:
                            print("All dice are kept!")
                else:
                    print(f"Error: {message}")

            except ValueError:
                print("Invalid input. Use numbers only (e.g., 'keep 1 3 5' or 'keep 1 3 stop')")

        else:
            print("Unknown command. Available: 'keep <numbers>', 'keep <numbers> stop', 'reset', 'quit'")

if __name__ == "__main__":
    main()
