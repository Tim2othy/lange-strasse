import random
from scoring import ScoreValidator, ScoreCalculator

class DiceSet:
    """Handle dice rolling and management"""
    def __init__(self, num_dice=6):
        self.dice = [0] * num_dice
        self.kept_dice = [False] * num_dice
        self.kept_groups = []  # Track groups of dice kept together
        # Auto-roll at start
        self.roll()

    def roll(self):
        """Roll all non-kept dice"""
        for i in range(len(self.dice)):
            if not self.kept_dice[i]:
                self.dice[i] = random.randint(1, 6)
        return self.dice

    def keep_dice(self, indices):
        """Keep dice at specified indices if the combination is valid"""
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

        # Auto-roll remaining dice
        available = self.get_available_dice()
        if available:
            self.roll()

        return True, "Dice kept successfully"

    def get_current_score(self):
        """Get the current score for all kept dice"""
        return ScoreCalculator.calculate_score_from_groups(self.kept_groups)

    def reset(self):
        """Reset all dice to be rollable again"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        # Auto-roll after reset
        self.roll()

    def get_available_dice(self):
        """Get indices of dice that can still be rolled"""
        return [i for i, kept in enumerate(self.kept_dice) if not kept]

    def display(self):
        """Display current dice state"""
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
    print("Commands: 'keep <numbers>', 'reset', 'quit'")
    print("Example: 'keep 1 3 5' to keep dice 1, 3, and 5")
    print("\nScoring rules:")
    print("- Individual 1s: 100 points each")
    print("- Individual 5s: 50 points each")
    print("- Three of a kind: number × 100 (1s = 1000, 5s = 500)")
    print("- Each additional die doubles the score")
    print("\nKeeping rules:")
    print("- You can keep any number of 1s or 5s")
    print("- You can keep groups of 3 or more identical dice")
    print("- If you have 3+ of a kind kept, you can keep more of that kind")
    print("- Dice auto-roll after keeping some")

    # Show initial roll
    dice_set.display()
    print(f"Available dice to keep: {[i+1 for i in dice_set.get_available_dice()]}")

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == 'quit':
            print("Goodbye!")
            break

        elif command == 'reset':
            dice_set.reset()
            print("All dice reset and re-rolled!")
            dice_set.display()
            print(f"Available dice to keep: {[i+1 for i in dice_set.get_available_dice()]}")

        elif command.startswith('keep'):
            try:
                parts = command.split()
                if len(parts) == 1:
                    print("Please specify which dice to keep (e.g., 'keep 1 3 5')")
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
                success, message = dice_set.keep_dice(valid_indices)

                if success:
                    print(f"Kept dice: {[i+1 for i in valid_indices]}")
                    dice_set.display()

                    available = dice_set.get_available_dice()
                    if available:
                        print(f"Available dice to keep: {[i+1 for i in available]}")
                    else:
                        print("All dice are kept! Use 'reset' to start over.")
                else:
                    print(f"Invalid combination: {message}")
                    print("Try again with a valid combination.")

            except ValueError:
                print("Invalid input. Use numbers only (e.g., 'keep 1 3 5')")

        else:
            print("Unknown command. Available: 'keep <numbers>', 'reset', 'quit'")

if __name__ == "__main__":
    main()
