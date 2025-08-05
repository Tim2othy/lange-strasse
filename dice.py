import random
from scoring import ScoreValidator

class DiceSet:
    """Handle dice rolling and management"""
    def __init__(self, num_dice=6):
        self.dice = [0] * num_dice
        self.kept_dice = [False] * num_dice

    def roll(self):
        for i in range(len(self.dice)):
            if not self.kept_dice[i]:
                self.dice[i] = random.randint(1, 6)
        return self.dice

    def keep_dice(self, indices):
        """Keep dice at specified indices if the combination is valid"""
        # Get the values of dice we want to keep
        values_to_keep = [self.dice[i] for i in indices if 0 <= i < len(self.dice) and not self.kept_dice[i]]

        # Validate the combination
        is_valid, error_msg = ScoreValidator.is_valid_keep(values_to_keep)

        if not is_valid:
            return False, error_msg

        # If valid, keep the dice
        for i in indices:
            if 0 <= i < len(self.dice):
                self.kept_dice[i] = True

        return True, "Dice kept successfully"

    def reset(self):
        """Reset all dice to be rollable again"""
        self.kept_dice = [False] * len(self.dice)

    def get_available_dice(self):
        """Get indices of dice that can still be rolled"""
        return [i for i, kept in enumerate(self.kept_dice) if not kept]

    def display(self):
        """Display current dice state"""
        print("\nCurrent dice:")
        for i, (die, kept) in enumerate(zip(self.dice, self.kept_dice)):
            status = "[KEPT]" if kept else ""
            print(f"Die {i+1}: {die} {status}")

def main():
    """Interactive command line dice rolling"""
    dice_set = DiceSet()

    print("Welcome to Lange Strasse Dice Roller!")
    print("Commands: 'roll', 'keep <numbers>', 'reset', 'quit'")
    print("Example: 'keep 1 3 5' to keep dice 1, 3, and 5")
    print("\nKeeping rules:")
    print("- You can keep any number of 1s or 5s")
    print("- You can keep groups of 3 or more identical dice")
    print("- You cannot keep other combinations")

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == 'quit':
            print("Goodbye!")
            break

        elif command == 'roll':
            available = dice_set.get_available_dice()
            if not available:
                print("All dice are kept! Use 'reset' to roll all dice again.")
                continue

            dice_set.roll()
            dice_set.display()
            print(f"Available dice to keep: {[i+1 for i in available]}")

        elif command == 'reset':
            dice_set.reset()
            print("All dice reset - ready to roll!")

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
                else:
                    print(f"Invalid combination: {message}")
                    print("Try again with a valid combination.")

            except ValueError:
                print("Invalid input. Use numbers only (e.g., 'keep 1 3 5')")

        else:
            print("Unknown command. Available: 'roll', 'keep <numbers>', 'reset', 'quit'")

if __name__ == "__main__":
    main()
