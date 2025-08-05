import random

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
        for i in indices:
            if 0 <= i < len(self.dice):
                self.kept_dice[i] = True

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

                # Validate indices
                valid_indices = []
                for idx in indices:
                    if 0 <= idx < len(dice_set.dice):
                        if not dice_set.kept_dice[idx]:
                            valid_indices.append(idx)
                        else:
                            print(f"Die {idx+1} is already kept!")
                    else:
                        print(f"Invalid die number: {idx+1}")

                if valid_indices:
                    dice_set.keep_dice(valid_indices)
                    print(f"Kept dice: {[i+1 for i in valid_indices]}")
                    dice_set.display()

            except ValueError:
                print("Invalid input. Use numbers only (e.g., 'keep 1 3 5')")

        else:
            print("Unknown command. Available: 'roll', 'keep <numbers>', 'reset', 'quit'")

if __name__ == "__main__":
    main()
