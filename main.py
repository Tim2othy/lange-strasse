"""main.py"""
from game import Player, Game

def main():
    """Main game loop for 2-player Lange Strasse"""
    game = Game()

    print("🎲 Welcome to 2-Player Lange Strasse! 🎲")
    print("Goal: First to 10,000 points wins!")
    print("Commands: 'keep <values>', 'keep <values> stop', 'quit'")
    print("Example: 'keep 1 5 5' to keep one 1 and two 5s")
    print("Example: 'keep 1 1 1 stop' to keep three 1s and end turn")
    print("\nScoring rules:")
    print("- Individual 1s: 100 points each")
    print("- Individual 5s: 50 points each")
    print("- Three of a kind: number × 100 (1s = 1000, 5s = 500)")
    print("- Each additional die doubles the score")
    print("\nGame rules:")
    print("- Add 'stop' to end your turn and keep your score")
    print("- Cannot stop when keeping all 6 dice")
    print("- If no dice can be kept, your turn ends with 0 points")
    print("- If all 6 dice are kept, continue your turn with fresh dice")

    # Start first turn
    game.start_new_turn()
    game.dice_set.display()

    while not game.game_over:
        game_over_result = game.dice_set.check_game_over()
        if game_over_result:
            # Current turn ended
            if game_over_result == "TOTALE":
                game.handle_totale()
            game.end_turn(0)
            if not game.game_over:
                game.dice_set.display()
            continue

        current_player = game.get_current_player()
        command = input(f"\n{current_player.name}, enter command: ").strip().lower()

        if command == 'quit':
            print("Game ended!")
            break

        elif command.startswith('keep'):
            try:
                parts = command.split()
                if len(parts) == 1:
                    print("Please specify which dice values to keep (e.g., 'keep 1 5 5')")
                    continue

                # Check if 'stop' is in the command
                stop_after = 'stop' in parts
                if stop_after:
                    parts.remove('stop')

                if len(parts) == 1:  # Only 'keep' and 'stop'
                    print("Please specify which dice values to keep before 'stop'")
                    continue

                # Convert to dice values
                dice_values = [int(x) for x in parts[1:]]

                # Validate dice values are 1-6
                for val in dice_values:
                    if val < 1 or val > 6:
                        print(f"Invalid dice value: {val}. Must be between 1 and 6.")
                        continue

                # Try to keep the dice
                was_lange_strasse_achieved = game.dice_set.lange_strasse_achieved
                success, message = game.dice_set.keep_dice_by_value(dice_values, stop_after)

                if success:
                    # Check if Lange Strasse was just completed
                    if game.dice_set.lange_strasse_achieved and not was_lange_strasse_achieved:
                        game.handle_lange_strasse()

                    # Check for special combinations
                    if message.startswith("TALHEIM_"):
                        talheim_score = int(message.split("_")[1])
                        game.end_turn(talheim_score)
                        if not game.game_over:
                            game.dice_set.display()
                    elif message.startswith("All dice used!"):
                        # This handles both normal "all 6 dice" and Lange Strasse cases
                        print()  # Add spacing
                        game.dice_set.display()
                    elif stop_after or game.dice_set.game_over:
                        # Normal turn ending
                        if stop_after:
                            turn_score = game.dice_set.get_current_total_score()
                        else:
                            turn_score = 0

                        game.end_turn(turn_score)
                        if not game.game_over:
                            game.dice_set.display()
                    else:
                        # Turn continues
                        print()
                        game.dice_set.display()
                else:
                    print(f"Error: {message}")

            except ValueError:
                print("Invalid input. Use dice values only (e.g., 'keep 1 5 5' or 'keep 1 1 1 stop')")

        elif command.startswith('force'):
            # Debug command: force next dice roll
            try:
                parts = command.split()
                if len(parts) > 1:
                    values = [int(x) for x in parts[1:]]
                    game.dice_set.force_next_roll(values)
                    game.dice_set.roll()
                    print("Debug: Forced dice roll")
                    game.dice_set.display()
                else:
                    print("Usage: force 1 2 3 4 5 6")
            except ValueError:
                print("Invalid values for force command")

        else:
            print("Unknown command. Available: 'keep <values>', 'keep <values> stop', 'quit'")

if __name__ == "__main__":
    main()
