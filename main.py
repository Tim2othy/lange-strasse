from game import Player, Game


def main():
    """Main game loop for 2-player Lange Strasse"""
    game = Game()

    print("🎲 Welcome to 2-Player Lange Strasse! 🎲")
    print("Goal: First to 10,000 points wins!")
    print("Commands: 'keep <numbers>', 'keep <numbers> stop', 'quit'")
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
    if not game.dice_set.game_over:
        print(f"Available dice to keep: {[i+1 for i in game.dice_set.get_available_dice()]}")

    while not game.game_over:
        if game.dice_set.game_over:
            # Current turn ended with 0 points
            game.end_turn(0)
            if not game.game_over:
                game.dice_set.display()
                if not game.dice_set.game_over:
                    print(f"Available dice to keep: {[i+1 for i in game.dice_set.get_available_dice()]}")
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
                    if 0 <= idx < len(game.dice_set.dice):
                        if not game.dice_set.kept_dice[idx]:
                            valid_indices.append(idx)
                        else:
                            print(f"Die {idx+1} is already kept!")
                    else:
                        print(f"Invalid die number: {idx+1}")

                if not valid_indices:
                    continue

                # Try to keep the dice
                success, message = game.dice_set.keep_dice(valid_indices, stop_after)

                if success:
                    action = "Stopped" if stop_after else "Kept"
                    print(f"{action} dice: {[i+1 for i in valid_indices]}")

                    if stop_after or game.dice_set.game_over:
                        # Turn ended
                        if stop_after:
                            # Player chose to stop - use their accumulated score
                            turn_score = game.dice_set.get_current_total_score()
                        else:
                            # Game over due to no valid moves - score is 0
                            turn_score = 0

                        game.end_turn(turn_score)
                        if not game.game_over:
                            game.dice_set.display()
                            if not game.dice_set.game_over:
                                print(f"Available dice to keep: {[i+1 for i in game.dice_set.get_available_dice()]}")
                    else:
                        # Turn continues
                        game.dice_set.display()
                        if not game.dice_set.game_over:
                            available = game.dice_set.get_available_dice()
                            if available:
                                print(f"Available dice to keep: {[i+1 for i in available]}")
                            else:
                                print("All dice are kept!")
                else:
                    print(f"Error: {message}")

            except ValueError:
                print("Invalid input. Use numbers only (e.g., 'keep 1 3 5' or 'keep 1 3 stop')")

        else:
            print("Unknown command. Available: 'keep <numbers>', 'keep <numbers> stop', 'quit'")

if __name__ == "__main__":
    main()
