"""main.py"""
from game import Player, Game
from ai_player import AIPlayer, create_mixed_game
from nn_state import NNStateExtractor

def main():
    """Main game loop for 3-player Lange Strasse"""
    print("🎲 Welcome to 3-Player Lange Strasse! 🎲")
    print("Choose game mode:")
    print("1. Human vs Human vs Human")
    print("2. Human vs AI vs AI")
    print("3. Human vs Human vs AI")
    print("4. AI vs AI vs AI (watch mode)")

    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if 1 <= choice <= 4:
                break
            else:
                print("Please enter 1, 2, 3, or 4")
        except ValueError:
            print("Please enter a valid number")

    # Create game based on choice
    if choice == 1:
        game = Game()  # All human players
    elif choice == 2:
        game = Game()
        game.players[1] = AIPlayer("AI Player 2")
        game.players[2] = AIPlayer("AI Player 3")
    elif choice == 3:
        game = Game()
        game.players[2] = AIPlayer("AI Player 3")
    elif choice == 4:
        game = create_mixed_game(3)  # All AI players

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
    print("\nSpecial combinations:")
    print("- Lange Strasse (1-2-3-4-5-6): 1250 points + 50¢ from each opponent")
    print("- Talheim (3 pairs): 500 points (1000 if consecutive)")
    print("- Totale (no keepable dice at start): Pay 50¢ to each opponent")

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

        # Check if current player is AI
        if hasattr(current_player, 'is_ai') and current_player.is_ai:
            # AI player's turn
            action = current_player.choose_action(game)
            if action is None:
                print(f"{current_player.name} has no valid moves!")
                continue

            print(f"\n{current_player.name} chooses: {action}")
            explanation = current_player.get_explanation(game, action)
            print(explanation)

            # Execute AI action using unified method
            success, result = game.process_dice_action(action.dice_to_keep, action.stop_after)

            if success:
                if result == "continue_turn":
                    print()
                    game.dice_set.display()
                elif result == "turn_ended" and not game.game_over:
                    game.dice_set.display()
            else:
                print(f"AI Error: {result}")

            continue

        # Human player's turn
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

                # Try to keep the dice using unified method
                success, result = game.process_dice_action(dice_values, stop_after)

                if success:
                    # Print state AFTER the dice are kept and processed
                    print("\n" + "="*50)
                    print("GAME STATE AFTER MOVE:")
                    print("="*50)

                    state = NNStateExtractor.extract_state(game)
                    print(f"Available dice:      {state.available_dice}")
                    print(f"Kept dice counts:    {state.kept_dice_counts}")
                    print(f"Kept 1s grouped:     {state.kept_1s_grouped}")
                    print(f"Kept 5s grouped:     {state.kept_5s_grouped}")
                    print(f"Turn score:          {state.turn_score}")
                    print(f"Current set score:   {state.current_set_score}")
                    print(f"Player scores:       {state.player_scores}")
                    print(f"Player strich:       {state.player_strich}")
                    print(f"Turn number:         {state.turn_number}")
                    print(f"Your player number:  {state.your_player_number}")
                    print("="*50)

                    if result == "continue_turn":
                        print()
                        game.dice_set.display()
                    elif result == "turn_ended" and not game.game_over:
                        game.dice_set.display()
                else:
                    print(f"Error: {result}")

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

        elif command == 'hint' and hasattr(current_player, 'is_ai') and not current_player.is_ai:
            # Give AI hint to human player
            from ai_evaluator import SimpleAI
            temp_ai = SimpleAI()
            suggested_action = temp_ai.choose_action(game)
            if suggested_action:
                explanation = temp_ai.get_action_explanation(game, suggested_action)
                print(f"AI suggests: {suggested_action}")
                print(explanation)
            else:
                print("No valid moves available")

        else:
            print("Unknown command. Available: 'keep <values>', 'keep <values> stop', 'hint', 'quit'")

if __name__ == "__main__":
    main()
