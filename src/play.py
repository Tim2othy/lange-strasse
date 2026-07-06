"""Single entry point. What it does is decided by config.py:

- AIS_PLAY = True  -> simulate N_GAMES all-AI games (with ALGOS) and report wins.
                      Set VERBOSE=True, N_GAMES=1 to watch a single game play out.
- AIS_PLAY = False -> play one interactive game (you, optionally vs AI opponents).
"""

import sys
from collections import Counter

from game.ai_actions import ActionGenerator
from ai_player import AIPlayer
from config import AIS_PLAY, SEED, VERBOSE
from game.game import TheGame
from game_state import StateExtractor
from log import log


# --------------------------------------------------------------------------- #
# AI simulation (players decide for themselves; no console input)
# --------------------------------------------------------------------------- #
def run_ai_game(algorithms) -> TheGame:
    """Play one all-AI game to completion and return the finished game."""
    game = TheGame(choice=4, algorithms=algorithms)
    game.start_new_turn()

    while not game.game_over:
        game.advance_to_decision()
        if game.game_over:
            break
        if VERBOSE:
            game.dice_set.display()

        player = game.get_current_player()
        assert isinstance(player, AIPlayer)  # choice=4 -> every seat is an AI
        state = StateExtractor.extract_state(game)
        actions = ActionGenerator.get_valid_actions(game)
        action = player.choose_action(state, actions)

        log(f"\n{player.name} chooses: {action}")
        game.process_dice_action(action.dice_to_keep, action.stop_after)

    return game


def _progress_bar(percent: int, width: int = 10) -> str:
    filled = percent * width // 100
    return "[" + "#" * filled + " " * (width - filled) + "]"


def run_matchup(n_games, algorithms):
    """Run rotated-seat AI games and return win/money totals."""

    wins = Counter()
    money = Counter()  # cumulative end-of-game money (¢) by algorithm
    n = len(algorithms)
    next_percent = 10
    for g in range(n_games):
        seat_algorithms = [algorithms[(i + g) % n] for i in range(n)]  # rotate seats
        game = run_ai_game(seat_algorithms)
        assert game.winner is not None
        wins[seat_algorithms[game.players.index(game.winner)]] += 1
        for seat, algorithm in enumerate(seat_algorithms):
            money[algorithm] += game.players[seat].money

        progress = int((g + 1) * 100 / n_games)
        while next_percent <= 100 and progress >= next_percent:
            sys.stdout.write(f"\r{_progress_bar(next_percent)} {next_percent:3d}%")
            sys.stdout.flush()
            next_percent += 10

    sys.stdout.write("\n")
    return wins, money


def simulate(n_games, algorithms, seed=None):
    """Play ``n_games`` all-AI games and report wins and cumulative money by algorithm."""
    wins, money = run_matchup(n_games, algorithms)

    print(f"\nSimulated {n_games} game(s) with {algorithms}.")
    print("Wins by algorithm:")
    for algorithm, count in wins.most_common():
        print(f"  {algorithm:8s} {count:5d}  ({count / n_games:.0%})")
    print("Money by algorithm (settled at the end, as in real play):")
    for algorithm, total in money.most_common():
        print(f"  {algorithm:8s} {total:+7d}¢  ({total / n_games:+.1f}¢/game)")


# --------------------------------------------------------------------------- #
# Interactive play
# --------------------------------------------------------------------------- #
def choose_game_mode() -> int:
    """Ask the human which human/AI mix to play."""
    print("Choose game mode:")
    print("1. Human vs Human vs Human")
    print("2. Human vs AI vs AI")
    print("3. Human vs Human vs AI")
    print("4. AI vs AI vs AI (watch mode)")

    while True:
        try:
            choice = int(input("Enter choice (1-4): "))
            if 1 <= choice <= 4:
                return choice
            print("Please enter 1, 2, 3, or 4")
        except ValueError:
            print("Please enter a valid number")


def print_rules():
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


def take_ai_turn(game: TheGame):
    """Let the current (AI) player choose and apply an action."""
    current_player = game.get_current_player()
    assert isinstance(current_player, AIPlayer)
    state = StateExtractor.extract_state(game)
    actions = ActionGenerator.get_valid_actions(game)
    action = current_player.choose_action(state, actions)
    print(f"\n{current_player.name} chooses: {action}")
    game.process_dice_action(action.dice_to_keep, action.stop_after)


def take_human_turn(game: TheGame) -> bool:
    """Prompt the human for one command. Returns False if they quit."""
    player = game.get_current_player()
    command = input(f"\n{player.name}, enter command: ").strip().lower()

    if command == "quit":
        print("Game ended!")
        return False

    if command.startswith("keep"):
        parts = command.split()
        stop_after = "stop" in parts
        if stop_after:
            parts.remove("stop")
        if len(parts) == 1:
            print("Please specify which dice values to keep (e.g., 'keep 1 5 5')")
            return True
        try:
            dice_values = [int(x) for x in parts[1:]]
        except ValueError:
            print("Invalid input. Use dice values only (e.g., 'keep 1 5 5').")
            return True
        if any(v < 1 or v > 6 for v in dice_values):
            print("Dice values must be between 1 and 6.")
            return True

        success, result = game.process_dice_action(dice_values, stop_after)
        if not success:
            print(f"Error: {result}")
        return True

    if command.startswith("force"):
        parts = command.split()
        try:
            values = [int(x) for x in parts[1:]]
        except ValueError:
            print("Invalid values for force command")
            return True
        if not values:
            print("Usage: force 1 2 3 4 5 6")
            return True
        game.dice_set.force_next_roll(values)
        game.dice_set.roll()
        print("💬 Debug: Forced dice roll")
        return True

    print("Unknown command. Available: 'keep <values>', 'keep <values> stop', 'quit'")
    return True


def play_interactive(algorithms):
    """Play one game with a human at the keyboard (optionally vs AI opponents)."""
    choice = choose_game_mode()
    game = TheGame(choice, algorithms)
    print_rules()

    game.start_new_turn()
    while not game.game_over:
        game.advance_to_decision()
        if game.game_over:
            break
        game.dice_set.display()

        if isinstance(game.get_current_player(), AIPlayer):
            take_ai_turn(game)
        elif not take_human_turn(game):
            return  # human quit


def play(n_games, algorithms):
    print("🎲 Welcome to 3-Player Lange Strasse! 🎲")
    if AIS_PLAY:
        simulate(n_games, algorithms, SEED)
    else:
        play_interactive(algorithms)
