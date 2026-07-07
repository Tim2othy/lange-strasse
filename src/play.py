"""Single entry point. What it does is decided by config.py:

- AIS_PLAY = True  -> simulate N_GAMES all-AI games (with ALGOS) and report wins.
                      Set VERBOSE=True, N_GAMES=1 to watch a single game play out.
- AIS_PLAY = False -> play one interactive game (you, optionally vs AI opponents).
"""

import sys
from collections import Counter

import log as log_module
from ai_player import AIPlayer
from config import AIS_PLAY, SEED
from game.game import Game, Player
from game.rules import Action
from game_state import StateExtractor
from log import log


# --------------------------------------------------------------------------- #
# The one game loop (AI seats decide for themselves, human seats are prompted)
# --------------------------------------------------------------------------- #
def run_game(game: Game) -> Game | None:
    """Play ``game`` to completion. Returns the finished game (None if a human quit)."""
    while not game.game_over:
        game.advance_to_decision()
        if game.game_over:
            break
        game.dice_set.display()

        player = game.current_player
        if isinstance(player, AIPlayer):
            state = StateExtractor.extract_state(game)
            action = player.choose_action(state, game.legal_actions())
            log(f"\n{player.name} chooses: {action}")
            game.apply_action(action)
        elif not take_human_turn(game):
            return None  # human quit
    return game


# --------------------------------------------------------------------------- #
# AI simulation (no console input)
# --------------------------------------------------------------------------- #
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
        players = [
            AIPlayer(f"AI Player {i + 1}", algorithm)
            for i, algorithm in enumerate(seat_algorithms)
        ]
        game = run_game(Game(players))
        assert game is not None and game.winner is not None
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
def choose_players(algorithms) -> list[Player] | list[AIPlayer]:
    """Ask the human which human/AI mix to play and build the seats."""
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
            print("Please enter 1, 2, 3, or 4")
        except ValueError:
            print("Please enter a valid number")

    players = [Player(f"Player {i + 1}") for i in range(3)]
    if choice == 2:
        players[1] = AIPlayer("AI Player 2", algorithms[0])
        players[2] = AIPlayer("AI Player 3", algorithms[1])
    elif choice == 3:
        players[2] = AIPlayer("AI Player 3", algorithms[0])
    elif choice == 4:
        players = [AIPlayer(f"AI Player {i + 1}", algorithms[i]) for i in range(3)]
    return players


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


def take_human_turn(game: Game) -> bool:
    """Prompt the human for one command. Returns False if they quit."""
    player = game.current_player
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

        success, result = game.apply_action(Action(dice_values, stop_after))
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
        game.dice_set.force_roll(values)
        print("💬 Debug: Forced dice roll")
        return True

    print("Unknown command. Available: 'keep <values>', 'keep <values> stop', 'quit'")
    return True


def play_interactive(algorithms):
    """Play one game with a human at the keyboard (optionally vs AI opponents)."""
    log_module.VERBOSE = True  # a human is watching, whatever config.VERBOSE says
    players = choose_players(algorithms)
    print_rules()
    run_game(Game(players))


def play(n_games, algorithms):
    print("🎲 Welcome to 3-Player Lange Strasse! 🎲")
    if AIS_PLAY:
        simulate(n_games, algorithms, SEED)
    else:
        play_interactive(algorithms)
