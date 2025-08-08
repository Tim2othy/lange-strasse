"""game.py where the game is on"""
from dice import DiceSet


class Player:
    """Represents a player in the game"""
    def __init__(self, name):
        self.name = name
        self.total_score = 0
        self.money = 0  # Money in cents

    def add_money(self, cents):
        """Add money (can be negative for losses)"""
        self.money += cents
        if cents > 0:
            print(f"💰 {self.name} gains {cents}¢!")
        else:
            print(f"💸 {self.name} loses {abs(cents)}¢!")


class Game:
    """Main game controller for 2-player Lange Strasse"""
    def __init__(self):
        self.players = [Player("Player 1"), Player("Player 2")]
        self.current_player_idx = 0
        self.dice_set = DiceSet()
        self.game_over = False
        self.winner = None
        self.final_turn = False  # True when someone reaches 10000+ and other player gets final turn

    def get_current_player(self):
        return self.players[self.current_player_idx]

    def switch_player(self):
        self.current_player_idx = (self.current_player_idx + 1) % 2

    def handle_lange_strasse(self):
        """Handle Lange Strasse money distribution"""
        current_player = self.get_current_player()
        other_player = self.players[1 - self.current_player_idx]

        # Current player gains 50¢, other player loses 50¢
        current_player.add_money(50)
        other_player.add_money(-50)

    def handle_totale(self):
        """Handle Totale money penalty"""
        current_player = self.get_current_player()
        other_player = self.players[1 - self.current_player_idx]

        # Current player loses 50¢, other player gains 50¢
        current_player.add_money(-50)
        other_player.add_money(50)

    def end_turn(self, turn_score):
        """End current player's turn and add score"""
        current_player = self.get_current_player()
        current_player.total_score += turn_score

        print(f"\n{current_player.name} scored {turn_score} points this turn!")
        print(f"{current_player.name}'s total score: {current_player.total_score}")

        # Check win conditions
        if current_player.total_score >= 2000: # temporarily lowered for testing
            if self.current_player_idx == 0 and not self.final_turn:
                # Player 1 reached 10000+, Player 2 gets one more turn
                self.final_turn = True
                print(f"\n{current_player.name} reached 10,000 points! {self.players[1].name} gets one final turn to overtake!")
            else:
                # Either Player 2 reached 10000+ or it's the final turn
                self.check_winner()
                return

        # Switch to next player
        self.switch_player()

        # If it was the final turn, check winner
        if self.final_turn and self.current_player_idx == 0:
            self.check_winner()
            return

        # Start next player's turn
        self.start_new_turn()

    def check_winner(self):
        """Determine the winner and handle money"""
        player1, player2 = self.players

        if player1.total_score > player2.total_score:
            self.winner = player1
            loser = player2
        elif player2.total_score > player1.total_score:
            self.winner = player2
            loser = player1
        else:
            # Tie - player who reached 10000 first wins (Player 1)
            self.winner = player1
            loser = player2

        # Money distribution for winning/losing
        self.winner.add_money(50)
        loser.add_money(-50)

        self.game_over = True
        print(f"\n🎉 GAME OVER! {self.winner.name} wins! 🎉")
        print(f"Final scores:")
        print(f"Player 1: {player1.total_score} points (Money: {player1.money}¢)")
        print(f"Player 2: {player2.total_score} points (Money: {player2.money}¢)")

    def start_new_turn(self):
        """Start a new turn for the current player"""
        self.dice_set.reset_for_new_turn()
        current_player = self.get_current_player()
        print(f"\n--- {current_player.name}'s turn ---")
        print(f"Current scores: Player 1: {self.players[0].total_score}, Player 2: {self.players[1].total_score}")

        if self.final_turn and self.current_player_idx == 1:
            print("⚡ FINAL TURN! Last chance to overtake! ⚡")

        # Check for Totale at start of turn
        if self.dice_set.check_totale():
            print("💀 TOTALE! No dice can be kept at start of turn! 💀")
            self.handle_totale()
            self.end_turn(0)
            return
