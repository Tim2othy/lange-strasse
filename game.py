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
    """Main game controller for 3-player Lange Strasse"""
    def __init__(self):
        self.players = [Player("Player 1"), Player("Player 2"), Player("Player 3")]
        self.current_player_idx = 0
        self.dice_set = DiceSet()
        self.game_over = False
        self.winner = None
        self.final_turn = False
        self.starting_player_idx = 0  # Track who started the game

    def handle_lange_strasse(self):
        """Handle Lange Strasse money distribution"""
        current_player = self.get_current_player()

        # Current player gains 50¢ from each other player
        for i, player in enumerate(self.players):
            if i != self.current_player_idx:
                player.add_money(-50)
                current_player.add_money(50)

    def handle_totale(self):
        """Handle Totale money penalty"""
        current_player = self.get_current_player()

        # Current player loses 50¢ to each other player
        for i, player in enumerate(self.players):
            if i != self.current_player_idx:
                player.add_money(50)
                current_player.add_money(-50)

    def end_turn(self, turn_score):
        """End current player's turn and add score"""
        current_player = self.get_current_player()
        current_player.total_score += turn_score

        print(f"\n{current_player.name} scored {turn_score} points this turn!")
        print(f"{current_player.name}'s total score: {current_player.total_score}")

        # Check win conditions
        if current_player.total_score >= 10000:
            if not self.final_turn:
                # Someone reached 10000+, finish the round
                self.final_turn = True
                print(f"\n{current_player.name} reached 10,000 points! Finishing the round...")

            # Check if we've completed the round (back to starting player)
            next_player_idx = (self.current_player_idx + 1) % 3
            if next_player_idx == self.starting_player_idx and self.final_turn:
                self.check_winner()
                return

        # Switch to next player
        self.switch_player()

        # Start next player's turn
        self.start_new_turn()

    def switch_player(self):
        """Switch to the next player"""
        self.current_player_idx = (self.current_player_idx + 1) % 3

    def check_winner(self):
        """Determine the winner and handle money"""
        # Sort players by score (descending)
        sorted_players = sorted(self.players, key=lambda p: p.total_score, reverse=True)

        winner = sorted_players[0]
        second_place = sorted_players[1]
        third_place = sorted_players[2]

        # Money distribution for winning/losing
        # Winner gets money from 2nd and 3rd place
        winner.add_money(50)  # From 2nd place
        winner.add_money(70)  # From 3rd place
        second_place.add_money(-50)
        third_place.add_money(-70)

        self.winner = winner
        self.game_over = True
        print(f"\n🎉 GAME OVER! {winner.name} wins! 🎉")
        print(f"Final scores:")
        for i, player in enumerate(sorted_players, 1):
            print(f"{i}. {player.name}: {player.total_score} points (Money: {player.money}¢)")

    def start_new_turn(self):
        """Start a new turn for the current player"""
        self.dice_set.reset_for_new_turn()
        current_player = self.get_current_player()
        print(f"\n--- {current_player.name}'s turn ---")

        # Show all players' scores
        score_display = []
        for player in self.players:
            score_display.append(f"{player.name}: {player.total_score}")
        print(f"Current scores: {', '.join(score_display)}")

        if self.final_turn:
            print("⚡ FINAL ROUND! Everyone gets one last chance! ⚡")

        # Check for Totale at start of turn
        if self.dice_set.check_totale():
            print("💀 TOTALE! No dice can be kept at start of turn! 💀")
            self.handle_totale()
            self.end_turn(0)
            return

    def get_current_player(self):
        """Get the current player object"""
        return self.players[self.current_player_idx]
