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
        self.starting_player_idx = 0
        self.turn_number = 1
        self.someone_reached_10k = False  # Track if anyone reached 10k

        # Initialize has_strich for all players
        for player in self.players:
            player.has_strich = False

    def handle_lange_strasse(self, is_super=False):
        """Handle Lange Strasse money distribution"""
        current_player = self.get_current_player()
        money_amount = 100 if is_super else 50

        # Current player gains money from each other player
        for i, player in enumerate(self.players):
            if i != self.current_player_idx:
                player.add_money(-money_amount)
                current_player.add_money(money_amount)

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

        # Mark strich if player got 0 points
        if turn_score == 0:
            current_player.has_strich = True

        print(f"\n{current_player.name} scored {turn_score} points this turn!")
        print(f"{current_player.name}'s total score: {current_player.total_score}")

        # Check win conditions
        if current_player.total_score >= 10000:
            if not self.final_turn:
                # First player to reach 10k
                self.final_turn = True
                self.someone_reached_10k = True
                print(f"\n{current_player.name} reached 10,000 points!")
                print("⚡ FINAL ROUND! This is the final round we are just going to finish it. ⚡")

        # Check if final round is complete
        if self.final_turn and self.someone_reached_10k:
            # Final round ends when we've completed a full round after someone reached 10k
            next_player_idx = (self.current_player_idx + 1) % len(self.players)
            if next_player_idx == self.starting_player_idx:
                # We've completed the final round
                self.check_winner()
                return

        # Switch to next player and continue if game not over
        if not self.game_over:
            self.switch_player()

            # Increment turn number when we complete a full round (back to starting player)
            if self.current_player_idx == self.starting_player_idx:
                self.turn_number += 1

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

        # Basic money distribution for winning/losing
        winner.add_money(50)  # From 2nd place
        winner.add_money(70)  # From 3rd place
        second_place.add_money(-50)
        third_place.add_money(-70)

        # Bonus for winning by 10th round
        if self.turn_number <= 10:
            print(f"🎉 {winner.name} won by round {self.turn_number}! Early win bonus!")
            for i, player in enumerate(self.players):
                if i != self.players.index(winner):
                    player.add_money(-50)
                    winner.add_money(50)

        # No strich bonus - everyone without a strich gets 50¢ from everyone else
        no_strich_players = [p for p in self.players if not p.has_strich]
        if no_strich_players:
            print(f"🍀 No-strich bonus for: {', '.join(p.name for p in no_strich_players)}")
            for no_strich_player in no_strich_players:
                for other_player in self.players:
                    if other_player != no_strich_player:
                        other_player.add_money(-50)
                        no_strich_player.add_money(50)

        # Under 5000 penalty - pay winner extra 50¢
        for player in self.players:
            if player != winner and player.total_score < 5000:
                print(f"💸 {player.name} pays extra 50¢ for being under 5000 points!")
                player.add_money(-50)
                winner.add_money(50)

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

        # Check for Totale at start of turn
        if self.dice_set.check_totale():
            print("💀 TOTALE! No dice can be kept at start of turn! 💀")
            self.handle_totale()
            self.end_turn(0)
            return

    def get_current_player(self):
        """Get the current player object"""
        return self.players[self.current_player_idx]

    def process_dice_action(self, dice_values, stop_after):
        """Unified processing for both AI and human dice actions"""
        was_lange_strasse_achieved = self.dice_set.lange_strasse_achieved
        success, message = self.dice_set.keep_dice_by_value(dice_values, stop_after)

        if not success:
            return False, message

        # Handle Lange Strasse money (check for Super Strasse)
        if self.dice_set.lange_strasse_achieved:
            is_super = self.dice_set.super_strasse_achieved
            self.handle_lange_strasse(is_super)

        # Handle turn progression
        if message.startswith("TALHEIM_"):
            talheim_score = int(message.split("_")[1])
            self.end_turn(talheim_score)
            return True, "turn_ended"
        elif message.startswith("All dice used!"):
            return True, "continue_turn"
        elif stop_after or self.dice_set.game_over:
            turn_score = self.dice_set.get_current_total_score() if stop_after else 0
            self.end_turn(turn_score)
            return True, "turn_ended"
        else:
            return True, "continue_turn"
