"""Game Class"""

import random
from collections import Counter

from ai_player import AIPlayer, Player
from config import DEFAULT_ALGORITHMS
from game.rules import (
    ScoreCalculator,
    ScoreValidator,
    can_keep_any,
    flatten,
    is_lange_strasse,
    merge_kept,
    talheim_score,
)
from log import log


class DiceSet:
    """Handle dice rolling and management"""

    def __init__(self, num_dice=6):
        self.dice = [0] * num_dice
        self.kept_dice = [False] * num_dice
        self.kept_groups = []
        self.turn_accumulated_score = 0
        self.game_over = False
        self.lange_strasse_achieved = False
        self.super_strasse_achieved = False  # Track if this is a super strasse
        self.roll_count = 0  # Track number of rolls this turn
        self.debug_next_roll = None  # For testing - force next dice values
        # Auto-roll at start
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def roll(self):
        """Roll all non-kept dice"""
        self.roll_count += 1  # Increment roll count

        if self.debug_next_roll:
            # Use forced values for debugging
            available_indices = self.get_available_dice()
            for i, idx in enumerate(available_indices):
                if i < len(self.debug_next_roll):
                    self.dice[idx] = self.debug_next_roll[i]
                else:
                    self.dice[idx] = random.randint(1, 6)
            self.debug_next_roll = None  # Clear after use
        else:
            # Normal random roll
            for i in range(len(self.dice)):
                if not self.kept_dice[i]:
                    self.dice[i] = random.randint(1, 6)
        return self.dice

    def force_next_roll(self, values):
        """Debug: Force the next roll to have specific values"""
        self.debug_next_roll = values
        self.roll_count -= 1

    def check_game_over(self):
        """Central method to check if game should end due to no valid moves"""
        if not self.can_keep_any_dice() and not self.game_over:
            self.game_over = True
            # Check if this is a Totale (no dice can be kept on first roll)
            if self.roll_count == 1:
                log("💀 TOTALE! No dice can be kept! 💀")
                return "TOTALE"
            return True
        return False

    def can_keep_any_dice(self):
        """True if at least one legal keep exists from the available dice."""
        return can_keep_any(self.get_available_dice_values(), flatten(self.kept_groups))

    def check_lange_strasse(self):
        """Check if all dice 1-6 have been kept (Lange Strasse)"""
        return is_lange_strasse(flatten(self.kept_groups))

    def check_talheim(self):
        """Return ``(is_talheim, score)`` for the currently kept dice."""
        points = talheim_score(flatten(self.kept_groups))
        return points > 0, points

    def check_totale(self):
        """Check if this is a Totale (no dice can be kept on first roll)"""
        return self.roll_count == 1 and not self.can_keep_any_dice()

    def keep_dice_by_value(self, dice_values, stop_after=False):
        """Keep dice by their face values (e.g., keep two 5s, one 1)"""
        # Convert dice_values to counts
        values_to_keep = Counter(dice_values)

        # Check if we can keep all 6 dice with this selection
        total_kept_after = sum(self.kept_dice) + len(dice_values)
        if stop_after and total_kept_after == 6:
            return False, "Cannot stop when keeping all 6 dice. Must keep 5 or fewer."

        # Get available dice and their counts
        available_indices = self.get_available_dice()
        available_dice = [self.dice[i] for i in available_indices]
        available_counts = Counter(available_dice)

        # Check if we have enough of each requested value
        for value, count in values_to_keep.items():
            if available_counts[value] < count:
                return (
                    False,
                    f"Only {available_counts[value]} dice with value {value} available, but {count} requested.",
                )

        all_kept_values = flatten(self.kept_groups)

        dice_values_list = []
        for value, count in values_to_keep.items():
            dice_values_list.extend([value] * count)

        # Validate the keep (ScoreValidator allows Lange Strasse / Talheim too).
        is_valid, error_msg = ScoreValidator.is_valid_keep(
            dice_values_list, all_kept_values
        )
        if not is_valid:
            return False, error_msg

        # Check minimum score requirement for stopping BEFORE keeping dice
        if stop_after:
            # Calculate what the score would be if we kept these dice
            temp_groups = self.kept_groups.copy()
            temp_groups.append(dice_values_list)
            projected_set_score = ScoreCalculator.calculate_score_from_groups(
                temp_groups
            )

            if projected_set_score < 300:
                return (
                    False,
                    f"Cannot stop with less than 300 points from current dice set. Would score {projected_set_score} points.",
                )

        # Keep the dice (find indices and mark them as kept)
        for i in available_indices:
            die_value = self.dice[i]
            if values_to_keep[die_value] > 0:
                values_to_keep[die_value] -= 1
                self.kept_dice[i] = True

        self._merge_kept_dice(dice_values_list)

        # Check for Talheim first (takes priority)
        is_talheim, talheim_points = self.check_talheim()
        if is_talheim:
            self.game_over = True
            log(f"🎯 TALHEIM! Three pairs worth {talheim_points} points! 🎯")
            return True, f"TALHEIM_{talheim_points}"

        # Check for Lange Strasse
        if self.check_lange_strasse():
            self.lange_strasse_achieved = True

            # Check if this is a Super Strasse (achieved on 3rd roll)
            if self.roll_count >= 3:
                self.super_strasse_achieved = True
                log("🌟 SUPER STRASSE! Lange Strasse on the 3rd roll! 🌟")
            else:
                log("🎉 LANGE STRASSE! 1-2-3-4-5-6 completed! 🎉")
        # Check if stopping
        if stop_after:
            self.game_over = True
            turn_score = self.get_current_total_score()
            return True, f"Turn ended! Score this turn: {turn_score} points"

        # Check if all dice are kept
        available = self.get_available_dice()
        if not available:
            # All 6 dice kept - bank this set's score and roll a fresh set of 6.
            # (Strasse money/flags are handled by Game.process_dice_action, which
            # reads the achieved flags before the next set clears them.)
            current_round_score = self.get_current_score()
            self.turn_accumulated_score += current_round_score
            log(
                f"All 6 dice kept! Adding {current_round_score} points. Turn total so far: {self.turn_accumulated_score}"
            )
            self.reset_dice_only()
            return (
                True,
                f"All dice used! Rolling again with {self.turn_accumulated_score} points accumulated this turn.",
            )

        # Auto-roll remaining dice
        self.roll()

        # Check if player can keep any of the new dice
        if self.check_game_over():
            return True, "No valid moves available! Turn ends with 0 points"

        return True, "Dice kept successfully"

    def get_current_score(self):
        """Get the score for the current dice set (this sub-round only)"""
        # Special combinations score a fixed amount and take priority.
        is_talheim, talheim_score = self.check_talheim()
        if is_talheim:
            return talheim_score

        if self.check_lange_strasse():
            return 1250

        return ScoreCalculator.calculate_score_from_groups(self.kept_groups)

    def _merge_kept_dice(self, new_dice_values):
        """Add new dice values, only merging when it creates a scoring advantage."""
        self.kept_groups = merge_kept(self.kept_groups, new_dice_values)

    def get_current_total_score(self):
        """Get the total score for the entire turn (accumulated + current round)"""
        return self.turn_accumulated_score + self.get_current_score()

    def reset_for_new_turn(self):
        """Reset for a new player's turn"""
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        self.turn_accumulated_score = 0
        self.game_over = False
        self.lange_strasse_achieved = False  # Reset for new turn
        self.super_strasse_achieved = False  # Reset for new turn
        self.roll_count = 0  # Reset roll count for new turn
        # Auto-roll for new turn
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def reset_dice_only(self):
        """Reset the dice for a fresh set of 6 within the same turn.

        Deliberately does NOT clear lange_strasse_achieved/super_strasse_achieved:
        those are read (and then consumed) by Game.process_dice_action so the
        Strasse money is paid out. They are only fully cleared in
        reset_for_new_turn when a new player takes over.
        """
        self.kept_dice = [False] * len(self.dice)
        self.kept_groups = []
        self.roll_count = 0
        self.roll()
        # Check if game is over immediately
        self.check_game_over()

    def get_available_dice(self):
        """Get indices of dice that can still be rolled"""
        return [i for i, kept in enumerate(self.kept_dice) if not kept]

    def get_available_dice_values(self):
        """Get the values of dice that can still be rolled"""
        available_indices = self.get_available_dice()
        return [self.dice[i] for i in available_indices]

    def display(self):
        """Display current dice state in a cleaner format"""
        if self.game_over:
            log("\nTurn over! Score: 0 points (no valid moves)")
            return

        # Show kept dice groups
        if self.kept_groups:
            kept_display = []
            for group in self.kept_groups:
                if len(group) == 1:
                    # Single die - just show the value
                    kept_display.append(str(group[0]))
                elif len(group) >= 3 and len(set(group)) == 1:
                    # Three or more of the same value - show as a group
                    kept_display.append(f"[{', '.join(map(str, sorted(group)))}]")
                else:
                    # Mixed group - show individual values
                    kept_display.extend(map(str, sorted(group)))

            log(f"Kept dice: {', '.join(kept_display)}")

        # Show available dice
        available_values = self.get_available_dice_values()
        if available_values:
            log(f"Available dice: {', '.join(map(str, sorted(available_values)))}")
        else:
            log("All dice are kept!")

        # Show scores
        if self.turn_accumulated_score > 0:
            log(f"Current set score: {self.get_current_score()} points")

        total_turn_score = self.get_current_total_score()
        log(f"🧩Total turn  score: {total_turn_score} points")


class TheGame:
    """Main game controller for 3-player Lange Strasse"""

    def __init__(self, choice=1, algorithms=DEFAULT_ALGORITHMS):
        self.players = [Player("Player 1"), Player("Player 2"), Player("Player 3")]
        self.current_player_idx = 0
        self.dice_set = DiceSet()
        self.game_over = False
        self.winner = None
        self.final_turn = False
        self.starting_player_idx = 0
        self.turn_number = 1
        self.someone_reached_10k = False  # Track if anyone reached 10k

        if choice == 1:
            pass
        elif choice == 2:
            self.players[1] = AIPlayer("AI Player 2", algorithms[0])
            self.players[2] = AIPlayer("AI Player 3", algorithms[1])
        elif choice == 3:
            self.players[2] = AIPlayer("AI Player 3", algorithms[0])
        elif choice == 4:
            for i in range(3):
                self.players[i] = AIPlayer(f"AI Player {i+1}", algorithms[i])

    def _opponents(self):
        """All players other than the current one."""
        return [p for p in self.players if p is not self.get_current_player()]

    def handle_lange_strasse(self, is_super=False):
        """Current player collects 50¢ (100¢ for a Super Strasse) from each opponent."""
        current_player = self.get_current_player()
        amount = 100 if is_super else 50
        for opponent in self._opponents():
            opponent.add_money(-amount)
            current_player.add_money(amount)

    def handle_totale(self):
        """Current player pays 50¢ to each opponent."""
        current_player = self.get_current_player()
        for opponent in self._opponents():
            opponent.add_money(50)
            current_player.add_money(-50)

    def end_turn(self, turn_score):
        """End current player's turn and add score"""
        current_player = self.get_current_player()
        current_player.total_score += turn_score

        # Mark strich if player got 0 points
        if turn_score == 0:
            current_player.has_strich = True

        log(f"\n{current_player.name} scored {turn_score} points this turn!")
        log(f"{current_player.name}'s total score: {current_player.total_score}")

        # Check win conditions
        if current_player.total_score >= 10000:
            if not self.final_turn:
                # First player to reach 10k
                self.final_turn = True
                self.someone_reached_10k = True
                log(f"\n{current_player.name} reached 10,000 points!")
                log(
                    "⚡ FINAL ROUND! This is the final round we are just going to finish it. ⚡"
                )

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
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

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
            log(f"🎉 {winner.name} won by round {self.turn_number}! Early win bonus!")
            for i, player in enumerate(self.players):
                if i != self.players.index(winner):
                    player.add_money(-50)
                    winner.add_money(50)

        # No strich bonus - everyone without a strich gets 50¢ from everyone else
        no_strich_players = [p for p in self.players if not p.has_strich]
        if no_strich_players:
            log(
                f"🍀 No-strich bonus for: {', '.join(p.name for p in no_strich_players)}"
            )
            for no_strich_player in no_strich_players:
                for other_player in self.players:
                    if other_player != no_strich_player:
                        other_player.add_money(-50)
                        no_strich_player.add_money(50)

        # Under 5000 penalty - pay winner extra 50¢
        for player in self.players:
            if player != winner and player.total_score < 5000:
                log(f"💸 {player.name} pays extra 50¢ for being under 5000 points!")
                player.add_money(-50)
                winner.add_money(50)

        self.winner = winner
        self.game_over = True
        log(f"\n🎉 GAME OVER! {winner.name} wins! 🎉")
        log("Final scores:")
        for i, player in enumerate(sorted_players, 1):
            if isinstance(player, AIPlayer):
                type = f"({player.ai_type})"
            else:
                type = ""
            score = player.total_score
            log(f"{i}. {player.name} {type}: {score} points (Money: {player.money}¢)")

    def start_new_turn(self):
        """Start a new turn for the current player"""
        self.dice_set.reset_for_new_turn()
        current_player = self.get_current_player()
        log(f"\n--- {current_player.name}'s turn ---")

        # Show all players' scores
        scores = ", ".join(f"{p.name}: {p.total_score}" for p in self.players)
        log(f"Current scores: {scores}")

        # Check for Totale at start of turn
        if self.dice_set.check_totale():
            log("💀 TOTALE! No dice can be kept at start of turn! 💀")
            # Note: handle_totale() and end_turn(0) will be called by main.py game loop
            return

    def get_current_player(self):
        """Get the current player object"""
        return self.players[self.current_player_idx]

    def advance_to_decision(self):
        """Resolve forced turn-endings until a real decision is needed or the game ends.

        A player with no keepable dice ends their turn immediately (paying the
        Totale penalty if it happened on the first roll); several dead turns in a
        row are all skipped. Afterwards, either ``game_over`` is set or the
        current player has at least one legal action.
        """
        while not self.game_over:
            dice_set = self.dice_set
            if dice_set.game_over or not dice_set.can_keep_any_dice():
                if dice_set.check_totale():
                    self.handle_totale()
                self.end_turn(0)
                continue
            break

    def process_dice_action(self, dice_values, stop_after):
        """Unified processing for both AI and human dice actions"""
        was_lange_strasse_achieved = self.dice_set.lange_strasse_achieved
        success, message = self.dice_set.keep_dice_by_value(dice_values, stop_after)

        if not success:
            return False, message

        # Pay out Lange Strasse money once, when it is first achieved this set.
        # Then consume the flags so a later Strasse in the same turn pays again.
        if self.dice_set.lange_strasse_achieved and not was_lange_strasse_achieved:
            self.handle_lange_strasse(self.dice_set.super_strasse_achieved)
            self.dice_set.lange_strasse_achieved = False
            self.dice_set.super_strasse_achieved = False

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
