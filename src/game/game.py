"""The stateful side of Lange Strasse: Player, DiceSet, and the Game controller.

All rule questions (what may be kept, what it scores, which actions exist) are
answered by game.rules; this module only holds state and turn/money flow.
"""

import random
from collections import Counter

from game import rules
from game.rules import (
    NUM_DICE,
    STOP_MIN,
    can_keep_any,
    flatten,
    is_lange_strasse,
    is_valid_keep,
    merge_kept,
    score_groups,
    talheim_score,
)
from log import log


class Player:
    """A seat at the table: score, money (in cents), and strich state."""

    def __init__(self, name):
        self.name = name
        self.total_score = 0
        self.money = 0
        self.has_strich = False

    def describe(self):
        return self.name

    def add_money(self, cents):
        """Add money (can be negative for losses)."""
        self.money += cents
        if cents > 0:
            log(f"💰 {self.name} gains {cents}¢!")
        else:
            log(f"💸 {self.name} loses {abs(cents)}¢!")


class DiceSet:
    """The dice of one player's turn: the current roll, the kept groups, and
    the turn's score state."""

    def __init__(self):
        self.reset_for_new_turn()

    def reset_for_new_turn(self):
        """Fresh turn: nothing kept, nothing banked, roll all six dice."""
        self.kept_groups: list[list[int]] = []  # kept dice of the current set
        self.turn_accumulated_score = 0  # banked from earlier sets this turn
        self.roll_count = 0  # rolls in the current set
        self.turn_over = False  # stopped, Talheim'd, or busted
        self.busted = False  # ended with no keepable dice -> 0 points
        # A completed (Super) Strasse; read and consumed by Game.apply_action
        # so the money is paid out exactly once.
        self.lange_strasse_achieved = False
        self.super_strasse_achieved = False
        self._roll(NUM_DICE)

    # ------------------------------------------------------------------ #
    # Rolling
    # ------------------------------------------------------------------ #
    def _roll(self, n):
        self.available = [random.randint(1, 6) for _ in range(n)]
        self.roll_count += 1
        self._check_bust()

    def force_roll(self, values):
        """Debug/test helper: replace the current roll's dice with ``values``."""
        self.available = list(values)
        self.turn_over = self.busted = False
        self._check_bust()

    def _check_bust(self):
        if not can_keep_any(self.available, flatten(self.kept_groups)):
            self.turn_over = self.busted = True
            if self.is_totale:
                log("💀 TOTALE! No dice can be kept! 💀")

    @property
    def is_totale(self):
        """Busted on the first roll of a fresh set of six."""
        return self.busted and self.roll_count == 1

    # ------------------------------------------------------------------ #
    # Keeping dice
    # ------------------------------------------------------------------ #
    def keep(self, values, stop_after=False):
        """Keep ``values`` from the available dice, then stop or roll the rest.

        Returns ``(success, message)``. On success the turn state is advanced:
        Talheim or stopping sets ``turn_over``; keeping all six banks the set
        and rolls a fresh six ("hot dice"); otherwise the remaining dice are
        rolled, which may bust (``busted``).
        """
        if self.turn_over:
            return False, "The turn is already over."

        available = Counter(self.available)
        requested = Counter(values)
        for value, count in requested.items():
            if available[value] < count:
                return False, (
                    f"Only {available[value]} dice with value {value} available, "
                    f"but {count} requested."
                )

        is_valid, error = is_valid_keep(list(values), flatten(self.kept_groups))
        if not is_valid:
            return False, error

        merged = merge_kept(self.kept_groups, values)
        if stop_after:
            if len(flatten(merged)) == NUM_DICE:
                return (
                    False,
                    "Cannot stop when keeping all 6 dice. Must keep 5 or fewer.",
                )
            projected = score_groups(merged)
            if projected < STOP_MIN:
                return False, (
                    f"Cannot stop with less than {STOP_MIN} points from current "
                    f"dice set. Would score {projected} points."
                )

        available.subtract(requested)
        self.available = list(available.elements())
        self.kept_groups = merged
        kept_values = flatten(merged)

        # Talheim ends the turn on the spot.
        talheim = talheim_score(kept_values)
        if talheim:
            self.turn_over = True
            log(f"🎯 TALHEIM! Three pairs worth {talheim} points! 🎯")
            return True, f"Talheim! Turn ends with {self.total_turn_score} points"

        if is_lange_strasse(kept_values):
            self.lange_strasse_achieved = True
            if self.roll_count >= 3:
                self.super_strasse_achieved = True
                log("🌟 SUPER STRASSE! Lange Strasse on the 3rd roll! 🌟")
            else:
                log("🎉 LANGE STRASSE! 1-2-3-4-5-6 completed! 🎉")

        if stop_after:
            self.turn_over = True
            return True, f"Turn ended! Score this turn: {self.total_turn_score} points"

        if not self.available:
            # Hot dice: all six kept -- bank this set and roll a fresh six.
            self.turn_accumulated_score += self.current_set_score
            log(f"All 6 dice kept! Turn total so far: {self.turn_accumulated_score}")
            self.kept_groups = []
            self.roll_count = 0
            self._roll(NUM_DICE)
            return True, (
                f"All dice used! Rolling again with {self.turn_accumulated_score} "
                "points accumulated this turn."
            )

        self._roll(len(self.available))
        if self.busted:
            return True, "No valid moves available! Turn ends with 0 points"
        return True, "Dice kept successfully"

    # ------------------------------------------------------------------ #
    # Scores
    # ------------------------------------------------------------------ #
    @property
    def current_set_score(self):
        """Score of the currently kept set (Talheim / Strasse included)."""
        return score_groups(self.kept_groups)

    @property
    def total_turn_score(self):
        """Everything the player would bank by ending the turn now."""
        return self.turn_accumulated_score + self.current_set_score

    def display(self):
        """Log the current dice state (silent unless verbose logging is on)."""
        if self.turn_over:
            log("\nTurn over! Score: 0 points (no valid moves)")
            return

        if self.kept_groups:
            shown = []
            for group in self.kept_groups:
                if len(group) >= 3:
                    shown.append(f"[{', '.join(map(str, sorted(group)))}]")
                else:
                    shown.extend(map(str, sorted(group)))
            log(f"Kept dice: {', '.join(shown)}")

        log(f"Available dice: {', '.join(map(str, sorted(self.available)))}")
        if self.turn_accumulated_score:
            log(f"Current set score: {self.current_set_score} points")
        log(f"🧩 Total turn score: {self.total_turn_score} points")


class TheGame:
    """Game controller for 3-player Lange Strasse: turn order, win condition,
    and the money economy."""

    def __init__(self, players=None):
        self.players = players or [Player(f"Player {i + 1}") for i in range(3)]
        self.current_player_idx = 0
        self.starting_player_idx = 0
        self.turn_number = 1
        self.final_round = False  # someone has reached 10,000 points
        self.game_over = False
        self.winner = None
        self._announce_turn()
        self.dice_set = DiceSet()

    @property
    def current_player(self):
        return self.players[self.current_player_idx]

    # ------------------------------------------------------------------ #
    # Money
    # ------------------------------------------------------------------ #
    def _transfer(self, payer, receiver, cents):
        payer.add_money(-cents)
        receiver.add_money(cents)

    def _collect_from_others(self, player, cents):
        """``player`` receives ``cents`` from every other player (negative: pays them)."""
        for other in self.players:
            if other is not player:
                self._transfer(other, player, cents)

    def handle_lange_strasse(self, is_super=False):
        """Current player collects 50¢ (100¢ for a Super Strasse) from each opponent."""
        self._collect_from_others(self.current_player, 100 if is_super else 50)

    def handle_totale(self):
        """Current player pays 50¢ to each opponent."""
        self._collect_from_others(self.current_player, -50)

    # ------------------------------------------------------------------ #
    # Turn flow
    # ------------------------------------------------------------------ #
    def legal_actions(self):
        """All legal keep/stop actions at the current decision point."""
        if self.game_over or self.dice_set.turn_over:
            return []
        return rules.legal_actions(self.dice_set.available, self.dice_set.kept_groups)

    def apply_action(self, action):
        """Apply one keep/stop decision for the current player.

        Returns ``(success, message)``. Pays out a completed Lange Strasse and
        banks the turn when the action ends it. Busts are left for
        ``advance_to_decision`` so dead rolls and Totales are resolved in one place.
        """
        success, message = self.dice_set.keep(action.dice_to_keep, action.stop_after)
        if not success:
            return False, message

        if self.dice_set.lange_strasse_achieved:
            self.handle_lange_strasse(self.dice_set.super_strasse_achieved)
            self.dice_set.lange_strasse_achieved = False
            self.dice_set.super_strasse_achieved = False

        if self.dice_set.turn_over and not self.dice_set.busted:
            self.end_turn(self.dice_set.total_turn_score)
        return True, message

    def advance_to_decision(self):
        """Resolve busted turns (paying Totales) until the current player has a
        real decision or the game is over."""
        while not self.game_over and self.dice_set.busted:
            if self.dice_set.is_totale:
                self.handle_totale()
            self.end_turn(0)

    def end_turn(self, turn_score):
        """Bank ``turn_score`` for the current player and pass the turn on."""
        player = self.current_player
        player.total_score += turn_score
        if turn_score == 0:
            player.has_strich = True

        log(f"\n{player.name} scored {turn_score} points this turn!")
        log(f"{player.name}'s total score: {player.total_score}")

        if player.total_score >= 10000 and not self.final_round:
            self.final_round = True
            log(f"\n{player.name} reached 10,000 points!")
            log("⚡ FINAL ROUND! Everyone left gets one last turn. ⚡")

        next_idx = (self.current_player_idx + 1) % len(self.players)
        if self.final_round and next_idx == self.starting_player_idx:
            self.finish_game()
            return

        self.current_player_idx = next_idx
        if next_idx == self.starting_player_idx:
            self.turn_number += 1
        self.start_new_turn()

    def start_new_turn(self):
        self._announce_turn()
        self.dice_set.reset_for_new_turn()

    def _announce_turn(self):
        log(f"\n--- {self.current_player.name}'s turn ---")
        scores = ", ".join(f"{p.name}: {p.total_score}" for p in self.players)
        log(f"Current scores: {scores}")

    # ------------------------------------------------------------------ #
    # Game end
    # ------------------------------------------------------------------ #
    def finish_game(self):
        """Settle all end-of-game money and declare the winner."""
        ranked = sorted(self.players, key=lambda p: p.total_score, reverse=True)
        winner, second, third = ranked

        # Basic stakes: winner collects from 2nd and 3rd place.
        self._transfer(second, winner, 50)
        self._transfer(third, winner, 70)

        if self.turn_number <= 10:
            log(f"🎉 {winner.name} won by round {self.turn_number}! Early win bonus!")
            self._collect_from_others(winner, 50)

        no_strich_players = [p for p in self.players if not p.has_strich]
        if no_strich_players:
            log(
                f"🍀 No-strich bonus for: {', '.join(p.name for p in no_strich_players)}"
            )
            for player in no_strich_players:
                self._collect_from_others(player, 50)

        for player in self.players:
            if player is not winner and player.total_score < 5000:
                log(f"💸 {player.name} pays extra 50¢ for being under 5000 points!")
                self._transfer(player, winner, 50)

        self.winner = winner
        self.game_over = True
        log(f"\n🎉 GAME OVER! {winner.name} wins! 🎉")
        log("Final scores:")
        for i, player in enumerate(ranked, 1):
            log(
                f"{i}. {player.describe()}: {player.total_score} points "
                f"(Money: {player.money}¢)"
            )
