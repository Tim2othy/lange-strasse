"""Canonical game-state snapshot and its ML feature encoding.

Design: ``GameState`` stores only the *minimal, non-derivable* ("Markov") state
of a decision point. Anything computable from those fields is exposed as a
``@property`` (computed on access, so it can never go stale) rather than being
stored. The richer, redundant representation a learning model wants is built by
``StateExtractor``: ``_atom_features`` computes every feature once (each with a
name), and a model is just a list of those names (see ``select_features`` and the
``TD_*_KEYS`` lists in ``algorithms/td.py``).
"""

from collections import Counter
from dataclasses import dataclass
from typing import List

from game.scoring import (
    can_keep_any,
    flatten,
    is_lange_strasse,
    merge_kept,
    score_groups,
    talheim_score,
)


@dataclass
class PlayerState:
    """State information for a single player."""

    total_score: int
    has_strich: bool
    money: int


@dataclass
class GameState:
    """Canonical snapshot of one decision point.

    Only non-derivable fields are stored. Derived quantities (current-set score,
    whether a special can be completed, ...) are ``@property`` views below.
    """

    # Dice information
    available_dice: List[int]  # values available to keep this roll, e.g. [1, 4]
    kept_groups: List[List[int]]  # Groups of kept dice

    # Turn history
    turn_accumulated_score: int  # points banked from previous sets this turn
    roll_count: int  # rolls in the current set (needed for Super Strasse)

    # Player information
    players: List[PlayerState]
    current_player_idx: int
    starting_player_idx: int

    # Game context
    turn_number: int
    is_final_round: bool

    # ------------------------------------------------------------------ #
    # Derived views: computed, never stored -> never inconsistent.
    # ------------------------------------------------------------------ #
    @property
    def current_player(self) -> PlayerState:  # not being used can maybe be removed
        return self.players[self.current_player_idx]

    @property
    def current_set_score(self) -> int:
        """Score of the current dice set (a pure function of kept_groups)."""
        return score_groups(self.kept_groups)

    @property
    def total_turn_score(self) -> int:
        """Everything the current player would bank by stopping right now."""
        return self.turn_accumulated_score + self.current_set_score

    @property
    def can_complete_lange_strasse(self) -> bool:
        return is_lange_strasse(flatten(self.kept_groups) + self.available_dice)

    @property
    def can_complete_talheim(self) -> bool:
        return talheim_score(flatten(self.kept_groups) + self.available_dice) > 0

    @property
    def can_keep_any(self) -> bool:
        return can_keep_any(self.available_dice, flatten(self.kept_groups))


"""
Example GameState

    available_dice: [1,4]
    kept_groups: [[2,2,2],[5]]
    turn_accumulated_score: 500      # banked from earlier sets this turn
    roll_count: 2
    players: [
        PlayerState(total_score=5350, has_strich=True, money=200),
        PlayerState(total_score=6050, has_strich=True, money=-50),
        PlayerState(total_score=4500, has_strich=False, money=-150),
    ]
    current_player_idx: 2
    starting_player_idx: 1
    turn_number: 8
    is_final_round: False

    # derived (not stored):
    current_set_score -> 250         # 2,2,2 (200) + 5 (50)
    total_turn_score  -> 750
    can_complete_lange_strasse -> False
"""


class StateExtractor:
    """Build a GameState from a live game, and encode it for ML models."""

    @staticmethod
    def extract_state(game) -> GameState:
        """Snapshot the current decision point of ``game`` as a GameState."""
        dice_set = game.dice_set
        return GameState(
            available_dice=dice_set.get_available_dice_values(),
            kept_groups=[group[:] for group in dice_set.kept_groups],
            turn_accumulated_score=dice_set.turn_accumulated_score,
            roll_count=dice_set.roll_count,
            players=[
                PlayerState(p.total_score, p.has_strich, p.money) for p in game.players
            ],
            current_player_idx=game.current_player_idx,
            starting_player_idx=game.starting_player_idx,
            turn_number=game.turn_number,
            is_final_round=game.final_turn,
        )

    # ------------------------------------------------------------------ #
    # Feature menu: the single place any feature is computed. A model is an
    # ordered list of atom names (see ``select_features`` and the TD_*_KEYS
    # lists in algorithms/td.py). Adding a model to compare is just choosing
    # keys -- no new encoder, no duplicated normalization.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _atom_features(after: GameState, ends_turn: bool) -> dict[str, float]:
        """Every atomic afterstate feature, keyed by name and normalized.

        Players are named ego-centrically: ``me`` (acting), ``p2`` (next to act),
        ``p3`` (last). The kept dice are offered two ways -- raw per-face counts
        (``kept{n}``, with ``grouped1``/``grouped5`` for 1s/5s inside a triplet) and
        triplet strength (``group{n}``, geometric in size, with ``loose1``/``loose5``)
        -- because different models want different ones.
        """
        feats: dict[str, float] = {}

        kept_counts = Counter(flatten(after.kept_groups))
        for face in range(1, 7):
            feats[f"kept{face}"] = kept_counts.get(face, 0) / 6.0

        # merge_kept keeps at most one triplet group per face and stores loose
        # 1s/5s as separate length-1 groups, so triplet_size[face] is well-defined.
        triplet_size = {face: 0 for face in range(1, 7)}
        loose = {1: 0, 5: 0}
        for group in after.kept_groups:
            if len(group) >= 3:
                triplet_size[group[0]] = len(group)
            elif group[0] in loose:
                loose[group[0]] += len(group)
        for face in range(1, 7):
            size = triplet_size[face]
            feats[f"group{face}"] = 2 ** (size - 3) / 8.0 if size >= 3 else 0.0
        feats["loose1"] = loose[1] / 6.0
        feats["loose5"] = loose[5] / 6.0
        feats["grouped1"] = triplet_size[1] / 6.0  # 1s sitting inside a triplet
        feats["grouped5"] = triplet_size[5] / 6.0  # 5s sitting inside a triplet

        # Turn flow (raw + derived scores).
        feats["current_set_score"] = after.current_set_score / 2000.0
        feats["turn_accumulated"] = after.turn_accumulated_score / 10000.0
        feats["total_turn_score"] = after.total_turn_score / 10000.0
        feats["roll_count"] = after.roll_count / 6.0
        feats["dice_left"] = (6 - sum(kept_counts.values())) / 6.0

        # Situation flags (derived).
        feats["flag_lange_strasse"] = 1.0 if after.can_complete_lange_strasse else 0.0
        feats["flag_talheim"] = 1.0 if after.can_complete_talheim else 0.0
        feats["flag_keep_any"] = 1.0 if after.can_keep_any else 0.0

        # Position in the round / game.
        n = len(after.players)
        offset = (after.current_player_idx - after.starting_player_idx) % n
        feats["seat_offset"] = offset / (n - 1) if n > 1 else 0.0
        feats["turn_number"] = after.turn_number / 20.0
        feats["is_final_round"] = 1.0 if after.is_final_round else 0.0
        feats["ends_turn"] = 1.0 if ends_turn else 0.0

        # Players, acting player first.
        labels = ["me", "p2", "p3"]
        for i in range(n):
            p = after.players[(after.current_player_idx + i) % n]
            label = labels[i] if i < len(labels) else f"p{i + 1}"
            feats[f"score_{label}"] = p.total_score / 10000.0
            feats[f"strich_{label}"] = 1.0 if p.has_strich else 0.0
            feats[f"money_{label}"] = p.money / 1000.0

        return feats

    # ------------------------------------------------------------------ #
    # Afterstate: the position an action leaves behind, before the next
    # roll, from the acting player's perspective. Every model shares this
    # transition and differs only in which atoms it reads from the result.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _afterstate(state: GameState, action) -> tuple[GameState, bool]:
        """The position ``action`` leaves behind, before the next roll, from the
        acting player's perspective -- plus whether the action ended the turn.

        Banking the turn on a stop or Talheim, taking hot dice when all six are
        kept, otherwise leaving the set at risk. A completed Lange Strasse also pays
        the mover immediately, so the afterstate money reflects that transfer.
        """
        prev = state.turn_accumulated_score
        merged = merge_kept(state.kept_groups, action.dice_to_keep)
        values = flatten(merged)
        set_score = score_groups(merged)

        ends_turn = action.stop_after or talheim_score(values) > 0
        hot_dice = not ends_turn and len(values) == 6

        me = state.current_player_idx
        players = [
            PlayerState(p.total_score, p.has_strich, p.money) for p in state.players
        ]

        # A newly completed Lange Strasse pays the mover immediately -- 50c (100c for
        # a Super Strasse, completed on the 3rd+ roll) from every opponent -- so this
        # mirrors game.handle_lange_strasse. Without it, money would be identical
        # across every action at a decision and could never influence the choice.
        if is_lange_strasse(values) and not is_lange_strasse(
            flatten(state.kept_groups)
        ):
            amount = 200 if state.roll_count >= 3 else 70
            players = [
                PlayerState(
                    p.total_score,
                    p.has_strich,
                    p.money + (amount * (len(players) - 1) if i == me else -amount),
                )
                for i, p in enumerate(players)
            ]

        if ends_turn:
            # Turn banked: add the turn score to my total; nothing left at risk.
            mine = players[me]
            players[me] = PlayerState(
                mine.total_score + prev + set_score, mine.has_strich, mine.money
            )
            kept_groups, accumulated, roll_count = [], 0, 0
        elif hot_dice:
            # All six kept: bank the set into the accumulator, roll a fresh six.
            kept_groups, accumulated, roll_count = [], prev + set_score, 0
        else:
            # Still my turn, this set at risk.
            kept_groups, accumulated, roll_count = merged, prev, state.roll_count

        after = GameState(
            available_dice=[],  # afterstate is before the next roll
            kept_groups=kept_groups,
            turn_accumulated_score=accumulated,
            roll_count=roll_count,
            players=players,
            current_player_idx=me,  # keep the mover's perspective
            starting_player_idx=state.starting_player_idx,
            turn_number=state.turn_number,
            is_final_round=state.is_final_round,
        )
        return after, ends_turn

    @staticmethod
    def select_features(state: GameState, action, keys: list[str]) -> List[float]:
        """The afterstate atoms named in ``keys``, in order (an unknown key raises)."""
        atoms = StateExtractor.afterstate_atoms(state, action)
        return [atoms[key] for key in keys]

    @staticmethod
    def afterstate_atoms(state: GameState, action) -> dict[str, float]:
        """The named afterstate features for ``action`` -- the feature menu, valued.

        Returns the whole atom dict so a caller can add its own extra atoms (e.g. a
        solver's value, which can't live here) before selecting. ``feature_menu``
        lists the names.
        """
        after, ends_turn = StateExtractor._afterstate(state, action)
        return StateExtractor._atom_features(after, ends_turn)

    @staticmethod
    def feature_menu() -> list[str]:
        """All atom names available to ``select_features`` (in _atom_features order)."""
        blank = GameState(
            available_dice=[],
            kept_groups=[],
            turn_accumulated_score=0,
            roll_count=0,
            players=[PlayerState(0, False, 0) for _ in range(3)],
            current_player_idx=0,
            starting_player_idx=0,
            turn_number=0,
            is_final_round=False,
        )
        return list(StateExtractor._atom_features(blank, False))
