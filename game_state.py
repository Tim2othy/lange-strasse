"""Canonical game-state snapshot and its ML feature encoding.

Design: ``GameState`` stores only the *minimal, non-derivable* ("Markov") state
of a decision point. Anything computable from those fields is exposed as a
``@property`` (computed on access, so it can never go stale) rather than being
stored. The rich, redundant representation a learning model wants lives in
``to_vector`` -- that's the right place for derived features, because handing a
network the score/flags directly makes it learn far faster than re-deriving them.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List

from scoring import (
    can_keep_any,
    flatten,
    is_lange_strasse,
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
    def current_player(self) -> PlayerState:
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

    @staticmethod
    def to_vector(state: GameState) -> List[float]:
        """Encode a GameState as a fixed-length, normalized feature vector.

        Players are ordered ego-centrically
        Derived features (scores, completion flags) are included on purpose -- redundancy
        speeds learning.
        """
        vector: List[float] = []

        # Dice as per-face counts (position-free, fixed length).
        available_counts = Counter(state.available_dice)
        kept_counts = Counter(flatten(state.kept_groups))
        for face in range(1, 7):
            vector.append(available_counts.get(face, 0) / 6.0)
        for face in range(1, 7):
            vector.append(kept_counts.get(face, 0) / 6.0)

        # Scoring (derived).
        vector.append(state.current_set_score / 2000.0)
        vector.append(state.turn_accumulated_score / 10000.0)
        vector.append(state.total_turn_score / 10000.0)
        vector.append(state.roll_count / 6.0)

        # Situation flags (derived).
        vector.append(1.0 if state.can_complete_lange_strasse else 0.0)
        vector.append(1.0 if state.can_complete_talheim else 0.0)
        vector.append(1.0 if state.can_keep_any else 0.0)

        # Players, current player first.
        n = len(state.players)
        for offset in range(n):
            p = state.players[(state.current_player_idx + offset) % n]
            vector.append(p.total_score / 10000.0)
            vector.append(1.0 if p.has_strich else 0.0)
            vector.append(p.money / 1000.0)

        # Game context.
        vector.append(state.turn_number / 20.0)
        vector.append(1.0 if state.is_final_round else 0.0)

        return vector
