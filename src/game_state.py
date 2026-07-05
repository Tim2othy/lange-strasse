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

        Players are ordered ego-centrically (current player first) so the model
        generalizes across seats. Derived features (scores, completion flags) are
        included on purpose -- the redundancy speeds learning.

        Returns a plain list of floats (no numpy dependency in the game code); an
        RL trainer can wrap it with ``np.asarray(vec, dtype=np.float32)``.
        """
        vector: List[float] = []

        # Dice as per-face counts (position-free, fixed length).
        available_counts = Counter(state.available_dice)
        kept_counts = Counter(flatten(state.kept_groups))
        for face in range(1, 7):
            vector.append(available_counts.get(face, 0) / 6.0)
        for face in range(1, 7):
            vector.append(kept_counts.get(face, 0) / 6.0)

        # 1s and 5s that sit in a triplet (extendable -- each extra die doubles
        # the group) vs. kept individually. The plain counts + total score can't
        # recover this distinction. Only 1s/5s are ambiguous: a kept 2/3/4/6 is
        # always part of a triplet.
        grouped = Counter()
        for group in state.kept_groups:
            if len(group) >= 3:
                grouped[group[0]] += len(group)
        vector.append(grouped.get(1, 0) / 6.0)
        vector.append(grouped.get(5, 0) / 6.0)

        # Scoring (derived).
        vector.append(state.current_set_score / 2000.0)
        vector.append(state.turn_accumulated_score / 10000.0)
        vector.append(state.total_turn_score / 10000.0)
        vector.append(state.roll_count / 6.0)
        vector.append((6 - sum(kept_counts.values())) / 6.0)  # dice left to roll

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

    @staticmethod
    def action_features(state: GameState, action) -> List[float]:
        """Feature vector of the position reached by ``action``, before the next roll.

        This is the single place where an *action's* features are built. It forms
        the "afterstate" -- banking the turn on a stop or Talheim, taking hot dice
        when all six are kept, otherwise leaving the set at risk -- always from the
        acting player's perspective, and encodes it with ``to_vector`` plus a flag
        for whether the turn ended. Algorithm-specific extras (e.g. a solver's own
        value estimate) are appended by the algorithm, not here, so this stays
        free of any dependency on the algorithm layer.
        """
        prev = state.turn_accumulated_score
        merged = merge_kept(state.kept_groups, action.dice_to_keep)
        values = flatten(merged)
        set_score = score_groups(merged)

        ends_turn = action.stop_after or talheim_score(values) > 0
        hot_dice = not ends_turn and len(values) == 6

        me = state.current_player_idx
        players = [PlayerState(p.total_score, p.has_strich, p.money) for p in state.players]

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

        features = StateExtractor.to_vector(after)
        features.append(1.0 if ends_turn else 0.0)  # did the turn just end?
        return features

    @staticmethod
    def action_feature_size(num_players: int = 3) -> int:
        """Length of an action_features() vector (to_vector + the turn-over flag)."""
        return StateExtractor.vector_size(num_players) + 1

    @staticmethod
    def vector_size(num_players: int = 3) -> int:
        """Length of a to_vector() encoding, for sizing a model's input layer.

        Computed from an empty state so it can never drift from to_vector().
        """
        dummy = GameState(
            available_dice=[],
            kept_groups=[],
            turn_accumulated_score=0,
            roll_count=0,
            players=[PlayerState(0, False, 0) for _ in range(num_players)],
            current_player_idx=0,
            starting_player_idx=0,
            turn_number=0,
            is_final_round=False,
        )
        return len(StateExtractor.to_vector(dummy))
