"""AI move evaluation for Lange Strasse"""

from ai_actions import Action
from game_state import GameState
from scoring import flatten, is_lange_strasse, score_groups, talheim_score


class MoveEvaluator:
    """Evaluate moves for the AI"""

    def __init__(self):
        self.weights = {
            "expected_score": 1.0,
            "risk_factor": 0.5,
            "special_bonus": 2.0,
            "endgame_urgency": 1.5,
            "money_opportunity": 0.3,
        }

    def evaluate_action(self, state: GameState, action: Action) -> float:
        """Evaluate how good an action is in the given state"""
        score = 0.0

        # Expected score component
        score += (
            self._evaluate_expected_score(state, action)
            * self.weights["expected_score"]
        )

        # Risk factor (probability of getting 0 points)
        score -= self._evaluate_risk(state, action) * self.weights["risk_factor"]

        # Special combination bonus
        score += (
            self._evaluate_special_combinations(state, action)
            * self.weights["special_bonus"]
        )

        # Endgame urgency
        score += (
            self._evaluate_endgame_urgency(state, action)
            * self.weights["endgame_urgency"]
        )

        # Money opportunity
        score += (
            self._evaluate_money_opportunity(state, action)
            * self.weights["money_opportunity"]
        )

        return score

    def _evaluate_expected_score(self, state: GameState, action: Action) -> float:
        """Evaluate expected score from this action"""
        if action.stop_after:
            # If stopping, return the guaranteed score
            return state.turn_score + self._calculate_action_score(state, action)
        else:
            # If continuing, estimate expected value
            immediate_score = self._calculate_action_score(state, action)
            # Simple heuristic: expect to continue for 1-2 more rounds
            continuation_value = immediate_score * 0.6  # Discounted future value
            return immediate_score + continuation_value

    def _calculate_action_score(self, state: GameState, action: Action) -> int:
        """Calculate immediate score from keeping these dice"""
        return score_groups(state.kept_groups + [action.dice_to_keep])

    def _evaluate_risk(self, state: GameState, action: Action) -> float:
        """Evaluate risk of getting 0 points (higher = more risky)"""
        if action.stop_after:
            return 0.0  # No risk if stopping

        # Risk increases with fewer dice remaining and current score
        remaining_dice = sum(1 for d in state.available_dice if d > 0) - len(
            action.dice_to_keep
        )

        if remaining_dice <= 2:
            return 0.8  # High risk with few dice
        elif remaining_dice <= 3:
            return 0.4  # Medium risk
        else:
            return 0.1  # Low risk

    def _evaluate_special_combinations(self, state: GameState, action: Action) -> float:
        """Bonus for enabling special combinations"""
        bonus = 0.0
        combined = flatten(state.kept_groups) + list(action.dice_to_keep)

        if state.can_complete_lange_strasse and is_lange_strasse(combined):
            bonus += 500  # Big bonus for completing Lange Strasse

        if state.can_complete_talheim and talheim_score(combined):
            bonus += 200  # Bonus for completing Talheim

        return bonus

    def _evaluate_endgame_urgency(self, state: GameState, action: Action) -> float:
        """Increase urgency in endgame situations"""
        urgency = 0.0
        current_player = state.players[state.current_player_idx]

        # Find highest opponent score
        max_opponent_score = max(
            (
                p.total_score
                for i, p in enumerate(state.players)
                if i != state.current_player_idx
            ),
            default=0,
        )

        # If opponents are close to winning, be more aggressive
        if max_opponent_score >= 8000:
            if action.stop_after and state.turn_score >= 300:
                urgency += 100  # Take guaranteed points

        # If we're close to winning, be more conservative
        if current_player.total_score >= 8000:
            if action.stop_after:
                urgency += 50  # Prefer to secure points

        return urgency

    def _evaluate_money_opportunity(self, state: GameState, action: Action) -> float:
        """Evaluate money-making opportunities"""
        # Lange Strasse takes 50¢ from each opponent.
        combined = flatten(state.kept_groups) + list(action.dice_to_keep)
        if state.can_complete_lange_strasse and is_lange_strasse(combined):
            return 100.0
        return 0.0


def random_action(state, valid_actions) -> Action:
    """Choose a random valid action"""
    import random

    return random.choice(valid_actions)


def simple_action(state, valid_actions) -> Action:
    """Choose action based on simple heuristic"""

    if not valid_actions:
        exception = "No valid actions available for the current state."
        raise ValueError(exception)

    # Evaluate all actions
    best_action = None
    best_score = float("-inf")

    for action in valid_actions:
        score = MoveEvaluator().evaluate_action(state, action)
        if score > best_score:
            best_score = score
            best_action = action
    assert (
        best_action is not None
    ), "No valid action found despite valid_actions being non-empty."

    return best_action

    def get_action_explanation(self, game, action: Action) -> str:
        """Get explanation for why this action was chosen"""
        # state = StateExtractor.extract_state(game)
        # score = self.evaluator.evaluate_action(state, action)
        return f"AI chose: {action} (score: {score:.2f})"
