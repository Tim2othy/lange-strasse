"""AI move evaluation for Lange Strasse"""
from typing import Dict, List
import random
from ai_state import GameState, StateExtractor
from ai_actions import Action, ActionGenerator
from collections import Counter

class MoveEvaluator:
    """Evaluate moves for the AI"""

    def __init__(self):
        self.weights = {
            'expected_score': 1.0,
            'risk_factor': 0.5,
            'special_bonus': 2.0,
            'endgame_urgency': 1.5,
            'money_opportunity': 0.3
        }

    def evaluate_action(self, state: GameState, action: Action) -> float:
        """Evaluate how good an action is in the given state"""
        score = 0.0

        # Expected score component
        score += self._evaluate_expected_score(state, action) * self.weights['expected_score']

        # Risk factor (probability of getting 0 points)
        score -= self._evaluate_risk(state, action) * self.weights['risk_factor']

        # Special combination bonus
        score += self._evaluate_special_combinations(state, action) * self.weights['special_bonus']

        # Endgame urgency
        score += self._evaluate_endgame_urgency(state, action) * self.weights['endgame_urgency']

        # Money opportunity
        score += self._evaluate_money_opportunity(state, action) * self.weights['money_opportunity']

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
        # Simulate adding the action to current kept groups
        temp_groups = state.kept_groups.copy()
        temp_groups.append(action.dice_to_keep)

        # Use the scoring calculator (simplified version)
        from collections import Counter

        # Check for special combinations first
        all_dice = []
        for group in temp_groups:
            all_dice.extend(group)

        # Talheim check
        if len(all_dice) == 6:
            counts = Counter(all_dice)
            if len(counts) == 3 and all(count == 2 for count in counts.values()):
                values = sorted(counts.keys())
                if values[1] == values[0] + 1 and values[2] == values[1] + 1:
                    return 1000  # Consecutive Talheim
                else:
                    return 500   # Non-consecutive Talheim

        # Lange Strasse check
        if set(all_dice).issuperset({1, 2, 3, 4, 5, 6}):
            return 1250

        # Regular scoring (simplified)
        score = 0
        dice_counts = Counter(all_dice)

        for value, count in dice_counts.items():
            if count >= 3:
                base_score = 1000 if value == 1 else value * 100
                # Each additional die doubles the score
                for _ in range(count - 3):
                    base_score *= 2
                score += base_score
                # Remove the grouped dice from individual scoring
                dice_counts[value] = count % 3 if count < 6 else 0

        # Add individual 1s and 5s
        score += dice_counts.get(1, 0) * 100
        score += dice_counts.get(5, 0) * 50

        return score

    def _evaluate_risk(self, state: GameState, action: Action) -> float:
        """Evaluate risk of getting 0 points (higher = more risky)"""
        if action.stop_after:
            return 0.0  # No risk if stopping

        # Risk increases with fewer dice remaining and current score
        remaining_dice = sum(1 for d in state.available_dice if d > 0) - len(action.dice_to_keep)

        if remaining_dice <= 2:
            return 0.8  # High risk with few dice
        elif remaining_dice <= 3:
            return 0.4  # Medium risk
        else:
            return 0.1  # Low risk

    def _evaluate_special_combinations(self, state: GameState, action: Action) -> float:
        """Bonus for enabling special combinations"""
        bonus = 0.0

        if state.can_complete_lange_strasse:
            # Check if this action completes Lange Strasse
            all_kept = []
            for group in state.kept_groups:
                all_kept.extend(group)
            all_kept.extend(action.dice_to_keep)

            if set(all_kept).issuperset({1, 2, 3, 4, 5, 6}):
                bonus += 500  # Big bonus for completing Lange Strasse

        if state.can_complete_talheim:
            # Check if this action completes Talheim
            all_kept = []
            for group in state.kept_groups:
                all_kept.extend(group)
            all_kept.extend(action.dice_to_keep)

            if len(all_kept) == 6:
                counts = Counter(all_kept)
                if len(counts) == 3 and all(count == 2 for count in counts.values()):
                    bonus += 200  # Bonus for completing Talheim

        return bonus

    def _evaluate_endgame_urgency(self, state: GameState, action: Action) -> float:
        """Increase urgency in endgame situations"""
        urgency = 0.0
        current_player = state.players[state.current_player_idx]

        # Find highest opponent score
        max_opponent_score = 0
        for i, player in enumerate(state.players):
            if i != state.current_player_idx:
                max_opponent_score = max(max_opponent_score, player.total_score)

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
        money_value = 0.0

        # Lange Strasse gives money
        if state.can_complete_lange_strasse:
            all_kept = []
            for group in state.kept_groups:
                all_kept.extend(group)
            all_kept.extend(action.dice_to_keep)

            if set(all_kept).issuperset({1, 2, 3, 4, 5, 6}):
                money_value += 100  # 50¢ from each opponent

        return money_value

class SimpleAI:
    """Simple AI that uses the move evaluator"""

    def __init__(self):
        self.evaluator = MoveEvaluator()

    def choose_action(self, game) -> Action:
        """Choose the best action for the current game state"""
        state = StateExtractor.extract_state(game)
        valid_actions = ActionGenerator.get_valid_actions(game)

        if not valid_actions:
            # Should not happen in normal play
            return None

        # Evaluate all actions
        best_action = None
        best_score = float('-inf')

        for action in valid_actions:
            score = self.evaluator.evaluate_action(state, action)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def get_action_explanation(self, game, action: Action) -> str:
        """Get explanation for why this action was chosen"""
        state = StateExtractor.extract_state(game)
        score = self.evaluator.evaluate_action(state, action)
        return f"AI chose: {action} (score: {score:.2f})"
