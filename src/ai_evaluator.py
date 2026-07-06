"""AI move evaluation for Lange Strasse"""

from ai_actions import Action
from algorithms.heuristic import MoveEvaluator
from algorithms.td import td_action_score
from algorithms.dp import action_value
from game_state import GameState


def random_action(state: GameState, actions: list[Action]) -> Action:
    """Choose a random valid action"""
    import random

    return random.choice(actions)


def best_action(state: GameState, actions: list[Action], algorithm: str) -> Action:
    """Choose action with highest value based on some algorithm"""

    if not actions:
        exception = "No valid actions available for the current state."
        raise ValueError(exception)

    # Evaluate all actions
    best_action = None
    best_score = float("-inf")

    for action in actions:
        if algorithm == "simple":
            score = MoveEvaluator().evaluate_action(state, action)
        elif algorithm == "dp":
            score = action_value(state, action)
        elif algorithm in ("td", "td_full", "td_small"):
            score = td_action_score(state, action, algorithm)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if score > best_score:
            best_score = score
            best_action = action

    text = "No valid action found despite valid_actions being non-empty."
    assert best_action is not None, text
    return best_action
