"""Point the trained TD AI at one specific decision and see how it thinks.

Set the hand in config.py:

    HAND_AVAILABLE   = [1, 4, 2, 3, 5, 1]   # dice showing on the current roll
    HAND_KEPT        = []                    # dice already set aside this turn
    HAND_ACCUMULATED = 0                     # points banked earlier this turn

Then, from the src/ directory:

    python -m algorithms.hand_eval

For that hand it lists every legal action with the value the TD model assigns
it -- the model's estimate of my final game money, in cents -- ranks them, and
for the top action breaks down which weight*feature terms drove the value. It
also shows which features most explain why the winner beat the runner-up.

Context note: the imaginary player is fresh (score 0, money 0, mid-game), so the
absolute cent numbers aren't a real-game prediction. What's meaningful is the
*ranking* and the *differences* between actions -- the context features are
identical across actions and cancel out, which is exactly the point of scoring
afterstates.
"""

from types import SimpleNamespace

from ai_player import ActionGenerator
from algorithms.dp import action_value
from algorithms.td import _model, td_features
from config import HAND_ACCUMULATED, HAND_AVAILABLE, HAND_KEPT, TD_INTERP
from game.rules import flatten, merge_kept
from game_state import GameState, PlayerState
from interp import feature_names


def build_state(available, kept, accumulated) -> GameState:
    """A synthetic decision point: fresh 3-player game, my hand as given."""
    return GameState(
        available_dice=list(available),
        kept_groups=merge_kept([], kept),  # canonical grouping of the kept dice
        turn_accumulated_score=accumulated,
        roll_count=1,
        players=[PlayerState(0, False, 0) for _ in range(3)],
        current_player_idx=0,
        starting_player_idx=0,
        turn_number=1,
        is_final_round=False,
    )


def legal_actions(state: GameState):
    """Reuse the real action generator by wrapping the state in a minimal stand-in
    for the ``game.dice_set`` it reads (so the rules can never drift from play)."""
    n_kept = len(flatten(state.kept_groups))
    dice_set = SimpleNamespace(
        game_over=False,
        kept_groups=state.kept_groups,
        kept_dice=[True] * n_kept + [False] * (6 - n_kept),  # sum() = number kept
        get_available_dice_values=lambda: list(state.available_dice),
    )
    return ActionGenerator.get_valid_actions(SimpleNamespace(dice_set=dice_set))


def _bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    """A little text gauge so the ranking is scannable at a glance."""
    frac = 0.0 if hi == lo else (value - lo) / (hi - lo)
    filled = round(frac * width)
    return "#" * filled + "-" * (width - filled)


def main() -> None:
    state = build_state(HAND_AVAILABLE, HAND_KEPT, HAND_ACCUMULATED)
    model = _model(TD_INTERP)
    names = feature_names()

    kept_str = " ".join(map(str, flatten(state.kept_groups))) or "(none)"
    print(f"\nHand:  available = {HAND_AVAILABLE}   kept = {kept_str}   "
          f"banked this turn = {HAND_ACCUMULATED}")

    actions = legal_actions(state)
    if not actions:
        print("No legal action for this hand (nothing keepable -- a Totale).")
        return

    # Score every action: TD value (w . features) and the DP turn-EV for reference.
    rows = []
    for a in actions:
        feats = td_features(state, a)
        rows.append((a, feats, model.value(feats), action_value(state, a)))
    rows.sort(key=lambda r: r[2], reverse=True)

    lo, hi = rows[-1][2], rows[0][2]
    print(f"\n{len(rows)} legal actions, best first "
          f"(TD value = model's estimate of my final money):\n")
    print(f"  {'action':<26}{'TD cents':>9}  {'DP pts':>7}  value-bar")
    for a, _feats, val, dp in rows:
        print(f"  {str(a):<26}{val * 100:>+8.1f}c  {dp:>7.0f}  {_bar(val, lo, hi)}")

    # Break down the winner: which weight*feature terms built its value?
    best_action, best_feats, best_val, _dp = rows[0]
    print(f"\nTop action:  {best_action}   (TD value {best_val * 100:+.1f}c)")
    print("what drove that value -- biggest weight*feature terms:")
    print(f"  {'contribution':>12}  {'= weight':>9} x {'feature':>7}   name")
    terms = [
        (names[i], model.w[i], best_feats[i], model.w[i] * best_feats[i])
        for i in range(len(best_feats))
    ]
    for name, wi, xi, contrib in sorted(terms, key=lambda t: abs(t[3]), reverse=True)[:10]:
        print(f"  {contrib * 100:>+11.1f}c  {wi:>+9.3f} x {xi:>7.3f}   {name}")

    # Why did #1 beat #2? The per-feature difference in contribution.
    if len(rows) > 1:
        runner = rows[1]
        r_action, r_feats, r_val, _ = runner
        print(f"\nWhy it beats runner-up ({r_action}, {r_val * 100:+.1f}c) "
              f"-- gap {(best_val - r_val) * 100:+.1f}c, by feature:")
        diffs = [
            (names[i], model.w[i] * (best_feats[i] - r_feats[i]))
            for i in range(len(best_feats))
        ]
        for name, d in sorted(diffs, key=lambda t: abs(t[1]), reverse=True)[:6]:
            if abs(d) < 1e-9:
                continue
            print(f"  {d * 100:>+8.1f}c   {name}")


if __name__ == "__main__":
    main()
