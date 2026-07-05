"""Inspect the trained linear TD weights.

    python -m algorithms.weights      (run from the src/ directory)

The TD value is linear -- V(afterstate) = w . features -- so each weight is the
model's opinion about one feature, in isolation. The learning target is
``final_money_cents / 100`` (see td.py), so the value's natural unit is
"hundreds of cents". That makes weights directly readable in money: a feature
worth weight ``w`` shifts the predicted end-of-game money by ``100 * w`` cents
for a full 0->1 move of that (normalized) feature. For the yes/no flags a 0->1
move is literally the feature turning on, so ``100 * w`` is "cents that flag is
worth". For the scaled features (counts /6, scores /2000 or /10000, money
/1000) it's cents per one full normalized unit -- the divisor is named so you
can rescale to raw units.

Caveats on interpreting a *linear* model:
  * Weights are correlated. Several features move together (e.g. total_turn
    score and turn_ends), so the model can split credit between them in ways
    that make an individual weight look odd while the sum stays sensible.
  * The six ``avail #`` features are always 0 in an afterstate (there are no
    available dice before the next roll), so they never get a gradient and
    stay exactly 0.0 -- dead inputs, shown only for completeness.
"""

import pickle
from pathlib import Path

_WEIGHTS_PATH = Path(__file__).with_name("td_weights.pkl")


def feature_names(n_players: int = 3) -> list[str]:
    """Human labels for each weight, in the exact order td.py builds them:
    StateExtractor.to_vector + turn-over flag (action_features) + [bias, dp]."""
    names: list[str] = []
    for face in range(1, 7):
        names.append(f"avail #{face}s  (/6, always 0 in afterstate)")
    for face in range(1, 7):
        names.append(f"kept #{face}s  (/6)")
    names.append("kept 1s inside a triplet  (/6)")
    names.append("kept 5s inside a triplet  (/6)")
    names.append("current-set score  (/2000)")
    names.append("turn banked score  (/10000)")
    names.append("total turn score, banked if I stop  (/10000)")
    names.append("roll count  (/6)")
    names.append("dice left to roll  (/6)")
    names.append("CAN complete Lange Strasse  (flag)")
    names.append("CAN complete Talheim  (flag)")
    names.append("CAN keep any die  (flag)")
    seats = ["me", "next player", "last player"]
    for offset in range(n_players):
        who = seats[offset] if offset < len(seats) else f"player +{offset}"
        names.append(f"{who}: total score  (/10000)")
        names.append(f"{who}: has strich  (flag)")
        names.append(f"{who}: money  (/1000)")
    names.append("turn number  (/20)")
    names.append("is final round  (flag)")
    names.append("this action ENDS my turn  (flag)")  # action_features extra
    names.append("bias  (always 1.0)")  # td_features extras
    names.append("DP turn-value of the action  (/1000)")
    return names


def main() -> None:
    if not _WEIGHTS_PATH.exists():
        print(
            f"No weights at {_WEIGHTS_PATH.name} -- train first: python -m algorithms.td"
        )
        return

    with open(_WEIGHTS_PATH, "rb") as f:
        data = pickle.load(f)

    w = data["w"]
    names = feature_names()
    if len(names) != len(w):
        print(f"WARNING: {len(names)} labels but {len(w)} weights -- layout drifted.")

    print(
        f"td_weights.pkl  (version {data.get('version')}, dim {data.get('dim')}, "
        f"trained on {data.get('games', '?')} self-play games)"
    )
    print(
        "value target is final_money/100, so  cents = 100 * weight  "
        "for a full 0->1 move of the (normalized) feature.\n"
    )

    print("In feature order:")
    print(f"  {'#':>2}  {'weight':>9}  {'cents':>7}  feature")
    for i, (name, wi) in enumerate(zip(names, w)):
        print(f"  {i:>2}  {wi:>9.3f}  {wi * 100:>+6.0f}c  {name}")

    print("\nRanked by influence (|weight|), most opinionated first:")
    ranked = sorted(zip(names, w), key=lambda nw: abs(nw[1]), reverse=True)
    for name, wi in ranked:
        if abs(wi) < 1e-9:
            continue  # skip dead / never-updated inputs
        print(f"  {wi:>+9.3f}  ({wi * 100:>+6.0f}c) value  <-  {name}")


if __name__ == "__main__":
    main()
