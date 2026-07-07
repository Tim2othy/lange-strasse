"""Inspect the trained linear TD weights.

    python interp.py      (run from the src/ directory)

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
"""

import pickle
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.td import VARIANTS
from config import TD_INTERP


def _pretty_name(key: str) -> str:
    return {
        "kept1": "kept #1s  (/6)",
        "kept2": "kept #2s  (/6)",
        "kept3": "kept #3s  (/6)",
        "kept4": "kept #4s  (/6)",
        "kept5": "kept #5s  (/6)",
        "kept6": "kept #6s  (/6)",
        "grouped1": "kept 1s inside a triplet  (/6)",
        "grouped5": "kept 5s inside a triplet  (/6)",
        "current_set_score": "current-set score  (/2000)",
        "turn_accumulated": "turn banked score  (/10000)",
        "total_turn_score": "total turn score, banked if I stop  (/10000)",
        "roll_count": "roll count  (/6)",
        "dice_left": "dice left to roll  (/6)",
        "flag_lange_strasse": "CAN complete Lange Strasse  (flag)",
        "flag_talheim": "CAN complete Talheim  (flag)",
        "flag_keep_any": "CAN keep any die  (flag)",
        "score_me": "me: total score  (/10000)",
        "strich_me": "me: has strich  (flag)",
        "money_me": "me: money  (/1000)",
        "score_p2": "next player: total score  (/10000)",
        "strich_p2": "next player: has strich  (flag)",
        "money_p2": "next player: money  (/1000)",
        "score_p3": "last player: total score  (/10000)",
        "strich_p3": "last player: has strich  (flag)",
        "money_p3": "last player: money  (/1000)",
        "turn_number": "turn number  (/20)",
        "is_final_round": "is final round  (flag)",
        "ends_turn": "this action ENDS my turn  (flag)",
        "bias": "bias  (always 1.0)",
        "dp_value": "DP turn-value of the action  (/1000)",
        "group1": "kept triplet size for #1s",
        "group2": "kept triplet size for #2s",
        "group3": "kept triplet size for #3s",
        "group4": "kept triplet size for #4s",
        "group5": "kept triplet size for #5s",
        "group6": "kept triplet size for #6s",
        "loose1": "loose #1s  (/6)",
        "loose5": "loose #5s  (/6)",
        # use one hot here
    }.get(key, key)


def feature_names() -> list[str]:
    # The vector an encoder builds is exactly the model's keys plus a trailing bias
    # (see td._encode), so its labels follow the same order.
    return [_pretty_name(key) for key in VARIANTS[TD_INTERP].keys + ["bias"]]


def main() -> None:
    if TD_INTERP not in VARIANTS:
        print(f"Unknown TD model {TD_INTERP!r}; choose from {list(VARIANTS)}")
        return
    weights_path = VARIANTS[TD_INTERP].path
    if not weights_path.exists():
        print(f"No weights at {weights_path.name}, train: python -m algorithms.td {TD_INTERP}")
        return

    with open(weights_path, "rb") as f:
        data = pickle.load(f)

    w = data["w"]
    names = feature_names()
    if len(names) != len(w):
        print(f"WARNING: {len(names)} labels but {len(w)} weights -- layout drifted.")

    print(
        f"{weights_path.name}, dim {data.get('dim')}, "
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
