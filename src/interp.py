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

import algorithms.td as td
from config import TD_INTERP


def feature_names() -> list[str]:
    # Derive the matching key list from the variant name, e.g. td_small -> TD_SMALL_KEYS.
    key_list_name = f"{TD_INTERP.upper()}_KEYS"
    keys = getattr(td, key_list_name)
    return [key.replace("_", " ") for key in keys + ["bias"]]


def main() -> None:
    if TD_INTERP not in td.VARIANTS:
        print(f"Unknown TD model {TD_INTERP!r}; choose from {list(td.VARIANTS)}")
        return
    weights_path = td.VARIANTS[TD_INTERP].path
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
