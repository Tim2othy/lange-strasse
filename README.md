# lange-strasse

Use
`cd C:\Users\timtj\GitHub\lange-strasse\src
python -m algorithms.td td_small 2000` to train model. Change Model name and number.


This project uses a linear TD value function over *afterstates*.

For every legal action the AI can take, it:

1. Applies that action to the current game state and builds the resulting afterstate.
2. Converts that afterstate into a fixed feature vector.
3. Computes a dot product `w · x` between the learned weights and the feature vector.
4. Picks the action with the highest value.

at runtime it is just matrix-multiply style scoring, followed by an argmax over legal actions. The important detail is that the model scores the *state reached by the action*, not the raw action itself.

The TD target is `final_money_cents / 100`, so the model predicts value in "hundreds of cents". To read a weight in ordinary money, multiply it by `100` cents. Because the input features are normalized too, each weight means:

`weight = how much predicted final money changes when that normalized feature increases by 1.0`

For a feature stored as `money / 1000`, a weight of `0.65` means:

* +1000 cents in money changes predicted final value by about 65 cents.
* +50 cents in money changes predicted final value by about 3.25 cents.

That is why the model does **not** multiply `50` by the weight directly. It first normalizes the feature to `0.05`, then multiplies `0.05 * weight`.

## How the money example works

Suppose your current money is `-100¢`, and one action completes a Lange Strasse that pays `+70¢`, so the afterstate money becomes `-30¢`.

The feature vector does not contain `+70` as a separate signal. It contains the afterstate balance itself:

* before: `money / 1000 = -0.10`
* after: `money / 1000 = -0.03`

The model learns that the afterstate with `-0.03` is better than the one with `-0.10` because it is a better predictor of final money. In other words, the reward is not encoded as a delta feature; it is folded into the resulting state.

That is the whole point of afterstates: a Strasse, Talheim, stop, or continue decision all become different resulting positions, and the TD model learns which resulting positions tend to lead to better final outcomes.

## Feature layout

The feature vector comes from `StateExtractor.action_features(state, action)` in `src/game_state.py`. It is:

`to_vector(afterstate) + [turn_ended_flag]`

and then `td.py` appends two extra algorithm-specific features:

`[bias, dp_turn_value]`

That gives the TD model one long vector with the exact layout below.

### Dice features

These are normalized counts, so each is in the range `0.0` to `1.0`.

| # | Feature | Meaning |
|---|---|---|
| 6 | kept #1s (/6) | How many 1s are currently kept in this afterstate. |
| 7 | kept #2s (/6) | How many 2s are kept. |
| 8 | kept #3s (/6) | How many 3s are kept. |
| 9 | kept #4s (/6) | How many 4s are kept. |
| 10 | kept #5s (/6) | How many 5s are kept. |
| 11 | kept #6s (/6) | How many 6s are kept. |
| 12 | kept 1s inside a triplet (/6) | 1s that are part of a grouped triplet-or-larger set, not individual 1s. |
| 13 | kept 5s inside a triplet (/6) | 5s that are part of a grouped triplet-or-larger set. |

These grouped 1s and 5s matter because a 1 or 5 can exist either as an individual scoring die or as part of a larger triplet, and those are strategically different.

### Scoring and turn-flow features

| # | Feature | Meaning |
|---|---|---|
| 14 | current-set score (/2000) | Score of the dice set currently kept on this turn, before any banking into total score. |
| 15 | turn banked score (/10000) | Score already accumulated this turn from previous sets. |
| 16 | total turn score, banked if I stop (/10000) | What the current player would bank right now if they stopped. This is the sum of banked turn score plus current-set score. |
| 17 | roll count (/6) | How many rolls have happened in the current set. |
| 18 | dice left to roll (/6) | How many dice are still unkept in this afterstate. |

### Situation flags

| # | Feature | Meaning |
|---|---|---|
| 19 | CAN complete Lange Strasse (flag) | `1.0` if the remaining dice plus kept dice could complete a Lange Strasse. |
| 20 | CAN complete Talheim (flag) | `1.0` if the remaining dice plus kept dice could complete a Talheim. |
| 21 | CAN keep any die (flag) | `1.0` if there is at least one legal keep available. |

These are pure yes/no features, so their weights are easy to read: a positive weight means the model likes that situation, a negative weight means it dislikes it.

### Player features

Players are ordered from the acting player outward, so the current player is always first. That makes the model seat-agnostic.

| # | Feature | Meaning |
|---|---|---|
| 22 | me: total score (/10000) | The acting player's total score in the afterstate. |
| 23 | me: has strich (flag) | Whether the acting player has already been marked with a strich. |
| 24 | me: money (/1000) | The acting player's current money balance in the afterstate, normalized by 1000. |
| 25 | next player: total score (/10000) | The next player's total score. |
| 26 | next player: has strich (flag) | Whether the next player has a strich. |
| 27 | next player: money (/1000) | The next player's money balance. |
| 28 | last player: total score (/10000) | The third player's total score. |
| 29 | last player: has strich (flag) | Whether the third player has a strich. |
| 30 | last player: money (/1000) | The third player's money balance. |

The money feature is not "money gained this action". It is the current balance after the action has been applied.

That means the model can learn things like:

* having `+50¢` is better than `-100¢`
* reducing your debt from `-100¢` to `-30¢` is already an improvement
* a Strasse that pays out now is valuable because it changes the afterstate balance

### Game-context features

| # | Feature | Meaning |
|---|---|---|
| 31 | turn number (/20) | Which turn of the game this afterstate belongs to. |
| 32 | is final round (flag) | Whether the game is in the final round. |
| 33 | this action ENDS my turn (flag) | `1.0` if the chosen action ends the turn immediately. This is appended by `action_features`. |
| 34 | bias (always 1.0) | Constant offset term for the linear model. |
| 35 | DP turn-value of the action (/1000) | The separate DP solver's estimate of this action's value, normalized before being appended as an extra feature. |

## Interpreting the top weights

If a weight is positive, increasing that feature pushes the predicted final money up. If it is negative, increasing that feature pushes the prediction down. Because the model is linear, each weight is the model's independent opinion about that one feature, holding all others fixed.

So for `me: money (/1000)` specifically:

* positive weight means the model likes being richer right now
* negative weight would mean the model thinks higher current money correlates with worse final results, which would be unusual

The same logic applies to the total-score features and the binary flags. The important caveat is that many features are correlated, so the model can split credit across them in ways that make one weight look surprising by itself even when the combined prediction is sensible.

## Why the afterstate representation helps

If the model tried to score raw actions directly, it would be harder to reuse the same value function across different dice positions and turns. By converting each action into the state it leaves behind, the model can learn a single value function over positions:

* stop now and bank the turn
* keep these dice and continue
* finish a special combination
* take hot dice and roll again

All of those are just different afterstates, and the TD learner can compare them with the same dot product.
