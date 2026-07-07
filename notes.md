## General considerations

how to train ai to be good at game?
similar approach to TD-Gammon seems good, using TD and NN for function approx?

simple approach ignore inter turn dynamics, just maximise EV for this turn.

right now using linear model that reads in some features and and dynamic programming solver that only optimizes for individual turn score not long term money.

## What the Model should Learn

1. If a player is close to winning or has already crossed 10K play more riskily. (for humans it's simple to understand that If someone else has won and I get one more turn I play forever until I've passed them (modulo some risk from getting totale (the only way playing can actually lose you money), normally of course once one has 3000 points one would always stop, but if it's the last turn, you just keep playing for that tiny chance to overtake the leader.) 
2. ALso take into account seating position. Will you have one more turn after other player crosses 10K or not.
3. Lange strasse
4. Super Strasse
5.  totale
6.  If you win by the 10th round you get 50ct from each player, so keep track of this, possibly play more riskily if not 10th round yet
7. If the game ends and you have no "strich" you get 50ct from each player (even if you didn't win), so keep track and play more carefully if you have no strich, especially if game almost over
8. If someone wins and you don't have 5000 points yet you must give the winner 50ct. so if below 5000 points, play differently, and if you close to winning and someone below 5000 points also take into consideration
9. Second place pays the winner 50ct, third place pays 70ct: even when winning is out of reach, fighting for 2nd place is worth 20ct. So in the final round the target isn't just "overtake the leader" — overtaking the other non-winner also pays.
10. Talheim auto-ends the turn: keeping pairs that complete it is an implicit "stop" decision. 500/1000 points but you lose the option to keep rolling — bad exactly when you must chase (final round, far behind).
11. Super Strasse timing: a strasse completed on the 3rd+ roll pays double (100ct vs 50ct per player). With a 1 and/or 5 kept early, deliberately *not* completing the strasse immediately and going for the 3rd-roll completion can be worth it.
12. Money already exchanged this game is sunk: it can never change which action is best (only future money matters). Solved structurally in the NN: its target is *future* money (reward = money change between a player's consecutive decisions), so the balances — the one unbounded state quantity — aren't inputs at all. The linear td models still predict total final money and keep the money features.

## Features

### Direct Game state representation

The GameState class and it's minimal feature representation:

1. Dice information
   ```
    available_dice: List[int]
    ```
    values available to keep this roll, e.g. [1, 4], drops out because we evaluate the state after an action is taken at which point there are no available dice
    ```
    kept_groups: List[List[int]]  # Groups of kept dice
   ```
   becomes
   `"group1", "group2", "group3", "group4", "group5", "group6"`
   incodes groups using exponential function, if three are kept gives 1, four gives 2, five gives 4 and six gives 8, else 0

   `"loose1", "loose5"` linear encoding of number of 1s or 5s keept


2. Turn history
   ```
      turn_accumulated_score: int     -> "turn_accumulated": int, 
      roll_count: int                 -> "roll_count": int,
   ```

4. Player information
   ```
      players: List[PlayerState]
   ```
   becomes
   ```
      "score_me": int, "strich_me": bool, "money_me": int,
      "score_p2": int, "strich_p2": bool, "money_p2": int,
      "score_p3": int, "strich_p3": bool, "money_p3": int,
   ```
4. Seating position
   ```
      current_player_idx: int
      starting_player_idx: int
   ```
   becomes

   ``` 
      "seat0", "seat1", "seat2",
   ```
   starting player and own player id is formatted into one, loses no information

5. Game context
   ```
    turn_number: int            -> "turn_number": int,
   ```

6. Extra is 
   ```
    "ends_turn"
   ```
   this is extra, it's not in game state, information depending on action not just game state, simple bool


### Hand crafted features

```
is_final_round: bool        -> "is_final_round": bool,
```
## Steps

1. heuristic + env + `simulate()` arena. ✅
2. **DP single-turn solver** → plug in as algo `"dp"`. Teaches Bellman / value iteration. Strong baseline immediately. ✅
   - ALso passed creates `dp_value` feature for linear TD model
3. **Linear TD value function over between-turn states, via self-play.** Reuses your `to_vector`. Linear = stable, fast, and the weights are *readable* (you can see what it learned). ✅
4. Implement small MLP ✅
5. Improve Feature usefulness, specific features for special ways to get money ✅
6. Normalize Every feature ✅
7. *(optional)* use the value net as the evaluator for shallow expectimax/MCTS lookahead.
8. give the net multiple heads
   - instead of one output predicting "expected money," give the net multiple heads predicting the probabilities of the known payout events — P(I place 1st/2nd/3rd), P(win by round 10), P(I finish with no strich), P(I finish under 5000), plus the same for opponents — and combine them into expected money analytically using the payout table. Each head is a much easier, denser learning target than the raw money sum, and the combination is exact game knowledge, not a heuristic. This directly attacks your worry: the net doesn't need to discover that no-strich is worth something; it only needs to estimate P(no strich), which every game gives a label for.
9. Redefine the objective. Reward = money deltas as they occur (strasse, totale) plus terminal settlement. Value = expected future money, predicted as the event-probability heads combined analytically via the payout table.
10. Repurpose the DP solver as within-turn search. This is the key architectural move: keep the DP recursion over rolls/keeps inside a turn, but replace its terminal objective (turn points) with V_net(game state after the turn ends). The solver handles tactical dice combinatorics exactly; the net supplies strategic context (scores, strich, round, money).



# Summary

- Keep DP for within-turn optimization; use learning only for between-turn strategy.
	- The solver should decide how good a turn line is in isolation.
	- The learned part should decide when that line is worth taking in the full game.
- Train with TD on self-play, with final money as the real target.
	- TD is useful because it can learn from intermediate states instead of waiting for full game outcomes.
	- That makes it a better fit than pure Monte Carlo for long games with delayed payoff.
- Start with a linear TD baseline, then move to a small nonlinear model.
	- Linear is easier to debug and tells us which features matter.
	- Nonlinear is the likely end state because the important interactions are not linear.
- Feed the model raw state plus a few exact derived features that matter for money and endgame risk.
	- Keep the encoding mostly lossless.
	- Add only exact derived signals, not vague handcrafted heuristics.
- Keep `dp_value` as an input at first; later ablate it if the learned model becomes strong enough.
	- Early on it should act like a strong prior.
	- Later it becomes a test of whether the network can surpass the solver instead of just copying it.
- Use the current state features to capture endgame pressure, turn bonus timing, strich/no-strich effects, and opponent score gaps.
	- These are the cases where a pure turn-value solver is blind.
	- The model needs to learn when playing riskier is correct even if the current turn value drops.
- Evaluate by playing new versions against frozen baselines and comparing money per game.
	- Compare against DP, linear, and previous learned versions.
	- Use money/game, not just win rate, because bonuses and penalties matter.


## Feature policy

- Keep features that are exact and cheap.
	- Examples: turn number, final-round flag, score gap, strich counts, seat offset.
	- These are not “bias”; they are direct information already present in the game state.
- Prefer lossless or near-lossless inputs over aggressive abstraction.
	- The learner can ignore irrelevant information.
	- It cannot recover information that was thrown away.
- Use hand-crafted nonlinear features only when they encode exact game facts.
	- Example: a feature that directly expresses danger of losing a bonus or missing a round cutoff is fine.
	- Avoid features that are just rephrased opinions about what the move should be.
- Keep `ends_turn` separate because it depends on the action.
	- That makes action scoring easier.
	- It also helps the model compare stop vs continue decisions.

## Model choice

- Start with a small MLP once the linear baseline is proven.
	- Two hidden layers is probably enough to begin with.
	- The goal is to capture interactions like score gap times turn number times strich state.
- Do not jump straight to a large network.
	- Large models are harder to debug and harder to train well on small self-play data.
	- The first useful result should be easy to understand.
- If the small MLP fails, inspect the feature set before scaling the model.
	- The problem may be missing exact signals, not insufficient depth.
	- Only then increase capacity.

## Training loop

- Generate self-play games with the current policy.
	- Keep some exploration so the model sees more than greedy lines.
	- Freeze older versions occasionally so evaluation is stable.
- Update from game trajectories, not isolated positions.
	- That lets TD connect tactical outcomes to game-long value.
	- It also makes bonuses and endgame transitions visible to training.
- Measure progress in a simple ladder.
	- random -> DP -> linear TD -> small MLP.
	- Each step should beat the previous one before moving on.

## Main idea to keep in mind

- Exact search where the rules are cheap.
- Learning where the long-term value is hard.
- Raw state plus exact derived signals.
- Small model first, nonlinear later.

# Notes

- Your final-round-chase example is also handled by the architecture below rather than by learning alone: once the leaf values of the turn-level solver are "expected money" instead of "turn points," stopping at 9800 while an opponent has 10,000+ evaluates to a near-certain loss, and rolling on dominates automatically.
- Small MLP: 2 hidden layers of 64–128 units, ReLU, the multi-head output above. This problem has maybe 30–50 input dims and smooth structure; that's plenty. Scale up only if self-play performance plateaus.
- This is TD-Leaf, and it fixes your motivating examples by construction.
Self-play training loop. Play games with the current net (softmax or ε-greedy over action values for exploration), TD(λ) updates. Millions of games are feasible for a dice game; start with ~100k and watch the curves.
Evaluation harness. Fixed-seed tournaments of new net vs. frozen pure-DP and linear baselines, measuring average money/game. Money per game is high-variance and discrete — expect to need ~10k+ games per matchup for significance, so make evaluation cheap and automated.
Iterate: ablate dp_value and the derived features once training works; scale the net if it plateaus; keep a league of frozen past versions so self-play opponents don't drift in lockstep.
