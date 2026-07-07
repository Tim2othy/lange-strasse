## General considerations

how to train ai to be good at game?
similar approach to TD-Gammon seems good, using TD and NN for function approx?

simple approach ignore inter turn dynamics, just maximise EV for this turn.
One could do a one time expected value calculation for the very start of a turn, and then it would seem to me all the numbers would be crunchable.
Each option gives you a direct score, and some probability of getting to fill your set of dice and start a new turn, that value we know, so it seems like a pretty simple small markov chain one might be able to solve using DP or so.

But inter turn dynamics make it more complex
1. if one is behind how much riskier should one play
2. we get money from special events, straße, totale, winning quickly etc. harder to estimate how important these are

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
12. Money already exchanged this game is sunk: it can never change which action is best (only future money matters). The money features are needed to predict value, but the model should learn they don't rank actions.

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
      "seat_offset",
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
4. **Swap linear → small MLP** once the pipeline is proven.
5. *(optional)* use the value net as the evaluator for shallow expectimax/MCTS lookahead.

**Newbie warning I'd underline:** do **not** start at deep RL. That's where people drown in instability and un-debuggable training runs. Get a working self-play TD loop with a *linear* model and good features first; deep is a drop-in upgrade afterward.

## The "something cool" angle

The **arena** is the fun engine. Every new AI (`random → simple → dp → linear-TD → mlp`) enters the ring; you watch win-rate and money climb. It's motivating, it doubles as your regression test, and "my new AI beats my last one 62% of games" is a great progress signal.

**Suggested next concrete step:** build milestone 1 (`turn_value.py`, the DP solver) and wire it in as algo `"dp"`. Small, exact, and you'll immediately see it dominate the arena.
