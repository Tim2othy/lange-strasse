# Goal

Train AI to play Lange Strasse.

Be able to use it to know what the optimal move is in any situation and improve my gameplay.

# Rules of the game

## General Idea

Three players take turns rolling 6 dice, and acumilating points. The game ends once someone reaches 10.000 points (the current round is finished, if there are 3 players and player 1 start the game player 3 will have the very last dice roll.)

Players get "money" from winning a game and certain special dice rolls and from winning in certain ways.

The overall goal is to accumulate as much money as possible, over the course of many games.
Different games cannot affect each other so one can view every game as having the goal of getting as much money as possible during this game.

## A Turn

1. Choose which of the 6 rolled dice you want to keep which you want to roll again, or choose which dice to keep and end the turn.
2. If none of the rerolled dice can be kept your turn ends and your score is 0.
3. You can only end your turn if your score for the turn is at least 300 (this includes the points from the dice you are keeping right now).
4. If you end up keeping all 6 dice the turn starts over exactly like a new turn, except that you keep the score you have acumulated so far.
5. But you need to reach 300 again before you can end your turn.

## Dice scoring

- Every group of 3 is worth it's number * 100. Except 1s they are worth 1000. For each additional number in the group the score is doubled.
- 1s and 5s can also be kept in non groups. Then they are worth 100 and 50 respectively. And they don't affect the group scoring.
- If all 6 dice are used up groups reset.
- There are special dice rolls that are explained in the next section.


## Special Dice Rolls

1. All numbers from 1 to 6 are a lange strasse and worth 1250
   1. The dice don't have to be kept all at once, a 1 or 5 or both can be kept and then on the next roll a lange strasse still counts.
   2. If you get one on the third roll i.e. first a 1 or 5 was kept, on the second roll the other one and on the third you get a lange strasse it is a super strasse
2. Getting a strich i.e. no dice can be kept and you get 0 points, in the very first round is a totale
3. Getting three pairs (e.g. 2, 2, 5, 5, 3, 3) is talheim and worth 500 points.
   1. if the the pairs are consecutive (e.g. 2, 2, 3, 3, 4, 4) it is worth 1000 points.
   2. Talheim doesn't have to be confirmed your turn auto ends when it is kept
   3. It also doesn't have to be rolled at once. you can first get a 1. and on the next roll (1, 3, 3, 4,4)

## Ways to make money

1. The winner gets 50ct from the second place 70ct from third place.
2. Rolling a Lange strasse gets you 50ct from each player
3. Rolling a Totale you loose 50ct to each player
4. Rolling a super Strasse you get 100ct from each player
5. If you win by the 10th round you get 50ct from each player
6. If the game ends and you have no "strich" you get 50ct from each player (even if you didn't win)
7. If someone wins and you don't have 5000 points yet you must give the winner 50ct.

# Game Code structure


## File Overview

### 1. scoring.py - Game Rules & Validation
**Classes:**
- `ScoreValidator`: Validates which dice combinations can be kept
   - `is_valid_keep()`: Checks if dice selection follows rules (1s/5s individual, others need 3+)

- `ScoreCalculator`: Calculates points from kept dice
   - `calculate_score_from_groups()`: Converts kept dice into points using game scoring rules


### 2. dice.py - Dice Management & Game State
**Classes:**
- `DiceSet`: Core game mechanics - rolling, keeping, scoring
   - `roll()`: Rolls non-kept dice (with debug override)
   - `keep_dice_by_value()`: Main action handler - validates and keeps dice
   - `check_lange_strasse()`, `check_talheim()`, `check_totale()`: Special combination detection
   - `display()`: Shows current game state to player

### 3. game.py - Player & Game Management
**Classes:**
- `Player`: Stores name, score, money, strich status
- `Game`: Manages 3-player game flow, turn order, win conditions
   - `end_turn()`: Processes turn results, checks win conditions
   - `handle_lange_strasse()`, `handle_totale()`: Money distribution
   - `check_winner()`: Final scoring and money distribution



### 4. main.py - User Interface & Game Loop
**Functions:**
- `main()`: Command parsing, human/AI interaction, debug commands


## Game Flow: What Happens When You Type `keep 5`

### 1. **Command Input** (main.py lines 103-137)
```
Player types: "keep 5"
```
- Command parsed in main game loop
- `parts = command.split()` → `["keep", "5"]`
- `dice_values = [int(x) for x in parts[1:]]` → `[5]`
- Validation: check if values are 1-6

### 2. **Action Execution** (main.py line 138)
```python
success, message = game.dice_set.keep_dice_by_value([5], stop_after=False)
```

### 3. **Dice Validation** (dice.py lines 125-170)
- Check if enough 5s are available on unrolled dice
- Get current kept dice for validation context
- Call `ScoreValidator.is_valid_keep([5], already_kept_dice)`

### 4. **Score Validation** (scoring.py lines 20-50)
- Since `value == 5`, individual 5s are always valid
- Return `(True, "")` - validation passed

### 5. **Keep Dice Process** (dice.py lines 172-180)
- Find first available die with value 5
- Mark that die position as `kept_dice[i] = True`
- Call `_merge_kept_dice([5])` to add to kept groups

### 6. **Group Management** (dice.py lines 285-310)
- Since it's a single 5, create individual group: `kept_groups.append([5])`

### 7. **Special Combination Checks** (dice.py lines 182-194)
- Check for Talheim (3 pairs) - not applicable
- Check for Lange Strasse (1-2-3-4-5-6) - not applicable

### 8. **Continue Turn Logic** (dice.py lines 205-220)
- Check if all dice are kept - if not, continue
- Auto-roll remaining dice with `self.roll()`
- Check if any new moves are possible

### 9. **Display Update** (main.py lines 180-185)
- Print game state after move
- Call `game.dice_set.display()` to show:
  - Kept dice: `5`
  - Available dice: (newly rolled values)
  - Current set score: `50 points`
  - Total turn score: `50 points`

### 10. **Score Calculation** (dice.py lines 270-280)
- `get_current_score()` calls `ScoreCalculator.calculate_score_from_groups()`
- For group `[5]`: individual 5 = 50 points

### 11. **Wait for Next Command** (main.py line 97)
- Return to input prompt for next player action

---

## Key Data Flow

**State Tracking:**
- `dice = [1,2,3,4,5,6]` (physical dice values)
- `kept_dice = [False,False,False,False,True,False]` (which positions are kept)
- `kept_groups = [[5]]` (logical grouping for scoring)

**Score Calculation Chain:**
1. `kept_groups` → `ScoreCalculator.calculate_score_from_groups()`
2. Individual 5s: 50 points each
3. Groups of 3+: base score × 2^(extra dice)
4. Special combinations override normal scoring

**Turn Flow:**
1. Roll dice → 2. Display state → 3. Player input → 4. Validate action → 5. Execute action → 6. Check specials → 7. Update display → 8. Repeat or end turn


# AI situation

## What the AI will know

The state of the game has the markov property, nothing about the past has to be known to make a correct decision.

At each point in time the information available to the AI is:
1. The available dice and kept dice, and if kept 1s and 5s are grouped or not
2. This turns score [int], and this set of dice score [int]
3. Everyones total score, and if they have gotten a "strich" so far
4. The current turn [int]
5. If he is the first second or third player


## For example:

### 1.

Current dice:
Kept groups: [[5, 5, 5]]
Available dice: [4, 1, 2]

### 2.

Turn score: 1500 points
Score for this set of dice: 300

### 3.

Current scores:
Player 1: 4550, strich: true
Player 2: 1300, strich: true
Player 3: 6400, strich: false

### 4.


Turn-number: 7


### 5.

Starting player: Player 1
Order of play: Player 1, you, Player 3

## What the AI will be able to do

Essentially we want a function that takes the current game state as input and outputs the optimal move for the current player.
