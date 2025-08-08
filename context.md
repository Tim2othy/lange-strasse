
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


# What the AI will know

The state of the game has the markov property, nothing about the past has to be known to make a correct decision.

At each point in time the information available to the AI is:
1. The available dice and kept dice, and if kept 1s and 5s are grouped or not
2. This turns score [int], and this set of dice score [int]
3. Everyones total score, and if they have gotten a "strich" so far
4. The current turn [int]
5. Starting player and order of play


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

# What the AI will do

Essentially we want a function that takes the current game state as input and outputs the optimal move for the current player.
