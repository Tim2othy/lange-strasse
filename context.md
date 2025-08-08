
# Goal

Train AI to play Lange Strasse.

Be able to use it to know what the optimal move is in any situation and improve my gameplay.

# Rules of the game

## General Idea

Three players take turns rolling 6 dice, and acumilating points. The game ends once someone reaches 10.000 points (the current round is finished, if there are 3 players and player 1 start the game player 3 will have the very last dice roll.)

Players get "money" from winning a game and certain special dice rolls and from winning in certain ways.

The overall goal is to accumulate as much money as possible, over the course of many games.
Different games cannot affect each other so one can view every game as having the goal of getting as much money as possible during this game.

## Dice scoring

TODO

## Special Dice Rolls


TODO

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
2. This turns score [int]
3. Everyones total score, and if they have gotten a "strich" so far
4. The current turn [int]
5. Starting player and order of play


## For example:

### 1.

Current dice:
Die 1: 4
Die 2: 5 [KEPT]
Die 3: 5 [KEPT]
Die 4: 5 [KEPT]
Die 5: 1
Die 6: 2
Kept groups: [[5, 5, 5]]

Available dice to keep: [1, 5, 6]



### 2.

Turn score: 500 points

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
