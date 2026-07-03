how to train ai to be good at game?
similar approach to TD-Gammon seems good, using TD and NN for function approx? but I'M RL newbi, is that good idea? other approach might be GO style deep search using MC search methods to estimate values or so, but not sure maybe game too stochastic for this?

at each point decision looks something like this:
have already 1, 
rolled [3,3,3] ,5,6 
what to do,
reasonable decisions are really only:
keep threes and possibly 5 and roll again, 
keep threes stop and 5 stop
keep only 5 roll again
typically one only has few options. I guess mostly fewer than 5.
When number is larger e.g. one rolls 6 times a 1, and has 6 different options firstly super rare, secondly obvious what one does.
so really we only care about the small differences in expected value for the 3 or so options.

how to think about this:

simple approach ignore inter turn dynamics, just maximise EV for this turn.
One could do a one time expected value calculation for the very start of a turn, and then it would seem to me all the numbers would be crunchable.
Each option gives you a direct score, and some probability of getting to fill your set of dice and start a new turn, that value we know, so it seems like a pretty simple small markov chain one might be able to solve using DP or so.

But inter turn dynamics make it more complex
1. if one is behind how much riskier should one play
2. we get money from special events, straße, totale, winning quickly etc. harder to estimate how important these are

so including all that we still only have a few options at each point but calculating exact EV seems impossible, need smart approximation or so.

Also even though each decision only has a few branches each move is so stochastic the actual game tree has a huge branching factor because of the dice rolls.

One thing to consider, how much knowledge to build in. E.g. in making decisions one could just input raw dice data etc. and have AI learn everything or:
we could look at what kinds of decisions even exist in detail etc. hmm, thinking about this I don't have many ideas, just giving AI raw data + some features seems good already.

I guess one feature could be purely the EV for this turn, maybe even train one AI to estimate that, and use as feature for second AI, might not be optimal though.
And reward function, in the end we only care about money but for intermediate rewards and to get starting using points seems very useful.
