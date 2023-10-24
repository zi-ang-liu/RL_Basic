# Simple_RL

This repository contains implementations of basic RL algorithms.

The implemented algorithms and problems are based on the book [Reinforcement Learning: An Introduction (second edition)](http://incompleteideas.net/book/the-book-2nd.html) by Sutton and Barto.

「強化学習（第2版）」の日本語版は[こちら](https://www.morikita.co.jp/books/mid/082662).

## Implemented Algorithms
* Chapter 4: Dynamic Programming
  * Value Iteration (Jack's Car Rental Problem)
* Chapter 6: Temporal-Difference Learning
  * SARSA (Cliff Walking Problem)
  * Q-Learning (Cliff Walking Problem)

## Problems
### Jack's Car Rental Problem
The problem is described in Example 4.2 (page 81) of the book. 
The goal is to find the optimal policy for moving cars between two locations based on the expected number of cars at each location at the end of the day. 
The problem is formulated as a Markov Decision Process (MDP) and solved using value iteration.
Our results are consistent with the results in the book.

### Cliff Walking Problem
The problem is described in Example 6.6 (page 132) of the book.
The goal is to find the optimal policy for moving an agent from a starting position to a goal position as quickly as possible while avoiding falling off a cliff.
The problem is solved using SARSA and Q-Learning.

The mean results are closed to the results in the book.
SARSA consistently outperformed Q-Learning. 
In particular, SARSA has demonstrated the capability to generate a policy that is markedly safer compared to Q-Learning. 
This observation is the same with the conclusions presented in the book.

However, the book's results may have smoothed over more runs.
Our results are not as smooth as the results in the book.