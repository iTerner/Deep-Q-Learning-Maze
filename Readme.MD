# Maze Solving Using Deep Q-Learning

## Table of contents

- [General Info](#general-Info)
- [Description](#Description)
- [Reward Policy](#Reward-Policy)
- [Repository Description](#repository-description)
- [Requirement](#Requirement)

## General Info

A simple implementation of DQN that uses PyTorch and a fully connected neural network to estimate the q-values of each state-action pair.

## Description

The environment is a maze that is randomly generated using a deep-first search algorithm to estimate the Q-values. Four moves are possible for the agent (up, down, left and right), whose objective is to reach a predetermined cell. The agent implements either an epsilon-greedy policy or a softmax behaviour policy with temperature equal to epsilon. After each episode, the starting position is sampled in such a way that at the beginning of the training the agent explores the area surrounding the goal, and as the training goes on it will explore further and further areas of the maze.

A convolutional neural network is also implemented for completeness.

## Reward Policy

1. For every move the agent do, he get a reward of -0.05
2. If the goal has been reached, the agent get a reward of 1.
3. If the cell has been reached before, the agent get a reward of -0.2
4. If the move the agent selected to do goes out of the maze or hit the wall, he get a reward of -1

## Repository Description

| Filename              | description                                                                                                |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `maze_generator`      | Folder which consists of the python file for generating a maze and a sample maze                           |
| `res`                 | Folder which consists of the results figures of the experiment (created automatically)                     |
| `RL_models`           | Folder which consists of the PyTorch model used for this experiment after training (created automatically) |
| `sol`                 | Folder which consists of the policy map of the agent during the experiment (created automatically)         |
| `agent.py`            | Python file of the Agent class                                                                             |
| `environment.py`      | Python file of the MazeEnvironment class                                                                   |
| `experience_reply.py` | Python file of the ExperienceReply class                                                                   |
| `models.py`           | Python file of the implementation of the models                                                            |
| `main.py`             | The main file                                                                                              |
| `main.ipynb`          | The solver as a python notebook                                                                            |
| `requirement.txt`     | File containing all the packages we used in this project                                                   |

## Requirement

To run this project, you need to install several packages. For convenience, we created a `requirement.txt` file consists of all the packages used in this projcet.

In order to install all the packages in the `requirement.txt` file, simply use to command `pip install -r requirements.txt`.
