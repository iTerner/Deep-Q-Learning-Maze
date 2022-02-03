from os import stat
import os
import numpy as np
import scipy.special as sp

from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import copy
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
import collections

from models import *
from experience_replay import ExperienceReplay
from agent import Agent
from environment import MazeEnvironment

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

Transition = collections.namedtuple("Experience", field_names=[
                                    "state", "action", "next_state", "reward", "is_game_on"])

SAVE_PATH = "res"
MODEL_PATH = "RL_models"
SOL_PATH = "sol"
p = [SAVE_PATH, MODEL_PATH, SOL_PATH]
for path in p:
    if not os.path.exists(path):
        os.mkdir(path)


def Q_loss(batch, net, device, gamma=0.99):
    """
    The function compute the Q loss of the given batch
    """
    # unpack the batch
    states, actions, next_states, rewards, _ = batch
    l_batch = len(states)
    state_action_values = net(states.view(l_batch, -1))
    state_action_values = state_action_values.gather(1, actions.unsqueeze(-1))
    state_action_values = state_action_values.squeeze(-1)

    next_state_values = net(next_states.view(l_batch, -1))
    next_state_values = next_state_values.max(1)[0]
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * gamma + rewards
    return nn.MSELoss()(state_action_values, expected_state_action_values)


# Load the maze and define the environment
maze = np.load("maze_generator/maze.npy")

initial_position = [0, 0]
goal = [len(maze)-1, len(maze)-1]

maze_env = MazeEnvironment(maze, initial_position, goal)
# print and save the maze
maze_env.draw(os.path.join(SAVE_PATH, "maze_20.pdf"))

# define the agent and the buffer for the experience reply object
buffer_capacity = 10_000
buffer_start_size = 1_000
memory_buffer = ExperienceReplay(buffer_capacity)

agent = Agent(maze_env, memory_buffer, device, use_softmax=True)

# define the NN model
learning_rate = 1e-4
batch_size = 24
gamma = 0.9
net = fc_nn(maze.size, [maze.size] * 2, 4)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# define the epsilon profile and plot the resetting probability
num_epochs = 10_000
cutoff = 3_000
epsilon = np.exp(-np.arange(num_epochs) / (cutoff))
epsilon[epsilon > epsilon[100*int(num_epochs/cutoff)]
        ] = epsilon[100*int(num_epochs/cutoff)]

plt.title("Epsilon Value per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Epsilon")
plt.plot(epsilon, ls="--")
plt.grid()
plt.savefig(os.path.join(SAVE_PATH, "epsilon_profile.pdf"))
plt.show()

mp, mpm = [], []
reg = 200  # the regularization
for e in epsilon:
    a = agent.env.reset_policy(e)
    mp.append(np.min(a))
    mpm.append(np.max(a))

plt.title("Epsilon Profile and Probability Difference per Epoch")
plt.xlabel("Epoch")
plt.ylabel(r"max $p^r$ - min $p^r$")
plt.plot(epsilon / 1.3, ls="--", alpha=0.5,
         label="Epsilon Profile (arbitrary unit)")
plt.plot(np.array(mpm) - np.array(mp), label="Probability Difference")
plt.grid()
plt.legend()
plt.savefig(os.path.join(SAVE_PATH, "reset_policy.pdf"))
plt.show()

# Training the model
loss_log = []
best_loss = 1e-5
running_loss = 0
estop = -1
for epoch in range(num_epochs):
    loss = 0
    counter = 0  # number of moves
    eps = epsilon[epoch]

    # set the is_game_on to True
    agent.is_game_on = True
    _ = agent.env.reset(eps)

    while agent.is_game_on:
        agent.make_move(net, eps, device)
        counter += 1

        if len(agent.buffer) < buffer_start_size:
            continue

        optimizer.zero_grad()
        batch = agent.buffer.sample(batch_size, device)
        loss_t = Q_loss(batch, net, device, gamma=gamma)
        loss_t.backward()
        optimizer.step()

        loss += loss_t.item()

    if (agent.env.current_pos == agent.env.goal).all():
        result = "won"
    else:
        result = "lost"

    if epoch % 1000 == 0:
        s_p = os.path.join(SOL_PATH, f"sol_epoch_{epoch}.pdf")
        agent.plot_policy_map(net, s_p, [0.35, -0.3])

    loss_log.append(loss)

    if epoch > 2000:
        running_loss = np.mean(loss_log[-50:])
        if running_loss < best_loss:
            print("saving model...")
            best_loss = running_loss
            # save the model
            torch.save(net.state_dict(), os.path.join(
                MODEL_PATH, "best_model.torch"))
            estop = epoch

    print(f"Epoch {epoch + 1}, number of moves {counter}")
    print(f"Game result: {result}")

    print("\t Average loss: " + f"{loss:.5f}")
    if (epoch > 2000):
        print("\t Best average loss of the last 50 epochs: " +
              f"{best_loss:.5f}" + ", achieved at epoch", estop)
    clear_output(wait=True)

# save the model after the training is done
torch.save(net.state_dict(), os.path.join(MODEL_PATH, "net.torch"))

# plot the results of the training
plt.plot(epsilon * 90, alpha=0.6, ls="--",
         label="Epsilon profile (arbitrary unit)")
plt.plot((np.array(mpm) - np.array(mp)) * 120, alpha=0.6, ls="--",
         label="Probability difference (arbitrary unit)")
plt.plot(loss_log, label="Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.grid()
plt.savefig(os.path.join(SAVE_PATH, "loss.pdf"))
plt.show()

# Show the maze solution and the policy learnt
net.eval()
agent.is_game_on = True
agent.use_softmax = False
_ = agent.env.reset(0)
counter = 0
while agent.is_game_on:
    agent.make_move(net, 0)
    if counter % 10 == 0:
        agent.env.draw(os.path.join(SAVE_PATH, f"step_{counter}.pdf"))
    counter += 1
agent.env.draw(os.path.join(SAVE_PATH, f"step_{counter}.pdf"))

# print the policy map
agent.plot_policy_map(net, os.path.join(
    SAVE_PATH, "solution.pdf"), [0.35, -0.3])

best_net = copy.deepcopy(net)
best_net.load_state_dict(torch.load(
    os.path.join(MODEL_PATH, "best_model.torch")))

agent.plot_policy_map(best_net, os.path.join(
    SAVE_PATH, "best_solution.pdf"), [0.35, -0.3])
