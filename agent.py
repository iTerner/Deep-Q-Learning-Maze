import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import collections

Transition = collections.namedtuple("Experience", field_names=[
                                    "state", "action", "next_state", "reward", "is_game_on"])


class Agent:
    def __init__(self, maze, memory_buffer, device, use_softmax=True):
        self.env = maze  # the environment (in our case this is a maze)
        self.buffer = memory_buffer  # this is actually a reference
        self.num_act = 4  # number of actions possible
        self.use_softmax = use_softmax  # whether to use softmax or not
        self.total_reward = 0  # total reward
        self.min_reward = -self.env.maze.size  # minimum reward
        self.is_game_on = True  # whether the game is still running or not
        self.device = device  # the device we are using (cpu or cuda)

    def make_move(self, net, epsilon, device):
        # select an action
        action = self.select_action(net, epsilon, device)
        current_state = self.env.state()
        next_state, reward, self.is_game_on = self.env.update_state(action)
        # add to the total reward
        self.total_reward += reward

        if self.total_reward < self.min_reward:  # stop the game in this case
            self.is_game_on = False
        if not self.is_game_on:
            self.total_reward = 0  # reset the total reward

        transition = Transition(current_state, action,
                                next_state, reward, self.is_game_on)

        self.buffer.push(transition)

    def select_action(self, net, epsilon, device):
        state = torch.Tensor(self.env.state()).to(device).view(1, -1)
        qvalues = net(state).cpu().detach().numpy().squeeze()

        # softmax sampling of the qvalues
        if self.use_softmax:
            p = sp.softmax(qvalues / epsilon).squeeze()
            p /= np.sum(p)
            action = np.random.choice(self.num_act, p=p)
        else:
            # else choose the best action with probability 1-epsilon
            # and with probability epsilon choose at random
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]
            else:
                action = np.argmax(qvalues, axis=0)
                action = int(action)

        return action

    def plot_policy_map(self, net, filename, offset):
        net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, "Greys")

            # loop on all the allowed cells
            for free_cell in self.env.allowed_states:
                # get the action
                self.env.current_pos = np.asarray(free_cell)
                qvalues = net(torch.Tensor(self.env.state()
                                           ).view(1, -1).to(self.device))
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
                # get the policy
                policy = self.env.directions[action]

                ax.text(free_cell[1] - offset[0],
                        free_cell[0]-offset[1], policy)

            ax = plt.gca()
            plt.xticks([], [])
            plt.yticks([], [])
            ax.plot(self.env.goal[1], self.env.goal[0], "bs", markersize=4)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.show()
