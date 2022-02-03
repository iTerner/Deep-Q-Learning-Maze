import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy


class MazeEnvironment:
    def __init__(self, maze, init_pos, goal):
        x, y = len(maze), len(maze)

        self.bound = np.asarray([x, y])
        self.maze = maze  # the maze
        self.init_pos = init_pos  # the initial position in the maze
        self.goal = goal  # the goal position in the maze
        # the current position, initalize to the start position
        self.current_pos = np.asarray(init_pos)

        self.visited = set()  # a set of all the visited positions in the maze
        self.visited.add(tuple(self.current_pos))

        # initialize the empty cells and the distance (euclidean) from the goal
        # and removing the goal cell
        self.allowed_states = np.asarray(np.where(self.maze == 0)).T.tolist()
        self.distances = np.sqrt(
            np.sum((np.array(self.allowed_states) - np.asarray(self.goal)) ** 2, axis=1))

        del(self.allowed_states[np.where(self.distances == 0)[0][0]])
        self.distances = np.delete(
            self.distances, np.where(self.distances == 0)[0][0])

        # define action map
        # the agent has 4 possible actions: go right/left/down/up
        self.action_map = {
            0: [0, 1],  # right
            1: [0, -1],  # left
            2: [1, 0],  # down
            3: [-1, 0]  # up
        }

        self.directions = {
            0: "→",
            1: "←",
            2: "↓ ",
            3: "↑"
        }

    def reset_policy(self, eps, reg=7):
        """
        The function reset the policy, so that for high epsilon the inital position is 
        nearer to the goal (very useful for large mazes)
        Args:
            eps - the epsilon value
            reg = regularization value (default 7)
        Return:
            reset policy
        """
        return sp.softmax(-self.distances / (reg * (1 - eps ** (2 / reg))) ** (reg / 2)).squeeze()

    def reset(self, epsilon, prand=0):
        """
        The function reset the environment when the game is completed with a given probability.
        Args:
            epsilon - the epsilon value
            prnad - the probability value for the reset to be random, otherwise, the reset policy
            at the given epsilon is used
        Return:
            reset environment 
        """
        # random reset
        if np.random.rand() < prand:
            index = np.random.choice(len(self.allowed_states))
        else:
            p = self.reset_policy(epsilon)
            print(len(p))
            print(len(self.allowed_states))
            index = np.random.choice(len(self.allowed_states), p=p)

        self.current_pos = np.asarray(self.allowed_states[index])

        # initialize the visited positions
        self.visited = set()
        self.visited.add(tuple(self.current_pos))

        return self.state()

    def update_state(self, action):
        """
        The function updates the current state with respect to the given action
        Args:
            action - the selected action
        Return:
            list[maze state, reward, is game on]
        """
        is_game_on = True

        # each move costs -0.05
        reward = -0.05

        move = self.action_map[action]
        next_pos = self.current_pos + np.asarray(move)

        # if the goal has been reached, the agent get a reward of 1
        if (self.current_pos == self.goal).all():
            reward = 1
            is_game_on = False
            return [self.state(), reward, is_game_on]
        else:
            # if the cell has been visited before, the agent get a reward of -0.2
            if tuple(self.current_pos) in self.visited:
                reward = -0.2

        # if the move goes out of the maze or to a wall, the agent get a reward of -1
        if self.is_state_valid(next_pos):
            # change the current pos
            self.current_pos = next_pos
        else:
            reward = -1

        self.visited.add(tuple(self.current_pos))
        return [self.state(), reward, is_game_on]

    def state(self):
        """
        The function returns the state to be feeded to the network
        Return:
            state
        """
        state = copy.deepcopy(self.maze)
        state[tuple(self.current_pos)] = 2
        return state

    def check_boundaries(self, pos):
        """
        The function checks the boundaries
        Args:
            pos - the position to check
        Return:
            bool, whether or not the position is in boundaries
        """
        out = len([n for n in pos if n < 0])
        out += len([n for n in (self.bound - np.asarray(pos)) if n <= 0])
        return out > 0

    def check_wall(self, pos):
        """
        The function checks if the given position is a wall
        Args:
            pos - the position to check
        Return:
            bool, whether or not the position is a wall
        """
        return self.maze[tuple(pos)] == 1

    def is_state_valid(self, pos):
        """
        The function checks if the given position is valid pos
        Args:
            pos - the position to check
        Return:
            bool, whether or not the position is valid
        """
        if self.check_boundaries(pos):
            return False
        if self.check_wall(pos):
            return False
        return True

    def draw(self, filename):
        """
        The function draw some results from the
        Args:
            filename - the filename to save the image
        """
        plt.Figure()
        im = plt.imshow(self.maze, interpolation="none",
                        aspect="equal", cmap="Greys")
        ax = plt.gca()

        plt.xticks([], [])
        plt.yticks([], [])
        ax.plot(self.goal[1], self.goal[0], "bs", markersize=4)
        ax.plot(self.current_pos[1], self.current_pos[0], "rs", markersize=4)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()
