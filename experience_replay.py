import collections
import numpy as np
import torch


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size, device):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, next_states, rewards, isgameon = zip(
            *[self.memory[idx] for idx in indices])

        return torch.Tensor(states).type(torch.float).to(device), torch.Tensor(actions).type(torch.long).to(device), torch.Tensor(next_states).to(device), torch.Tensor(rewards).to(device), torch.tensor(isgameon).to(device)
