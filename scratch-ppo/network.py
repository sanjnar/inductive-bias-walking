import torch
import argparse
import os
import random
import time

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *

class Controller(nn.Module):
    def __init__(self, actor_input, actor_output, critic_input, layers=64, lr=2.55673e-5):
        super().__init__()
        self.clip = 0.3
		# lr = 0.001

        self.actor = FeedForwardNN(actor_input, actor_output, layer=layers)
        self.critic = FeedForwardNN(critic_input, 1, layer=layers)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        print('parameters', lr, layers, self.clip)



class FeedForwardNN(nn.Module):
	def __init__(self, in_dim, out_dim ,layer=64):
		super(FeedForwardNN, self).__init__()
        # layer: 64
		self.layer1 = nn.Linear(in_dim, layer)
		self.layer2 = nn.Linear(layer, layer)
		self.layer3 = nn.Linear(layer, out_dim)

	def forward(self, obs):
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
