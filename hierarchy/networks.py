import torch
import argparse
import os
import random
import time

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from utils import *

class Controller(nn.Module):
    # def __init__(self, state_dim, goal_dim, action_dim):
    def __init__(self, actor_input, actor_output, critic_input):
        super().__init__()

        self.clip = 0.2
        self.actor_lr = 0.005
        self.critic_lr = 0.005

        self.critic = nn.Sequential( # Consider using layer init
            # nn.Linear(state_dim + goal_dim + action_dim, 300),
            nn.Linear(critic_input, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 1),
        )

        self.actor = nn.Sequential(
            # nn.Linear(state_dim + goal_dim, 300),
            nn.Linear(actor_input, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            # nn.Linear(300, action_dim),
            nn.Linear(300, actor_output),
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # obs = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape).to(device)
        # actions = torch.zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape).to(device)
        # logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        # rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        # dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        # values = torch.zeros((self.num_steps, self.num_envs)).to(device)

    def optimizer_step(V, curr_log_probs, batch_log_probs, A_k):
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # self.logger['actor_losses'].append(actor_loss.detach())

