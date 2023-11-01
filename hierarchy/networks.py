import torch
import argparse
import os
import random
import time

import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils import *

class Controller(nn.Module):
    def __init__(self, actor_input, actor_output, critic_input):
        super().__init__()

        self.clip = 0.2
        self.actor_lr = 0.005
        self.critic_lr = 0.005

        self.critic = nn.Sequential( # consider using layer init
            nn.Linear(critic_input, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, 1),
        )

        self.actor = nn.Sequential(
            nn.Linear(actor_input, 300),
            nn.Tanh(),
            nn.Linear(300, 300),
            nn.Tanh(),
            nn.Linear(300, actor_output),
        )

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def optimizer_step(self, V, curr_log_probs, batch_log_probs, batch_rtgs, A_k):
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)
        critic_loss.requires_grad = True

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # self.logger['actor_losses'].append(actor_loss.detach())

