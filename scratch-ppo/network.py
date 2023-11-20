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
    def __init__(self, actor_input, actor_output, critic_input, layers=120, lr=2.5e-4):
        super().__init__()

        self.clip = 0.2
        
        self.actor_lr = lr
        self.critic_lr = lr

        self.actor = nn.Sequential(
            nn.Linear(actor_input, layers),
            nn.Tanh(),
            nn.Linear(layers, layers),
            nn.Tanh(),
            nn.Linear(layers, actor_output),
        )

        self.critic = nn.Sequential( # consider using layer init
            nn.Linear(critic_input, layers),
            nn.Tanh(),
            nn.Linear(layers, layers),
            nn.Tanh(),
            nn.Linear(layers, 1),
        )


        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def optimizer_step(self, V, curr_log_probs, batch_log_probs, batch_rtgs, A_k):
        ratios = torch.exp(curr_log_probs - batch_log_probs)
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = nn.MSELoss()(V, batch_rtgs)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
                
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()