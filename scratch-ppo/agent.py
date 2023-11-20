import os
import copy
import time
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from utils import *
from network import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(
        self,
        envs,
        state_dim,
        action_dim,
        writer,
        model_save_path):

        self.device = device
        self.envs = envs

        # Model hyperparameters
        self.gamma = 0.95
        self.clip = 0.2 
        self.max_timesteps_per_episode = 200
        self.timesteps_per_batch = 1024
        self.n_updates_per_iteration = 10

        # Logging parameters
        self.save_freq = 1000
        self.model_save_path = model_save_path
        self.print_freq = 100
        self.writer = writer

        self.cov_mat = torch.diag(torch.full(size=(action_dim,), fill_value=0.5))


        self.con = Controller(
            actor_input=state_dim,
            actor_output=action_dim,
            critic_input=state_dim,
            )

        self.subgoal_transition_fcn_freq = 10
        self.subgoal = None

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0
        i_so_far = 0 
        
        while t_so_far < total_timesteps:
            self.writer.flush()

            # collect batch data via rollouts
            batch_state, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            
            avg_loss = []
            V, _ = self.evaluate(batch_state, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)


            avg_batch_rtgs = torch.mean(batch_rtgs) # is this correct metric of computing the mean?
            avg_ep_len = np.mean(batch_lens)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_state, batch_acts)

                actor_loss, critic_loss = self.con.optimizer_step(V, curr_log_probs, batch_log_probs, batch_rtgs, A_k)

                avg_loss.append(actor_loss)
            
            if self.writer:
                self.writer.add_scalar("loss/actor_loss", actor_loss, i_so_far)
                self.writer.add_scalar("loss/critic_loss", critic_loss, i_so_far)

                self.writer.add_scalar("reward/mean_batch_rew", avg_batch_rtgs, i_so_far)

                self.writer.add_scalar("len/avg_ep_len", avg_ep_len, i_so_far)

            if i_so_far % self.print_freq == 0:
                print(flush=True)
                print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
                print(f"Average Episodic Length: {np.mean(batch_lens)}", flush=True)
                print(f"Average Episodic Controller RTG: {avg_batch_rtgs}", flush=True)
                print(f"Average Controller Loss: {np.mean(avg_loss)}", flush=True)
                print(f"Timesteps So Far: {t_so_far}", flush=True)
                print(f"------------------------------------------------------", flush=True)
                print(flush=True)

            if i_so_far % self.save_freq == 0:
                torch.save(self.con.actor.state_dict(), '%s/ppo_actor_%d.pth' % (self.model_save_path, i_so_far))
                torch.save(self.con.critic.state_dict(), '%s/ppo_critic_%d.pth' % (self.model_save_path, i_so_far))

        print(f"Training Completed", flush=True)

        torch.save(self.con.actor.state_dict(), '%s/SB_ppo_actor_FINAL.pth' % (self.model_save_path))

        self.writer.close()

    def rollout(self):
        batch_state = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        
        ep_rewards = []
        
        t = 0 # Timesteps so far
        
        while t < self.timesteps_per_batch:
            ep_rewards = [] # Rewards per episode
            state, info = self.envs.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_state.append(state)
                action, log_prob = self.get_action(state)
                
                state, rew, term, trunc, _ = self.envs.step(action)
                done = term or trunc

                ep_rewards.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rewards)
            
        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_state, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    def compute_rtgs(self, batch_rews): # Compute reward-to-go
        batch_rtgs = []
        for ep_rewards in reversed(batch_rews):
            discounted_reward = 0 
            for rew in reversed(ep_rewards):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, state):
        mean = self.con.actor(torch.tensor(state, dtype=torch.float))
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
        
    def evaluate(self, batch_state, batch_acts):
        V = self.con.critic(batch_state).squeeze()

        mean = self.con.actor(batch_state)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs

