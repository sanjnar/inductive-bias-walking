import os
import copy
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.gamma = 0.99
        self.entropy_coeff = 3.61209e-05
        self.clip = 0.3

        self.save_freq = 1000
        self.model_save_path = model_save_path
        self.writer = writer
        self.print_freq = 100
        
        self.cov_var = torch.full(size=(action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        # self.cov_act_mat = torch.eye(action_dim)
        # self.cov_subg_mat = torch.eye(subgoal_dim)

        self.device = device

        self.con = Controller(
            actor_input=state_dim,
            actor_output=action_dim,
            critic_input=state_dim,
            )

        self.envs = envs
        self.subgoal_transition_fcn_freq = 10

        self.subgoal = None

        self.max_timesteps_per_episode = 200
        self.timesteps_per_batch = 2048

        self.n_updates_per_iteration = 10
        print('parameters: gamma', self.gamma, 'entropy coeff', self.entropy_coeff, 'max time', self.max_timesteps_per_episode, 'batch times', self.timesteps_per_batch, 'updates per iteration', self.n_updates_per_iteration)

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0
        i_so_far = 0 
        
        while t_so_far < total_timesteps:
            self.writer.flush()
            # collect batch data via rollouts
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout() # TODO: make mini batches from this
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            
            avg_loss = []

			# Calculate advantages for the rollout 
            V, _, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            avg_batch_rtgs = torch.mean(batch_rtgs)
            avg_ep_len = np.mean(batch_lens)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                
                entropy_loss = entropy.mean()

                actor_loss = actor_loss - self.entropy_coeff * entropy_loss

                print('actor parameters')
                for name, param in self.con.actor.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: {param.data}")

                print('critic parameters')
                for name, param in self.con.critic.named_parameters():
                    if param.requires_grad:
                        print(f"{name}: {param.data}")

                self.con.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.con.actor_optimizer.step()
                
                self.con.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.con.critic_optimizer.step()

                avg_loss.append(actor_loss.detach())
            
            if self.writer:
                self.writer.add_scalar("loss/actor_loss", actor_loss, i_so_far)
                self.writer.add_scalar("loss/critic_loss", critic_loss, i_so_far)
                self.writer.add_scalar("loss/entropy_loss", entropy_loss, i_so_far)

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
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        
        ep_rews = []
        
        t = 0 # Keeps track of how many timesteps we've run so far this batch
        
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            obs, info = self.envs.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1 # Increment timesteps ran this batch so far
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                
                obs, rew, term, trunc, _ = self.envs.step(action)
                done = term or trunc


                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    def compute_rtgs(self, batch_rews):
        # compute reward-to-go
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
                
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, state):
        mean = self.con.actor(state)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
        
    def evaluate(self, batch_obs, batch_acts):
        V = self.con.critic(batch_obs).squeeze()

        mean = self.con.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs, dist.entropy()

