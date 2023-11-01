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
from networks import Controller

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HierarchicalAgent():
    def __init__(
        self,
        envs,
        state_dim,
        action_dim,
        goal_dim,
        subgoal_dim,
        scale_low):

        self.gamma = 0.95 

        # self.subgoal_obj = Subgoal(subgoal_dim) # TODO: set limits on subgoal space? Currently set as same as action space
        # scale_high = self.action_space.high * np.ones(subgoal_dim)

        # Model parameters
        model_save_freq = 2000
        model_path = "model"

        batch_size = 100
        buffer_freq = 10

        start_training_steps = 2500 

        subgoal_dim = state_dim     # subgoal should be same as state (footstep planner)
        self.state_dim = state_dim # TODO: DELETE, temporary for footstep planner

        self.cov_act_mat = torch.diag(torch.full(size=(action_dim,), fill_value=0.5))
        self.cov_subg_mat = torch.diag(torch.full(size=(subgoal_dim,), fill_value=0.5))

        self.device = device

        self.high_con = Controller(
            actor_input=state_dim,
            actor_output=state_dim,
            critic_input=state_dim+action_dim,
            # state_dim=state_dim,
            # goal_dim=0, # not setting goal for high level controller
            # action_dim=action_dim,
            )

        self.low_con = Controller(
            actor_input=state_dim+subgoal_dim,
            actor_output=action_dim,
            critic_input=state_dim+action_dim+subgoal_dim,
            # state_dim=state_dim,
            # goal_dim=subgoal_dim,
            # action_dim=action_dim,
            )

        self.envs = envs
        self.buffer_freq = buffer_freq

        self.subgoal = None # TODO: Footstep planner

        self.start_training_steps = start_training_steps
        # self.max_timesteps_per_episode = 200
        # self.timesteps_per_batch = 2048

        self.max_timesteps_per_episode = 10
        self.timesteps_per_batch = 10

        print('initialized agent')

    def get_action(self, state, subgoal):
        state = torch.FloatTensor(state).to(self.device)
        subgoal = torch.FloatTensor(subgoal).to(self.device)
        mean_action = self.low_con.actor(torch.cat([state, subgoal]))
        dist = MultivariateNormal(mean_action, self.cov_act_mat) # vs Categorical
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def get_subgoal(self, step, state, new_state):
        state = torch.FloatTensor(state).to(self.device)
        mean_subgoal = self.high_con.actor(state)
        dist = MultivariateNormal(mean_subgoal, self.cov_subg_mat)
        subgoal = dist.sample()
        
        log_prob = dist.log_prob(subgoal)
        
        if step % self.buffer_freq == 0:
            return subgoal.detach().numpy(), log_prob.detach()
        else:
            state = torch.FloatTensor(state).to(self.device)
            footstep_subgoal = torch.FloatTensor(self.get_footstep_subgoal(state)).to(self.device)
            new_state = torch.FloatTensor(new_state).to(self.device)

            return state + footstep_subgoal - new_state, log_prob.detach() # TODO: What to return when using transition function?


    def get_footstep_subgoal(self, state):
        # TODO: Implement g* footstep controller
        return torch.ones(state.shape)

    def get_intrinsic_reward(self, state, subgoal, next_state):
        state = torch.FloatTensor(state).to(self.device)
        subgoal = torch.FloatTensor(subgoal).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)

        return torch.norm(state + subgoal - next_state, 2)

    def rollout_step(self):
        print('rollout step')
        
        batch_state = []
        batch_acts = []
        batch_subgoals = []
        batch_log_probs = []
        batch_rews_high = []
        batch_rews_low = []
        batch_rtgs = []
        batch_lens = []
        
        ep_rews = []
        t = 0
        
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode
            sub_rews= []

            done = False

            state, info = self.envs.reset()
            subgoal = self.get_footstep_subgoal(state)

            step = 0
            for ep_t in range(self.max_timesteps_per_episode):
                
                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_state.append(state)

                # low level controllers action
                action, log_prob = self.get_action(state, subgoal)

                next_state, reward, terminated, truncated, info = self.envs.step(action)
                done = terminated or truncated

                # high level controller
                next_subgoal, log_prob = self.get_subgoal(ep_t, state, next_state)
                self.next_subgoal = next_subgoal

                ep_rews.append(reward)
                sub_rews.append(self.get_intrinsic_reward(state, next_subgoal, next_state))
                batch_acts.append(action)
                batch_subgoals.append(next_subgoal)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews_low.append(sub_rews)
            batch_rews_high.append(ep_rews)
            
        # consider for replay buffer (not necessary for PPO)

        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_subgoals = torch.tensor(batch_subgoals, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs_high = torch.tensor(self.compute_rtgs(batch_rews_high), dtype=torch.float)                
        batch_rtgs_low = torch.tensor(self.compute_rtgs(batch_rews_low), dtype=torch.float)             

        # self.logger['batch_rews'] = batch_rews
        # self.logger['batch_lens'] = batch_lens
        
        return batch_state, batch_acts, batch_subgoals, batch_log_probs, batch_rtgs_low, batch_rtgs_high, batch_lens
        
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
        
    def evaluate_low(self, batch_state, batch_subgoals, batch_acts):
        print('evaluate')
        # estimate values of obs + subgoals, compute log probs

        V = self.low_con.critic(torch.cat([batch_state, batch_acts, batch_subgoals], 1)).squeeze() # TODO: Check that critic input makes sense?

        mean = self.low_con.actor(torch.cat([batch_state, batch_subgoals])) # TODO: append batch subgoals
        dist = MultivariateNormal(mean, self.cov_act_mat)
        log_probs = dist.log_prob(batch_acts)

        return torch.tensor(V, dtype=torch.float), log_probs

    def evaluate_high(self, batch_state, batch_acts, batch_subgoals):
        print('evaluate')
        # estimate values of obs, compute log probs
        print('batch stat', batch_state.shape)
        V = self.high_con.critic(torch.cat([batch_state, batch_acts], 1)).squeeze() # TODO: Check that critic input makes sense?
        print('compute V')
        mean = self.high_con.actor(batch_state)
        print('ok mean')
        dist = MultivariateNormal(mean, self.cov_subg_mat)
        log_probs = dist.log_prob(batch_subgoals)

        return torch.tensor(V, dtype=torch.float), log_probs

    def learn(self, total_timesteps):
        # HIGH LEVEL: Input: state. Output: Subgoal
        # LOW LEVEL: Input: Subgoal, state. Output: Low level atomic action
        
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        
        while t_so_far < total_timesteps:    
            # collect batch data via rollouts
            batch_state, batch_acts, batch_subgoals, batch_log_probs, batch_rtgs_low, batch_rtgs_high, batch_lens = self.rollout_step()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            # self.logger['t_so_far'] = t_so_far
            # self.logger['i_so_far'] = i_so_far

            print('calculate advantages')
            # calculate advantages for high level
            V, _ = self.evaluate_high(batch_state, batch_acts, batch_subgoals)
            print('batch types', type(batch_rtgs_high), 'V', type(V))
            A_k_high = batch_rtgs_high - V.detach()         
            A_k_high = (A_k_high - A_k_high.mean()) / (A_k_high.std() + 1e-10)

            # calculate advantages for low level
            V, _ = self.evaluate_low(batch_state, batch_subgoals, batch_acts)
            A_k_low = batch_rtgs - V.detach()      
            A_k_low = (A_k_low - A_k_low.mean()) / (A_k_low.std() + 1e-10)        

            print('iteration updates')
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate_high(batch_state, batch_subgoals)
                self.low_con.optimizer_step(V, curr_log_probs)
                
                V, curr_log_probs = self.evaluate_low(batch_state, batch_subgoals, batch_acts)
                self.high_con.optimizer_step(V, curr_log_probs)

            self._log_summary()

            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

