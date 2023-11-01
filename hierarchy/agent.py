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
        writer,
        model_save_path):

        self.gamma = 0.95

        # Model parameters
        self.save_freq = 2000
        self.model_path = model_save_path


        subgoal_dim = state_dim     # subgoal should be same as state (footstep planner)

        self.cov_act_mat = torch.diag(torch.full(size=(action_dim,), fill_value=0.5))
        self.cov_subg_mat = torch.diag(torch.full(size=(subgoal_dim,), fill_value=0.5))

        self.device = device

        self.high_con = Controller(
            actor_input=state_dim,
            actor_output=state_dim,
            critic_input=state_dim+action_dim,
            )

        self.low_con = Controller(
            actor_input=state_dim+subgoal_dim,
            actor_output=action_dim,
            critic_input=state_dim+subgoal_dim+action_dim,
            )

        self.envs = envs
        self.subgoal_transition_fcn_freq = 10

        self.subgoal = None # TODO: Footstep planner

        self.max_timesteps_per_episode = 200
        self.timesteps_per_batch = 2048
        self.n_updates_per_iteration = 5

        print('initialized agent')

    def get_action(self, state, subgoal):
        state = torch.FloatTensor(state).to(self.device)
        subgoal = torch.FloatTensor(subgoal).to(self.device)
        mean_action = self.low_con.actor(torch.cat([state, subgoal]))
        dist = MultivariateNormal(mean_action, self.cov_act_mat) # vs categorical, normal
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def get_subgoal(self, step, state, new_state):
        state = torch.FloatTensor(state).to(self.device)
        mean_subgoal = self.high_con.actor(state)
        dist = MultivariateNormal(mean_subgoal, self.cov_subg_mat)
        subgoal = dist.sample()
        
        log_prob = dist.log_prob(subgoal)
        
        if step % self.subgoal_transition_fcn_freq == 0:
            return subgoal.detach().numpy(), log_prob.detach()
        else:
            state = torch.FloatTensor(state).to(self.device)
            footstep_subgoal = torch.FloatTensor(self.get_footstep_subgoal(state)).to(self.device)
            new_state = torch.FloatTensor(new_state).to(self.device)

            return state + footstep_subgoal - new_state, log_prob.detach() # TODO: What to return when using transition function? Consider 0 covar matrix


    def get_footstep_subgoal(self, state):
        # g* footstep controller -- specific to env
        # TODO: Implement me!
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
        batch_log_probs_high = []
        batch_log_probs_low = []
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
                
                t += 1
                batch_state.append(state)

                # low level controllers
                action, log_prob_low = self.get_action(state, subgoal)

                next_state, reward, terminated, truncated, info = self.envs.step(action)
                done = terminated or truncated

                # high level controller
                next_subgoal, log_prob_high = self.get_subgoal(ep_t, state, next_state)
                self.next_subgoal = next_subgoal

                ep_rews.append(reward)
                sub_rews.append(self.get_intrinsic_reward(state, next_subgoal, next_state))
                batch_acts.append(action)
                batch_subgoals.append(next_subgoal)
                batch_log_probs_low.append(log_prob_low)
                batch_log_probs_high.append(log_prob_high)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews_low.append(sub_rews)
            batch_rews_high.append(ep_rews)
            
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_subgoals = torch.tensor(np.array(batch_subgoals), dtype=torch.float)
        batch_log_probs_low = torch.tensor(np.array(batch_log_probs_low), dtype=torch.float)
        batch_log_probs_high = torch.tensor(np.array(batch_log_probs_high), dtype=torch.float)
        batch_rtgs_high = torch.tensor(np.array(self.compute_rtgs(batch_rews_high)), dtype=torch.float)                
        batch_rtgs_low = torch.tensor(np.array(self.compute_rtgs(batch_rews_low)), dtype=torch.float)             

        if self.writer:
            self.writer.add_scalar("reward/high_mean_batch_rew", torch.mean(batch_rtgs_high))
            self.writer.add_scalar("reward/low_mean_batch_rew", torch.mean(batch_rtgs_low))
            
            self.writer.add_scalar("len/mean_batch_len", torch.mean(batch_lens))

            self.writer.add_scalar("reward/high_mean_batch_rew", low_actor_loss)
            self.writer.add_scalar("reward/high_mean_batch_rew", low_actor_loss)
        
        return batch_state, batch_acts, batch_subgoals, batch_log_probs_low, batch_log_probs_high, batch_rtgs_low, batch_rtgs_high, batch_lens
        
    def compute_rtgs(self, batch_rews):
        # compute reward-to-go
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        return batch_rtgs
        
    def evaluate_low(self, batch_state, batch_subgoals, batch_acts):
        print('evaluate')
        # estimate values of obs + subgoals, compute log probs

        V = self.low_con.critic(torch.cat([batch_state, batch_subgoals, batch_acts], 1)).squeeze(0)

        mean = self.low_con.actor(torch.cat([batch_state, batch_subgoals], 1))
        dist = MultivariateNormal(mean, self.cov_act_mat)
        log_probs = dist.log_prob(batch_acts)

        return V.clone().detach(), log_probs

    def evaluate_high(self, batch_state, batch_subgoals, batch_acts):
        # estimate values of obs, compute log probs
        V = self.high_con.critic(torch.cat([batch_state, batch_acts], 1)).squeeze(0)
        mean = self.high_con.actor(batch_state)
        dist = MultivariateNormal(mean, self.cov_subg_mat)
        log_probs = dist.log_prob(batch_subgoals)

        return V.clone().detach(), log_probs

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0 
        i_so_far = 0 
        
        while t_so_far < total_timesteps:    
            # collect batch data via rollouts
            batch_state, batch_acts, batch_subgoals, batch_log_probs_low, batch_log_probs_high, batch_rtgs_low, batch_rtgs_high, batch_lens = self.rollout_step()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            
            # self.logger['t_so_far'] = t_so_far
            # self.logger['i_so_far'] = i_so_far

            print('calculate advantages')
            # calculate advantages for high level
            V, _ = self.evaluate_high(batch_state, batch_subgoals, batch_acts)
            A_k_high = batch_rtgs_high - V.detach()         
            A_k_high = (A_k_high - A_k_high.mean()) / (A_k_high.std() + 1e-10)

            # calculate advantages for low level
            V, _ = self.evaluate_low(batch_state, batch_subgoals, batch_acts)
            A_k_low = batch_rtgs_low - V.detach()      
            A_k_low = (A_k_low - A_k_low.mean()) / (A_k_low.std() + 1e-10)        

            print('iteration updates')
            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate_high(batch_state, batch_subgoals, batch_acts)
                low_actor_loss, low_critic_loss = self.low_con.optimizer_step(V, curr_log_probs, batch_log_probs_low, batch_rtgs_low, A_k_low)
                
                V, curr_log_probs = self.evaluate_low(batch_state, batch_subgoals, batch_acts)
                high_actor_loss, high_critic_loss = self.high_con.optimizer_step(V, curr_log_probs, batch_log_probs_high, batch_rtgs_high, A_k_high)

            if i_so_far % self.save_freq == 0:
                torch.save(self.low_con.actor.state_dict(), '%s/ppo_low_actor_%d.pth' % (self.model_save_path, i_so_far))
                torch.save(self.low_con.critic.state_dict(), '%s/ppo_low_critic_%d.pth' % (self.model_save_path, i_so_far))
                torch.save(self.high_con.actor.state_dict(), '%s/ppo_high_actor_%d.pth' % (self.model_save_path, i_so_far))
                torch.save(self.high_con.critic.state_dict(), '%s/ppo_high_critic_%d.pth' % (self.model_save_path, i_so_far))
            
            if self.writer:
                self.writer.add_scalar("loss/low_actor_loss", low_actor_loss)
                self.writer.add_scalar("loss/low_critic_loss", low_critic_loss)
                self.writer.add_scalar("loss/high_actor_loss", high_actor_loss)
                self.writer.add_scalar("loss/high_actor_loss", high_actor_loss)