import gym
import os
import torch
import numpy as np
from utils import *
import argparse
from datetime import datetime
from agent import HierarchicalAgent
import shutil

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

ENV_NAME    = "Humanoid-v4"
NOW = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join("output", "training", NOW) + "_humanoid"
MAX_EPISODE_STEPS = 1000  # default: 100 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    if args.train:
        print('Setting up environment', ENV_NAME)
        # envs = make_parallel_envs(ENV_NAME, MAX_EPISODE_STEPS, TENSORBOARD_LOG, 16)
        # envs = VecNormalize(envs)
        envs = gymnasium.make(ENV_NAME)

        callbacks = set_callbacks(envs, TENSORBOARD_LOG)

        goal_dim = 2
        subgoal_dim = 17 # what is subgoal? sample of action space? 
        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]
        scale = envs.action_space.high * np.ones(action_dim)

        print('Setting up agent')
        agent = HierarchicalAgent(
                    envs=envs,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    goal_dim=goal_dim,
                    subgoal_dim=subgoal_dim,
                    scale_low=scale,
                    )

        # # setting libraries seeds to try and have repeatability
        # torch.manual_seed(args.seed)
        # np.random.seed(args.seed)
        # env.seed(args.seed)

        # Start training
        # trainer = Trainer(env, agent, TENSORBOARD_LOG)
        # trainer.train()
        print("start agent learning")
        agent.learn(total_timesteps=1000)

    if args.eval:
        run_evaluation(args, env, agent)



if __name__ == "__main__":
    main()
