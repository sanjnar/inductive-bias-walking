import gymnasium as gym
import os
import torch
import numpy as np
from utils import *
import argparse
from datetime import datetime
from agent import Agent
import shutil
from network import Controller

from torch.utils.tensorboard import SummaryWriter

from evaluate import eval_policy


ENV_NAME    = "Humanoid-v4"
LOGGING_NAME = "scratch-ppo-humanoid-v4"
NOW = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join("output", "training", NOW) + "_" + LOGGING_NAME
MAX_EPISODE_STEPS = 1000  # default: 100 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()

    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    if args.train:
        print('Setting up environment', ENV_NAME)
        envs = gym.make(ENV_NAME)

        # parallelize envs
        # envs = make_parallel_envs(ENV_NAME, MAX_EPISODE_STEPS, TENSORBOARD_LOG, 16)
        # envs = VecNormalize(envs)

        # callbacks = set_callbacks(envs, TENSORBOARD_LOG)

        # set up logging
        writer = SummaryWriter(log_dir=TENSORBOARD_LOG)

        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]

        # vec_env = make_vec_env("BipedalWalker-v3", n_envs=1)
        # model = PPO("MlpPolicy", vec_env, verbose=1)
        # model.learn(total_timesteps=2500)

        # mean_reward, std_reward = evaluate_policy(model.policy, vec_env, n_eval_episodes=5, deterministic=True)
        # print("mean reward", mean_reward, "std reward", std_reward)
        print('Setting up agent')
        agent = Agent(
                    envs=envs,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    writer=writer,
                    model_save_path=TENSORBOARD_LOG,
                    )

        # # setting libraries seeds to try and have repeatability
        # # torch.manual_seed(args.seed)
        # # np.random.seed(args.seed)
        # # env.seed(args.seed)

        print("start agent learning")
        agent.learn(total_timesteps=25000000)


    if args.eval: # TODO: Needs implementation
        envs = gym.make(ENV_NAME, render_mode='rgb_array')

        VIDEO_LOG = "output/frames/"
        TRAINING_KEY = '2023-11-20/18-52-26_scratch-ppo-humanoid-v4'
        # high_actor_model = 'output/training/2023-11-13/14-05-51_bipedalwalker/ppo_high_actor_FINAL.pth'
        # low_actor_model = 'output/training/2023-11-13/14-05-51_bipedalwalker/ppo_low_actor_FINAL.pth'

        actor_model = 'output/training/2023-11-20/18-52-26_scratch-ppo-humanoid-v4/ppo_actor_179000.pth'
        eval_log_dir = "output/eval/" + TRAINING_KEY

        # set up logging
        writer = SummaryWriter(log_dir=eval_log_dir)

        state_dim = envs.observation_space.shape[0]
        action_dim = envs.action_space.shape[0]
        subgoal_dim = state_dim

        con = Controller(
            actor_input=state_dim,
            actor_output=action_dim,
            critic_input=state_dim,
            )
            
        con.actor.load_state_dict(torch.load(actor_model))

        eval_policy(env=envs, num_episodes=5, policy=con, writer=writer, render=True,video_log=VIDEO_LOG, training_key=TRAINING_KEY)

if __name__ == "__main__":
    main()
