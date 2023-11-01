import os 

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class EvaluateLSTM(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard
    """
    
    def __init__(self, eval_freq, eval_env, name, num_episodes=20, verbose=0):
        super(EvaluateLSTM, self).__init__()
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.name = name
        self.num_episodes = num_episodes
        
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.num_timesteps % self.eval_freq ==0:
            
            perfs = [] 
            for _ in range(self.num_episodes):
                lstm_states, cum_rew, step = None, 0, 0
                obs = self.eval_env.reset()
                episode_starts = np.ones((1,),dtype=bool)
                done = False
                while not done:
                    (
                        action,
                        lstm_states,
                    ) = self.model.predict(
                        self.training_env.normalize_obs(
                            obs
                        ),
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    obs, rewards, done, _ = self.eval_env.step(action)
                    episode_starts = done
                    cum_rew += rewards
                    step += 1
                perfs.append(cum_rew)
            
            
            self.logger.record(self.name, np.mean(perfs))
        return True
    
class EnvDumpCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.save_path = save_path
        
    def _on_step(self):
        env_path = os.path.join(self.save_path, "training_env.pkl")
        if self.verbose > 0:
            print("Saving the training environment to path ", env_path)
        self.training_env.save(env_path)
        return True
    
class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.info_keywords = info_keywords
        self.rollout_info = {}
    
    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}
        
    def _on_step(self):
        for key in self.info_keywords:
            vals = [info[key] for info in self.locals["infos"]]
            self.rollout_info[key].extend(vals)
        return True
    
    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record("rollout/" + key, np.mean(self.rollout_info[key]))


def set_callbacks(env, log):
    # Define callbacks for evaluation and saving the agent
    eval_callback = EvalCallback(
        eval_env=env,
        callback_on_new_best=EnvDumpCallback(log, verbose=0),
        n_eval_episodes=10,
        best_model_save_path=log,
        log_path=log,
        eval_freq=1_000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=log,
        save_vecnormalize=True,
        verbose=1,
    )

    return [eval_callback, checkpoint_callback]

# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_name, ep_steps, log, num_env, start_index=0):
    def make_env(_):
        def _thunk():
            env = gymnasium.make(env_name,render_mode='rgb_array')
            env._max_episode_steps = ep_steps
            env = Monitor(env, log)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])