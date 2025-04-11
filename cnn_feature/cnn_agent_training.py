import sys
import os
import time
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from jump_env.jumper_env import JumperEnv

from cnn_feature.observation_stack_wrapper import ObservationStackWrapper
from cnn_feature.custom_policy import CNNPPOPolicy


class ForceTensorboardLogging(BaseCallback):
    def _on_rollout_end(self) -> None:
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True


def make_env(rank, stack_size):
    def _init():
        env = JumperEnv(render=False, log_interval=999999)
        env = TimeLimit(env, max_episode_steps=1500)
        env = ObservationStackWrapper(env, stack_size=stack_size)
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    stack_size = 10  # 👈 choose your stack size here
    num_envs = 4

    timestamp = int(time.time())
    logdir = f"logs/{timestamp}/"
    os.makedirs(logdir, exist_ok=True)

    train_env = SubprocVecEnv([make_env(i, stack_size) for i in range(num_envs)])
    train_env = VecMonitor(train_env)

    eval_env = make_env("eval", stack_size)()
    eval_env = VecMonitor(eval_env)

    model = PPO(
        CNNPPOPolicy,
        train_env,
        verbose=1,
        tensorboard_log=logdir,
        policy_kwargs={
            "features_extractor_kwargs": {
                "output_dim": 128,  # 👈 Change if needed
            }
        },
    )

    callback = ForceTensorboardLogging()
    model.learn(total_timesteps=200_000, callback=callback)
