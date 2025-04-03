import os
import sys
import time
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from jump_env.jumper_env import JumperEnv
from jump_env.custom_policy.custom_policy import CNNPPOPolicy
from jump_env.custom_policy.observation_stack_wrapper import ObservationStackWrapper

def make_env(rank):
    def _init():
        env = JumperEnv(render=False, log_interval=999999)
        env = TimeLimit(env, max_episode_steps=1500)
        env = ObservationStackWrapper(env, stack_size=4)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    NUM_ENVS = 4

    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    eval_env = make_env("eval")()

    model = PPO(
        CNNPPOPolicy,
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs/"
    )


    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=200_000, callback=eval_callback)
