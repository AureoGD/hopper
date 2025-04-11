import os
import sys
import time
import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from jump_env.jumper_env import JumperEnv
from cnn_feature.custom_policy import CNNPPOPolicy
from jump_env.custom_policy.observation_stack_wrapper import ObservationStackWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

# ─────────────────────────────────────────────────────
# 🔧 Configurations
NUM_ENVS = 4
STACK_SIZE = 10  # <-- Number of past observations stacked
MAX_EPISODE_STEPS = 1500
TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 5000

timestamp = int(time.time())
models_dir = f"models/{timestamp}/"
logdir = f"logs/{timestamp}/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)


# ─────────────────────────────────────────────────────
def make_env(rank):
    def _init():
        env = JumperEnv(render=False, log_interval=999999)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env = ObservationStackWrapper(env, stack_size=STACK_SIZE)
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    # Training and evaluation environments
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    train_env = VecMonitor(train_env)  # logs ep_reward_mean and ep_len_mean

    eval_env = DummyVecEnv([make_env("eval")])
    eval_env = VecMonitor(eval_env)
    # ─────────────────────────────────────────────────────
    # 🧠 PPO with custom CNN policy
    model = PPO(
        CNNPPOPolicy,
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        policy_kwargs=dict(
            features_extractor_kwargs=dict(stack_size=10, output_dim=128)  # ⬅️ here
        ),
    )
    # Evaluation callback to track best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(models_dir, "best_model"),
        log_path=os.path.join(logdir, "eval"),
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    # ─────────────────────────────────────────────────────
    # 🚀 Start training
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="CNN_PPO",
    )

    model.save(os.path.join(models_dir, "final_model"))
    train_env.close()
    eval_env.close()
