import os
import time
import argparse
import importlib
from typing import List

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from jump_env.jumper_env import JumperEnv


# Custom Callback to force tensorboard log flushing
class ForceTensorboardLogging(BaseCallback):
    def _on_rollout_end(self) -> None:
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True


# Environment factory


def make_env(rank: int, policy_type: str = "DNN", stack_size: int = 1):
    def _init():
        env = JumperEnv(render=False)
        env = TimeLimit(env, max_episode_steps=1500)

        if policy_type == "CNN" and stack_size > 1:
            from cnn_feature.observation_stack_wrapper import ObservationStackWrapper

            env = ObservationStackWrapper(env, stack_size=stack_size)

        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["DNN", "CNN"], default="DNN")
    parser.add_argument("--stack_size", type=int, default=4)
    parser.add_argument("--hidden_layers", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--save_freq", type=int, default=25_000)
    args = parser.parse_args()

    timestamp = int(time.time())
    base_dir = f"agents/{args.policy}_{timestamp}"
    model_path = os.path.join(base_dir, "models")
    log_path = os.path.join("training_logs", args.policy, str(timestamp))
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Save config
    with open(os.path.join(base_dir, "config.txt"), "w") as f:
        f.write(str(vars(args)))

    # Load policy class and feature extractor
    if args.policy == "CNN":
        from cnn_feature.custom_policy import CNNPPOPolicy

        policy_class = CNNPPOPolicy
        policy_kwargs = {
            "features_extractor_kwargs": {
                "stack_size": args.stack_size,
                "output_dim": 128,
            },
            "net_arch": args.hidden_layers,
        }
    else:  # DNN
        policy_class = "MlpPolicy"
        policy_kwargs = {"net_arch": args.hidden_layers}

    # Setup environments with policy_type passed for correct wrapper usage
    train_env = SubprocVecEnv(
        [make_env(i, policy_type=args.policy, stack_size=args.stack_size) for i in range(args.num_envs)]
    )
    train_env = VecMonitor(train_env, filename=os.path.join(log_path, "monitor.csv"))

    eval_env = DummyVecEnv([make_env("eval", policy_type=args.policy, stack_size=args.stack_size)])
    eval_env = VecMonitor(eval_env)

    # Setup model
    model = PPO(
        policy_class,
        train_env,
        verbose=1,
        tensorboard_log=log_path,
        n_steps=512,
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    callback = ForceTensorboardLogging()
    total_timesteps = args.total_timesteps
    best_reward = float("-inf")

    TIMESTEPS = 2048

    while model.num_timesteps < total_timesteps:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback)

        # Evaluate current policy
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            model.save(os.path.join(model_path, "best_model"))

        # Periodically save model
        if model.num_timesteps % args.save_freq < TIMESTEPS:
            model.save(os.path.join(model_path, f"ppo_model_{model.num_timesteps}.zip"))

    # Save final model
    model.save(os.path.join(model_path, "final_model"))

    train_env.close()
    eval_env.close()
