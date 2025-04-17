import os
import time
from typing import List
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from jump_env.jumper_env import JumperEnv


# ──────────────────────────────────────────────────────────────
# 🔹 Callback to force TensorBoard logging
class ForceTensorboardLogging(BaseCallback):
    def _on_rollout_end(self) -> None:
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True


# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
def train(
    policy: str = "DNN",
    stack_size: int = 4,
    hidden_layers: List[int] = [64, 64],
    total_interations: int = 500,
    num_envs: int = 8,
    eval_freq: int = 1000,
    save_freq: int = 1000,
):
    timestamp = int(time.time())
    base_dir = f"agents/{policy}/{timestamp}"
    model_path = os.path.join(base_dir, "models")
    log_path = os.path.join("training_logs", policy, str(timestamp))
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(base_dir, "config.txt"), "w") as f:
        f.write(f"Policy: {policy}\nStack Size: {stack_size}\nHidden Layers: {hidden_layers}\n")

    if policy == "CNN":
        from cnn_feature.custom_policy import CNNPPOPolicy

        policy_class = CNNPPOPolicy
        policy_kwargs = {
            "features_extractor_kwargs": {
                "stack_size": stack_size,
                "output_dim": 128,
            },
            "net_arch": hidden_layers,
        }
    else:
        policy_class = "MlpPolicy"
        policy_kwargs = {
            "net_arch": [dict(pi=hidden_layers, vf=hidden_layers)],
            "activation_fn": nn.ReLU,
        }

    # Create environments
    train_env = SubprocVecEnv([make_env(i, policy, stack_size) for i in range(num_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    train_env = VecMonitor(train_env, filename=os.path.join(log_path, "monitor.csv"))

    eval_env = DummyVecEnv([make_env("eval", policy, stack_size)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_env = VecMonitor(eval_env)

    # Instantiate PPO
    model = PPO(
        policy_class,
        train_env,
        verbose=1,
        tensorboard_log=log_path,
        n_steps=512,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        device="auto",
    )

    callback = ForceTensorboardLogging()
    TIMESTEPS = 2048
    inters = 0
    best_peak_reward = float("-inf")
    peak_index = 0
    while inters < total_interations:
        inters += 1
        print(f"\n Iteration {inters} | Timesteps: {model.num_timesteps}")

        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO",
            callback=callback,
            log_interval=1,
        )

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=3)
        model.logger.record("iteration", inters)
        model.logger.dump(model.num_timesteps)

        # Save if a new peak is found
        if mean_reward > best_peak_reward:
            peak_index += 1
            best_peak_reward = mean_reward

            peak_path = os.path.join(model_path, f"best_model_{peak_index}.zip")
            model.save(peak_path)

            model.save(os.path.join(model_path, "best_model.zip"))  # Latest best overall
            print(f"New peak #{peak_index}: Reward = {mean_reward:.2f}")
        else:
            print(f"Reward: {mean_reward:.2f} (Best so far: {best_peak_reward:.2f})")

        # Save periodic checkpoint
        if model.num_timesteps % save_freq < TIMESTEPS:
            model.save(os.path.join(model_path, f"ppo_model_{model.num_timesteps}.zip"))
            print(f"💾 Saved checkpoint at {model.num_timesteps}")

    # Final model
    model.save(os.path.join(model_path, "final_model.zip"))
    train_env.close()
    eval_env.close()
