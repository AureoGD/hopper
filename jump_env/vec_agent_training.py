import os
import time
import multiprocessing as mp
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import jumper_env


# ──────────────────────────────────────────────────────────────
# 🔹 Callback to force TensorBoard logging after each rollout
class ForceTensorboardLogging(BaseCallback):
    def _on_rollout_end(self) -> None:
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True  # Required for compatibility


# ──────────────────────────────────────────────────────────────
def make_env(rank):
    def _init():
        env = jumper_env.JumperEnv(render=False, render_every=False, log_interval=10, render_interval=10)
        return env

    return _init


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    # Setup directories
    timestamp = int(time.time())
    models_dir = f"models/{timestamp}/"
    logdir = f"logs/{timestamp}/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Parallel environments
    NUM_ENVS = 16
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    train_env = VecMonitor(train_env, filename=os.path.join(logdir, "monitor.csv"))

    # Evaluation environment
    eval_env = jumper_env.JumperEnv(render=False)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_env = VecMonitor(eval_env)

    # PPO model with faster logging updates
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(
            net_arch=[
                dict(
                    pi=[128, 64, 32],  # Policy network
                    vf=[128, 64, 32],  # Value network
                )
            ],
            activation_fn=nn.ReLU,
        ),
        verbose=1,
        tensorboard_log=logdir,
        n_steps=512,  # Smaller rollout to flush logs more frequently
    )

    callback = ForceTensorboardLogging()

    # Training loop
    TIMESTEPS = 2500
    iters = 0
    best_reward = float("-inf")
    best_model_path = os.path.join(models_dir, "best_model.zip")

    while iters < 500:
        iters += 1
        print(f"Iteration: {iters}")

        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name="PPO",
            callback=callback,
            log_interval=1,  # <--- ✅ This is required for logging loss, entropy, etc.
        )

        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1)

        model.logger.record("custom/debug_reward", mean_reward)
        model.logger.dump(model.num_timesteps)

        if mean_reward > best_reward:
            best_reward = mean_reward
            model.save(best_model_path)
            print(f"New best model saved at {best_model_path} with reward: {best_reward}")

        model.save(f"{models_dir}/ppo_model_{TIMESTEPS * iters}.zip")

    train_env.close()
    eval_env.close()
