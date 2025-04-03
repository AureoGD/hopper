from stable_baselines3 import PPO, A2C
import os
import time
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import jumper_env

# Setup directories
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Hyperparameters
TIMESTEPS = 2500  # Train for this many steps before evaluating
iters = 0
best_reward = float("-inf")  # Track the highest reward
best_model_path = os.path.join(models_dir, "best_model.zip")  # Always overwrite this file

# Initialize environment and model
env = jumper_env.JumperEnv(render=True, render_every=False, log_interval=10, render_interval=10)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

#model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
#model = PPO.load("models/1742240835/best_model.zip", env, verbose=1, tensorboard_log=logdir)

while iters < 500:
    iters += 1
    print(f"Iteration: {iters}")

    # Train for TIMESTEPS steps
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

    # Evaluate after every episode
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)  # Eval after every episode

    # Check if the new model is better
    if mean_reward > best_reward:
        best_reward = mean_reward
        model.save(best_model_path)  # Overwrite the previous best model
        print(f"🔥 New best model saved at {best_model_path} with reward: {best_reward}")

    # Save periodic models (optional)
    model.save(f"{models_dir}/ppo_model_{TIMESTEPS * iters}.zip")

env.close()
