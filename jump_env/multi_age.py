from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time
import jumper_env


# Create multiple training environments
def make_env(rank):
    def _init():
        return jumper_env.JumperEnv(render=False, render_every=False, log_interval=10, render_interval=10)
    return _init


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)

    # Setup directories
    timestamp = int(time.time())
    models_dir = f"models/{timestamp}/"
    logdir = f"logs/{timestamp}/"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # Create vectorized environment with 4 parallel instances
    num_envs = 4
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Create separate evaluation environment (headless)
    eval_env = jumper_env.JumperEnv(render=False)

    # Initialize PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=logdir)

    # Training loop parameters
    TIMESTEPS = 2500
    iters = 0
    best_reward = float("-inf")
    best_model_path = os.path.join(models_dir, "best_model.zip")

    while iters < 500:
        iters += 1
        print(f"Iteration: {iters}")
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")

        # Evaluate performance on a single environment
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=1)

        # Save best model
        if mean_reward > best_reward:
            best_reward = mean_reward
            model.save(best_model_path)
            print(f"🔥 New best model saved at {best_model_path} with reward: {best_reward}")

        # Optional: Save model checkpoint
        model.save(f"{models_dir}/ppo_model_{TIMESTEPS * iters}.zip")

    vec_env.close()
    eval_env.close()
