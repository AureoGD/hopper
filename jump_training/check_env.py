from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from icecream import ic
import numpy as np

from jump_env.jumper_env import JumperEnv

env = JumperEnv(render=True, render_every=True, render_interval=10, log_interval=10)
# check_env(env)

episodes = 1
episode_rewards = []

for episode in range(episodes):
    done = False
    obs, _ = env.reset()
    ic(episode)
    total_reward = 0
    mode = 7
    inter = 0
    while not done:
        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(np.array([mode]))
        total_reward += reward

        inter += 1
        if inter > 25 and inter < 500:
            mode = 1
        if inter > 600:
            mode = 2
        # if inter >= 900 and inter < 1000:
        #     mode = 4
        # if inter >= 1000:
        #     mode = 7
        episode_rewards.append(total_reward)

# ────────────────
# Plotting reward
# ────────────────
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, marker="o")
plt.title("Episode Reward over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()
