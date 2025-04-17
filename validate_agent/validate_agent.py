import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from jump_env.jumper_env import JumperEnv

from cnn_feature.observation_stack_wrapper import ObservationStackWrapper

# ───────────────────────────────────────────────────────────────
# Load environment and model
# ───────────────────────────────────────────────────────────────
# env = JumperEnv(render=True, render_every=True, log_interval=10)
# model_path = "agents/DNN/1744733462/models/best_model.zip"
# model_path = "agents/DNN/1744733462/models/ppo_model_917504.zip"

model_path = "agents/CNN/1744909023/models/best_model.zip"
env = JumperEnv(render=True, render_every=True, log_interval=10)
env = ObservationStackWrapper(env, stack_size=15)

model = PPO.load(model_path, env=env)

# ───────────────────────────────────────────────────────────────
# Run simulation
# ───────────────────────────────────────────────────────────────
n_episodes = 5
actions_all_episodes = []
rewards_all_episodes = []

success = 0
for i in range(n_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    interactions = 0

    actions = []
    rewards = []
    ep_reward = 0
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        # action = np.array([0])
        obs, reward, terminated, truncated, info = env.step(action)
        interactions += 1
        actions.append(int(action))
        ep_reward += reward
        rewards.append(ep_reward)

    actions_all_episodes.append(actions)
    rewards_all_episodes.append(rewards)

    if reward > 100:
        success += 1

print(f"Success count: {success}/{n_episodes}")

# ───────────────────────────────────────────────────────────────
# Plot: actions and rewards per episode
# ───────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 6))

# Actions
plt.subplot(1, 2, 1)
for idx, actions in enumerate(actions_all_episodes):
    plt.plot(range(len(actions)), actions, label=f"Episode {idx + 1}")
plt.title("Actions per Step")
plt.xlabel("Step")
plt.ylabel("Action")
plt.legend()

# Rewards
plt.subplot(1, 2, 2)
for idx, rewards in enumerate(rewards_all_episodes):
    plt.plot(range(len(rewards)), rewards, label=f"Episode {idx + 1}")
plt.title("Reward per Step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()

plt.tight_layout()
plt.show()
