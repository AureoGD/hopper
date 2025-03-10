import gymnasium as gym
from stable_baselines3 import PPO
import time
import jumper_env

env = jumper_env.JumperEnv(render=True, render_every=True, log_interval=10)

model_path = "models/1740662522/72500.zip"  # Replace with your model's file path
model = PPO.load(model_path, env=env)  # Load the model

obs, info = env.reset()  # Reset the environment at the beginning of each episode

while True:
    action, _ = model.predict(obs)  # Get the model's action
    obs, reward, done, truncated, info = env.step(action)
