import gymnasium as gym
from stable_baselines3 import PPO
import time
import jumper_env

env = jumper_env.JumperEnv(render=True, render_every=True, log_interval=10)

model_path = "models/1744317406/best_model.zip"  # Replace with your model's file path
# model_path = "models/1744314563/best_model.zip"  # Replace with your model's file path
model = PPO.load(model_path, env=env)  # Load the model

obs, info = env.reset()  # Reset the environment at the beginning of each episode
done = False
sucess = 0
interactions = 0
for i in range(5):
    while not done:
        action, _ = model.predict(obs)  # Get the model's action
        print(action)
        obs, reward, done, truncated, info = env.step(action)
        interactions += 1
    if interactions > 1250:
        sucess += 1
    done = False
    interactions = 0
    obs, info = env.reset()
print(sucess)
