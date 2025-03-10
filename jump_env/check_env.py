from stable_baselines3.common.env_checker import check_env
from icecream import ic

import jumper_env

env = jumper_env.JumperEnv(render=True, render_every=True, render_interval=10, log_interval=10)
# check_env(env)

episodes = 20

for episode in range(episodes):
    done = False
    obs, _ = env.reset()
    ic(episode)
    while not done:
        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(1)
