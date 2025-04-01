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
    mode = 4
    inter = 0
    while not done:
        # action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(mode)
        inter += 1
        if inter > 50 and inter < 500:
            mode = 1
        if inter > 500 and inter < 900:
            mode = 5
        if inter >= 900 and inter < 1000:
            mode = 7
        if inter >= 1000:
            mode = 2
