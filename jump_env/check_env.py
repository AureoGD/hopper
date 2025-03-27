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
        if inter > 50 and inter < 300:
            mode = 0
        # if inter > 300 and inter < 550:
        #     mode = 4
        # elif inter > 550 and inter < 800:
        #     mode = 6
        # elif inter > 800 and inter < 1400:
        #     mode = 5
        # elif inter > 1400:
        #     mode = 2
