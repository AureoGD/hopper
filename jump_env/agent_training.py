from stable_baselines3 import PPO, A2C
import os
import time
import jumper_env
from stable_baselines3.common.env_util import make_vec_env

env = jumper_env.JumperEnv(render=True, render_every=False, log_interval=10, render_interval=10)

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# env.reset()

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
# model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 2500
iters = 0
while iters < 500:
    iters += 1
    print("Inter: ", iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
