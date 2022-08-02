import os
import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make('LunarLander-v2')
env.reset()

model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log = logdir)

timesteps = 100000
for i in range(30):γ
    model.learn(total_timesteps = timesteps, reset_num_timesteps = False, tb_log_name = "PPO")
    model.save(f"{models_dir} / {timesteps * i}")
