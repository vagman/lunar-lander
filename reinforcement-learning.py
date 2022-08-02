import os
import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"
logdir = "logs"

# if not os.path.exists(models_dir):
#     os.makedirs(models_dir)

# if not os.path.exists(logdir):
#     os.makedirs(logdir)

env = gym.make('LunarLander-v2')
# env.reset()

# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 100000
# iters = 0
# for i in range(30):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")

model_path = f"{models_dir}/2600000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)
