import gym
from stable_baselines3 import PPO

models_dir = "models/PPO"
model_path = f"{models_dir}/2600000.zip"
env = gym.make('LunarLander-v2')

model = PPO.load(model_path, env = env)

episodes = 5
for episode in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)
        