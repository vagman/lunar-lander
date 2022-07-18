# 1. Import libraries
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# 2. Test Environment
environmet_name = 'CarRacing-v0'
env = gym.make(environmet_name)

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))


# 3. Train Model
env = gym.make(environmet_name)
env = DummyVecEnv([lambda: env])
log_path = os.path.join('Training', 'Logs')
model = PPO('CnnPolicy', env, verbose = 1, tensorboard_log = log_path)

model.learn(total_timesteps = 1000000)

# 4. Save Model
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_Model')
model.save(ppo_path)
model = PPO.load(ppo_path, env)

# 5. Evaluate and Test
evaluate_policy(model, env, n_eval_episodes = 10, render = True)
env.close()
