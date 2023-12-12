import gymnasium as gym

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Tennis_Env import TennisEnv

env = make_vec_env(TennisEnv, n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3)
model.learn(total_timesteps=300_000)
model.save("ppo_tennis_300_p_r0")

for i in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(f"Action: {action}, Reward: {rewards}, Done: {done}, Info: {info}")
        print(f"Obs: {obs}")

env.close()