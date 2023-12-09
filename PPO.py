import gymnasium as gym
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from Tennis_Env import TennisEnv

env = make_vec_env(TennisEnv, n_envs=1)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=9e-3)
model.learn(total_timesteps=1_000_000)
model.save("ppo_tennis_1_000_9e3")

env.close()