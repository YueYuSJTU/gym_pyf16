import gymnasium as gym
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


env_id = "gym_pyf16_env/GridWorld-v0"

# 创建训练环境
train_env = DummyVecEnv([lambda: gym.make(env_id)])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
# 创建评估环境
eval_env = DummyVecEnv([lambda: gym.make(env_id)])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
eval_env.training = False
eval_env.norm_reward = False

# 模型存储位置
log_path="./logs/"

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
                             log_path=log_path, eval_freq=5000,
                             deterministic=True, render=False)

model = PPO("MlpPolicy", train_env, verbose=1, device='cpu')
model.learn(total_timesteps=250_000, progress_bar=True, callback=[eval_callback])

# 保存训练结束的模型
# model.save(log_path + "final_model")
train_env.save(log_path + "final_train_env")
