import gymnasium as gym
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pyf16_env.wrappers import SkipObsWrapper
import numpy as np
from Best3Eval import BestThreeEvalCallback

env_id = "gym_pyf16_env/GridWorld-v0"

skipStep = 5
skipTimes = 3

# 设置随机种子
seed = 42
np.random.seed(seed)
gym.utils.seeding.np_random(seed)

# 创建训练环境
train_env = DummyVecEnv([lambda: SkipObsWrapper(gym.make(env_id), skip_step=skipStep, skip_times=skipTimes)])
train_env.seed(seed)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)
# 创建评估环境
eval_env = DummyVecEnv([lambda: SkipObsWrapper(gym.make(env_id), skip_step=skipStep, skip_times=skipTimes)])
eval_env.seed(seed)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)
eval_env.training = False
eval_env.norm_reward = False

# 模型存储位置
log_path="./logs/"

# Use deterministic actions for evaluation
# eval_callback = EvalCallback(eval_env, best_model_save_path=log_path,
#                              log_path=log_path, eval_freq=10000,
#                              deterministic=True, render=False)
eval_callback = BestThreeEvalCallback(eval_env, best_model_save_path=log_path,
                                      log_path=log_path, eval_freq=20000,
                                      deterministic=True, render=False)

model = PPO("MlpPolicy", train_env, verbose=1, device='cpu', tensorboard_log="./logs/tenorboard/")
model.learn(total_timesteps=15000_000, progress_bar=True, callback=[eval_callback])

# 保存训练结束的模型
model.save(log_path + "final_model")
train_env.save(log_path + "final_train_env")
