import gymnasium as gym
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pyf16_env.wrappers import SkipObsWrapper
import numpy as np

# 模型存储位置
log_path="./logs/"
# 读取模型选择
# model_name = "best_model"
# model_name = "final_model"
model_name = "best_model_1736.615260656178"

skipStep = 5
skipTimes = 3

# 设置随机种子
seed = 42
np.random.seed(seed)
gym.utils.seeding.np_random(seed)

# Create environment
env_id = "gym_pyf16_env/GridWorld-v0"
# vec_env = DummyVecEnv([lambda: gym.make(env_id)])
vec_env = DummyVecEnv([lambda: SkipObsWrapper(gym.make(env_id), skip_step=skipStep, skip_times=skipTimes)])
vec_env.seed(seed)
vec_env = VecNormalize.load(log_path + "best_train_env", vec_env)
vec_env.training = False
vec_env.norm_reward = False

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load(log_path + model_name, env=vec_env, device='cpu')

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# vec_env = model.get_env()
obs = vec_env.reset()

position = []
Euler = []
rewardDepart = []
# waypoints = []
actions = []
print("Start to simulate")
for i in range(15000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, _ = vec_env.step(action)
    rewardDepart.append([vec_env.get_attr('alpha'), vec_env.get_attr('height'), vec_env.get_attr('beta'), vec_env.get_attr('action_penalty')])
    unNom_obs = vec_env.unnormalize_obs(obs[0])  # 取消归一化
    # print(f"Step {i}: {unNom_obs[0:6]}")
    position.append([unNom_obs[0][0], unNom_obs[0][1], unNom_obs[0][2]])
    Euler.append([unNom_obs[0][3], unNom_obs[0][4], unNom_obs[0][5]])
    # waypoints.append(unNom_obs[0][-3:])
    actions.append(action[0])
    if terminated:
        print(f"Terminated at step {i}")
        break

# 绘制位置关于时间的三维图像
position.pop()              # 包装后的环境默认terminate后会reset，所以最后一个点是初值
Euler.pop()
rewardDepart.pop()
position = list(zip(*position))
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(141, projection='3d')
ax.plot(position[0], position[1], position[2])

# 区分起始点和终点
ax.scatter(position[0][0], position[1][0], position[2][0], color='red', s=100, label='Start')
ax.scatter(position[0][-1], position[1][-1], position[2][-1], color='blue', s=100, label='End')

# 绘制路径点
# waypoints = list(map(list, set(map(tuple, waypoints))))
# print(f"Waypoints: {waypoints}")
# waypoints = list(zip(*waypoints))
waypoints = vec_env.get_attr('waypoints')[0]
waypoints = list(zip(*waypoints))
print(f"Waypoints: {waypoints}")
ax.scatter(waypoints[0], waypoints[1], waypoints[2], color='green', s=100, label='Waypoints')

ax.set_zlim(0, 20000)
ax.set_xlabel('North Position')
ax.set_ylabel('East Position')
ax.set_zlabel('Altitude')
ax.legend()

# 绘制欧拉角关于时间的图像
Euler = list(zip(*Euler))
Roll = fig.add_subplot(442)
Roll.plot(Euler[0], label='Roll')
Roll.legend()
Pitch = fig.add_subplot(443)
Pitch.plot(Euler[1], label='Pitch')
Pitch.legend()
Yaw = fig.add_subplot(444)
Yaw.plot(Euler[2], label='Yaw')
Yaw.legend()

actions = list(zip(*actions))
Thrust = fig.add_subplot(446)
Thrust.plot(actions[0], label='Thrust')
Thrust.legend()
Elevator = fig.add_subplot(447)
Elevator.plot(actions[1], label='Elevator')
Elevator.legend()
Aileron = fig.add_subplot(448)
Aileron.plot(actions[2], label='Aileron')
Aileron.legend()
Rudder = fig.add_subplot(4,4,10)
Rudder.plot(actions[3], label='Rudder')
Rudder.legend()

# 绘制奖励函数
rewardDepart = list(zip(*rewardDepart))
alpha = fig.add_subplot(4,4,14)
alpha.plot(rewardDepart[0], label='alpha')
alpha.legend()
height = fig.add_subplot(4,4,15)
height.plot(rewardDepart[1], label='height')
height.legend()
beta = fig.add_subplot(4,4,16)
beta.plot(rewardDepart[2], label='beta')
beta.legend()
action_penalty = fig.add_subplot(4,4,12)
action_penalty.plot(rewardDepart[3], label='Action Penalty')
action_penalty.legend()

plt.show()

vec_env.close()