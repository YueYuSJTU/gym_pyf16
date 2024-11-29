import gymnasium as gym
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pyf16_env.wrappers import SkipObsWrapper

# 模型存储位置
log_path="./logs/"
# 读取模型选择
model_name = "best_model"

# Create environment
env_id = "gym_pyf16_env/GridWorld-v0"
# vec_env = DummyVecEnv([lambda: gym.make(env_id)])
vec_env = DummyVecEnv([lambda: SkipObsWrapper(gym.make(env_id), skip_step=1, skip_times=4)])
vec_env = VecNormalize.load(log_path + "final_train_env", vec_env)
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
waypoints = []
actions = []
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, _ = vec_env.step(action)
    unNom_obs = vec_env.unnormalize_obs(obs)  # 取消归一化
    print(f"Step {i}: {unNom_obs[0:6]}")
    position.append([unNom_obs[0][0], unNom_obs[0][1], unNom_obs[0][2]])
    waypoints.append(unNom_obs[0][-3:])
    actions.append(action[0])
    if terminated:
        print(f"Terminated at step {i}")
        break

# 绘制位置关于时间的三维图像
position.pop()              # 包装后的环境默认terminate后会reset，所以最后一个点是初值
position = list(zip(*position))
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(121, projection='3d')
ax.plot(position[0], position[1], position[2])

# 区分起始点和终点
ax.scatter(position[0][0], position[1][0], position[2][0], color='red', s=100, label='Start')
ax.scatter(position[0][-1], position[1][-1], position[2][-1], color='blue', s=100, label='End')

# 绘制路径点
waypoints = list(map(list, set(map(tuple, waypoints))))
print(f"Waypoints: {waypoints}")
waypoints = list(zip(*waypoints))
ax.scatter(waypoints[0], waypoints[1], waypoints[2], color='green', s=100, label='Waypoints')

ax.set_zlim(0, 20000)
ax.set_xlabel('North Position')
ax.set_ylabel('East Position')
ax.set_zlabel('Altitude')
ax.legend()

actions = list(zip(*actions))
Thrust = fig.add_subplot(422)
Thrust.plot(actions[0], label='Thrust')
Thrust.legend()
Elevator = fig.add_subplot(424)
Elevator.plot(actions[1], label='Elevator')
Elevator.legend()
Aileron = fig.add_subplot(426)
Aileron.plot(actions[2], label='Aileron')
Aileron.legend()
Rudder = fig.add_subplot(428)
Rudder.plot(actions[3], label='Rudder')
Rudder.legend()


plt.show()


vec_env.close()