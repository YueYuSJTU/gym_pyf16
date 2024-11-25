import gymnasium as gym
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make('gym_pyf16_env/GridWorld-v0')

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
model = PPO.load("./logs/best_model", env=env, device='cpu')

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()

position = []
waypoints = []
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, terminated, _ = env.step(action)
    position.append([obs[0][0], obs[0][1], obs[0][2]])
    waypoints.append(obs[0][-3:])
    if terminated:
        print(f"Terminated at step {i}")
        break

# 绘制位置关于时间的三维图像
position.pop()              # 包装后的环境默认terminate后会reset，所以最后一个点是初值
position = list(zip(*position))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
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
plt.show()


env.close()