import gymnasium as gym
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


env = gym.make('gym_pyf16_env/GridWorld-v0')

model = PPO("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=200_000)

env = model.get_env()
obs = env.reset()

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