import gymnasium
import gym_pyf16_env
from gym_pyf16_env.wrappers import SkipObsWrapper
import matplotlib.pyplot as plt
import numpy as np
import time
startTime = time.time()

initEnv = gymnasium.make('gym_pyf16_env/GridWorld-v0')
wrapped_env = SkipObsWrapper(initEnv, skip_step=1, skip_times=4)
print(f"wrapped observation space: {wrapped_env.observation_space}")

init_obs, _ = wrapped_env.reset()
print(f"init_obs: {init_obs}")
print(f"if space contain init_obs: {wrapped_env.observation_space.contains(init_obs)}")

# action = np.array([2109.4, -2.2441, -0.0936, 0.0945])
action = np.array([-0.5, -0.05, 0, 0])

position = []
waypoints = []
for i in range(3500):
    _, reward, terminated, _, obs = wrapped_env.step(action)
    position.append([obs[0], obs[1], obs[2]])
    waypoints.append(obs[-3:])
    if terminated:
        wrapped_env.reset()
        print(f"Terminated at step {i}")
        break
    # print(f"Step:{i}, position: {obs[0]}, {obs[1]}, {obs[2]}, Done: {terminated}")

endTime = time.time()
print(f"Time: {endTime - startTime}")



# 绘制位置关于时间的三维图像
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


wrapped_env.close()