import gymnasium
import gym_pyf16_env
from gymnasium.wrappers import FlattenObservation
import matplotlib.pyplot as plt


env = gymnasium.make('gym_pyf16_env/GridWorld-v0')

init_obs, _ = env.reset()

action = {"thrust": 2109.4, "elevator": -2.2441, "aileron": -0.0936, "rudder": 0.0945}

position = []
for i in range(3000):
    obs, reward, terminated, _, _ = env.step(action)
    position.append([obs['npos'], obs['epos'], obs['altitude']])
    print(f"Step:{i}, position: {obs['npos']}, {obs['epos']}, {obs['altitude']}, Done: {terminated}")

# 绘制位置关于时间的三维图像
position = list(zip(*position))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(position[0], position[1], position[2])

# 区分起始点和终点
ax.scatter(position[0][0], position[1][0], position[2][0], color='red', s=100, label='Start')
ax.scatter(position[0][-1], position[1][-1], position[2][-1], color='blue', s=100, label='End')

ax.set_zlim(0, 20000)
ax.set_xlabel('North Position')
ax.set_ylabel('East Position')
ax.set_zlabel('Altitude')
ax.legend()
plt.show()


env.close()