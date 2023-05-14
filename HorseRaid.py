import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个3D数组
data = np.random.randint(low=0, high=10, size=(10, 10, 10))

# 创建一个3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 将数组的每个点绘制为散点图
x, y, z = data.nonzero()  # 非零值的位置
ax.scatter(x, y, z, c=data[x, y, z], marker='o', cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 显示图像
plt.show()
