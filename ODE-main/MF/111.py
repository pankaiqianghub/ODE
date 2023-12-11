import random

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(0)

# 生成正态分布的数据
samples = 5000
mean = 2.5
std_dev = 18
data = np.random.normal(mean, std_dev, samples)
data = data[(data > -100) & (data < 100)]  # 保留(-50,50)范围内的数据
for i in range(len(data)):
    ran = random.uniform(-10, 5)
    data[i] = data[i]+ran
    print(ran)


# 创建直方图的bin
n_bins = 100
counts, bin_edges = np.histogram(data, bins=n_bins, range=(-50, 50))

# 根据直方图的频数调整bin的宽度以保持面积恒定
constant_area = len(data) / n_bins
bin_widths = constant_area / counts
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# 绘制柱状图
plt.figure(figsize=(10, 6))
for center, width in zip(bin_centers, bin_widths):
    plt.bar(center, constant_area / width, width=width, alpha=0.7, color='Orchid', edgecolor='Orchid')

# 假设的100维数组，您可以根据需要替换为自己的数据
line_data = np.random.rand(100) * 2000 + 2000  # 这是一个随机生成的示例数据



plt.title('Normal Distribution with Equal Area Bars and a Line Plot Overlay')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
