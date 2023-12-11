import pdb

import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可复现
np.random.seed(0)

# 生成正态分布的数据
samples = 10000
mean = 0
std_dev = 15
data = np.random.normal(mean, std_dev, samples)
data = data[(data > -50) & (data < 50)]  # 保留(-50,50)范围内的数据

# 创建直方图的bin
n_bins = 50
counts, bin_edges = np.histogram(data, bins=n_bins, range=(-50, 50))

# 根据直方图的频数调整bin的宽度以保持面积恒定
constant_area = len(data) / n_bins
bin_widths = constant_area / counts
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# 绘制柱状图
plt.figure(figsize=(10, 6))
for center, width in zip(bin_centers, bin_widths):
    plt.bar(center, constant_area / width, width=width, alpha=0.7, color='blue', edgecolor='blue')

# 假设的100维数组，您可以根据需要替换为自己的数据
line_data = np.random.rand(50) * 2000 + 2000  # 这是一个随机生成的示例数据
value = [1.01, 0.9891, 0.9811, 0.9873, 0.9844, 0.9814, 0.98, 0.9778, 0.9761, 0.9748, 0.9733,
         0.9715, 0.9705, 0.97, 0.9682608695652174, 0.9665217391304347, 0.9647826086956521, 0.9630434782608696, 0.961304347826087, 0.9595652173913043, 0.9578260869565217, 0.9560869565217391, 0.9543478260869566, 0.9526086956521739, 0.9508695652173913, 0.9491304347826087, 0.9473913043478261, 0.9456521739130435, 0.9439130434782609, 0.9421739130434783, 0.9404347826086957, 0.938695652173913, 0.9369565217391305, 0.9352173913043479, 0.9334782608695653, 0.9317391304347826, 0.93, 0.93, 0.9293055555555556, 0.9272222222222223, 0.9237500000000001, 0.918888888888889, 0.9126388888888889, 0.905, 0.8959722222222223, 0.8855555555555555, 0.87375, 0.8605555555555555, 0.8459722222222222, 0.83]
# 绘制折线图
value = np.array(value)
for i in range(len(value)):
    value[i] = (value[i] - 0.8)*2000
plt.plot(bin_centers, value, color='red', label='Custom Line Data')

plt.title('Normal Distribution with Equal Area Bars and a Line Plot Overlay')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
