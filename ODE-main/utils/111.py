import numpy as np
import matplotlib.pyplot as plt
data = [36.35, 30.03, 25.33, 20.08, 17.85, 15.53, 16.36, 19.87, 19.33, 21.34]
x_labels = ["0.00001%", "0.001%", "0.001%", "0.01%", "0.1%", "1%", "2%", "3%", "4%", "5%"]
# 基础数据点生成函数
def generate_data(mean, cov, num_samples):
    return np.random.multivariate_normal(mean, cov, num_samples)

# 初始化
np.random.seed(0)
mean_A = [2, 2]
cov_A = [[0.1, 0], [0, 0.1]]
mean_B = [2, 2]
cov_B = [[0.1, 0], [0, 0.1]]

samples_per_group = 10
num_groups = 10

A_points = generate_data(mean_A, cov_A, 1000)
B_points = generate_data(mean_B, cov_B, 1000)

plt.scatter(A_points[:, 0], A_points[:, 1], c='yellow', label='ODE initial')
plt.scatter(B_points[:, 0], B_points[:, 1], c='blue', label='APT initial')

colors_A = plt.cm.YlOrRd(np.linspace(0.5, 1, num_groups))
colors_B = plt.cm.Blues(np.linspace(0.5, 1, num_groups))

for i in range(1, num_groups + 1):
    # 每次迭代，我们稍微移动均值
    new_A = generate_data([mean_A[0] + i*0.1, mean_A[1] + i*0.1], cov_A, samples_per_group)
    new_B = generate_data([mean_B[0] - i*0.001, mean_B[1] - i*0.001], cov_B, samples_per_group)
    plt.scatter(new_A[:, 0], new_A[:, 1], c=colors_A[i-1], label=f'ODE extension' if i == 1 else "")
    plt.scatter(new_B[:, 0], new_B[:, 1], c=colors_B[i-1], label=f'APT extension' if i == 1 else "")

plt.title('Distribution of A and B with added points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()
