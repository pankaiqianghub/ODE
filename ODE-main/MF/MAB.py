import numpy as np
import matplotlib.pyplot as plt
import math

## 初始化基础数据
# 老虎机数量
number_of_bandits = 5
# 每个老虎机的摇臂数
number_of_arms = 10
# 拉动试验次数
number_of_pulls = 1000000

## 初始化算法参数
# 初始epsilon
epsilon = 0.2
# 最小decay
min_temp = 0.1
# 衰减率
decay_rate = 0.999


def pick_arm(q_values, counts, strategy, success, failure):
    # 参数：中奖概率，被选作最佳摇臂次数，选择策略（算法），成功（中奖）次数，失败（未中奖）次数
    global epsilon
    # opt1，完全随机选取摇臂（随机乱猜）
    if strategy == "random":
        return np.random.randint(0, len(q_values))

    # opt2，贪心算法，选择截止当前，平均收益最大的那个摇臂。
    if strategy == "greedy":
        # 最大试验概率值
        best_arms_value = np.max(q_values)
        # 最大概率对应的摇臂
        best_arms = np.argwhere(q_values == best_arms_value).flatten()
        # 若有多个最大概率相同的摇臂，随机从中选取一个
        return best_arms[np.random.randint(0, len(best_arms))]

    # opt3 & opt4，在贪心法的基础上，加epsilon结合上面两种算法（egreedy为固定epsilon值，egreedy_decay为衰减epsilon值）
    if strategy == "egreedy" or strategy == "egreedy_decay":
        if strategy == "egreedy_decay":
            epsilon = max(epsilon * decay_rate, min_temp)

        # 每次随机生成一个0~1的概率值，选择截止当前，平均收益最大的那个摇臂；否则从所有臂中随机选一个。
        if np.random.random() > epsilon:
            best_arms_value = np.max(q_values)
            best_arms = np.argwhere(q_values == best_arms_value).flatten()
            return best_arms[np.random.randint(0, len(best_arms))]
        else:
            return np.random.randint(0, len(q_values))

    # opt5，UCB算法，选择UCB值最大的那个摇臂
    if strategy == "ucb":
        total_counts = np.sum(counts)
        # ucb公式
        q_values_ucb = q_values + np.sqrt(
            np.reciprocal(counts + 0.001) * 2 * math.log(total_counts + 1.0))  # np.reciprocal：1/x
        best_arms_value = np.max(q_values_ucb)
        best_arms = np.argwhere(q_values_ucb == best_arms_value).flatten()
        return best_arms[np.random.randint(0, len(best_arms))]

    # opt6,Thompson采样算法
    # 假设每个臂是否产生收益，其背后有一个概率分布，产生收益的概率为p。
    # 我们不断地试验，去估计出一个置信度较高的‘概率p的概率分布’就能近似解决这个问题了。
    # 怎么能估计概率p的概率分布呢？ 答案是假设概率p的概率分布符合beta(success, failure)分布，它有两个参数: success, failure。
    # 每个臂都维护一个beta分布的参数。每次试验后，选中一个臂，摇一下，有收益则该臂的success增加1，否则该臂的failure增加1。
    # 每次选择臂的方式是：用每个臂现有的beta分布产生一个随机数b，选择所有臂产生的随机数中最大的那个臂作为最佳摇臂。
    if strategy == "thompson":
        sample_means = np.zeros(len(counts))
        for i in range(len(counts)):
            sample_means[i] = np.random.beta(success[i] + 1, failure[i] + 1)
        return np.argmax(sample_means)


## 画图对比六种算法效果
fig = plt.figure()
ax = fig.add_subplot(111)

for st in ["egreedy", "ucb", "thompson"]:
    # 初始化每个老虎机每次拉动试验后的最优摇臂命中率矩阵
    # 行：第i个老虎机  列：第j次拉动摇臂  value：对第i个老虎机第i次试验后最佳摇臂的命中率
    print(st)
    best_arm_counts = np.zeros((number_of_bandits, number_of_pulls))
    # 循环每个老虎机
    for i in range(number_of_bandits):
        # 随机生成该老虎机每个摇臂的期望中奖概率（真实概率）
        # arm_means = 0.01*np.random.rand(number_of_arms)
        arm_means = 0.01 + 0.02 * np.random.rand(number_of_arms)
        mean = np.mean(arm_means)
        arm_means[7] = mean - 0.001 * np.random.rand()
        arm_means[8] = mean - 0.001 * np.random.rand()
        arm_means[9] = mean - 0.001 * np.random.rand()
        print("arm_means:", arm_means)
        # arm_means=np.array([0.01,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])
        # 获取收益最大摇臂的索引位置（得到真实最大收益摇臂）
        # print("arm_means:",arm_means)
        best_arm = np.argmax(arm_means)
        print("best_arm:", best_arm)

        # 当前老虎机各个摇臂试验数据初始化（1 X number_of_arms全零矩阵）：
        # 中奖概率，被选作最佳摇臂次数，成功（中奖）次数，失败（未中奖）次数
        # q_values = np.zeros(number_of_arms) # 中奖概率
        q_values = arm_means
        q_values[7] = 0
        q_values[8] = 0
        q_values[9] = 0

        counts = 1000000 * np.ones(number_of_arms)
        counts[7] = 0
        counts[8] = 0
        counts[9] = 0

        # success = np.zeros(number_of_arms)
        success = (counts * q_values).astype(int)
        # failure = np.zeros(number_of_arms)
        failure = counts - success

        # 进行number_of_pulls次拉动试验
        for j in range(number_of_pulls):
            # 使用当前算法st计算出能够获得最佳收益的摇臂（通过计算认为的最佳收益摇臂）
            a = pick_arm(q_values, counts, st, success, failure)

            # 进行一次伯努利试验模拟最佳臂a是否能够中奖（1为中奖，0为未中奖）
            reward = np.random.binomial(1, arm_means[a])
            # 记录并更新被选作最佳摇臂的次数
            counts[a] += 1.0
            # 对所选最佳摇臂a计算更新试验中奖概率（试验概率）
            q_values[a] += (reward - q_values[a]) / counts[a]
            # 记录中奖次数
            success[a] += reward
            # 记录未中奖次数
            failure[a] += (1 - reward)
            # 更新第i个老虎机第j次试验后计算出的最佳摇臂的命中率。（最完美的是每次都是选最大收益摇臂best_arm）
            best_arm_counts[i][j] = (counts[best_arm] - 1000000) * 100.0 / (j + 1)

        # epsilon = 0.3

    # 计算每一次试验所有老虎机对最大收益摇臂的的平均命中率，作为y值
    # print("best_arm_counts[i][j]:",best_arm_counts)
    ys = np.mean(best_arm_counts, axis=0)
    print(ys)
    # 生成x，即有效试验次数
    xs = range(len(ys))
    ax.plot(xs, ys, label=st)

plt.xlabel('Steps')
plt.ylabel('Optimal pulls')
plt.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fancybox=True, shadow=True)
plt.ylim((0, 130))
plt.show()
