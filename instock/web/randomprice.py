import numpy as np
import matplotlib.pyplot as plt

# 初始设置
P1 = 1000  # 初始价格
std_dev = 0.0005  # 每个点的波动标准差（0.05%）
strong = 0.1 #趋势强化因子
windows_weights = {5: 0.25, 10: 0.25, 20: 0.15, 60: 0.35} #均线影响权重

volatility = 0.1  # 总波动范围10%

num_points = 300  # 分时数据点数量
windows = [5, 10, 20, 60]  # 移动平均窗口：5日、10日、20日、60日


# 生成价格序列
prices = [P1]

# 用于存储不同窗口的移动平均线
moving_averages = {window: [] for window in windows}

for i in range(1, num_points):
    # 计算每个窗口的移动平均
    for window in windows:
        if i >= window:
            moving_avg = np.mean(prices[i - window:i])  # 当前窗口的移动平均
        else:
            moving_avg = np.mean(prices[:i + 1])  # 计算窗口不够时的均值
        moving_averages[window].append(moving_avg)

    # 计算斜率：每个窗口的移动平均线斜率（当前MA与上一个MA的差值）

    # 计算每个窗口的斜率（通过差分法计算）
    slopes = {}
    for window in windows:
        if i >= window:
            ma = moving_averages[window]
            slopes[window] = (ma[-1] - ma[-2])/P1   # 斜率 = (最后值 - 倒数第二个值) / P1
        else:
            slopes[window] = 0
    weighted_slope = sum(windows_weights[window] * slopes[window] for window in windows) / sum(windows_weights.values())
    # print(weighted_slope)
    # 使用斜率调整正态分布的均值
    new_price = prices[-1] * (1 + np.random.normal(strong*weighted_slope, (1+strong)*std_dev))  # 用斜率调整正态分布的均值
    # 限制价格波动范围：每天的波动不超过初始价格的±10%
    new_price = np.clip(new_price, P1 * (1 - volatility), P1 * (1 + volatility))
    prices.append(new_price)

# 绘制价格序列
plt.plot(prices, label="Price")

# 绘制各个移动平均线
for window in windows:
    plt.plot(moving_averages[window], label=f"MA{window}",linewidth=0.4)

plt.title("Price with Trend Based on Multiple MAs")
plt.xlabel("Time")
plt.ylabel("Price")

# 显示网格
plt.grid(True)

# 显示图表
plt.legend()
plt.show()
