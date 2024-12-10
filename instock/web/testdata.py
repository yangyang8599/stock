import numpy as np
import matplotlib.pyplot as plt

def generate_price_data_random(n_points=200, start_price=100, max_daily_change=0.1):
    """
    生成基于每日随机波动的价格数据
    :param n_points: 数据点数量
    :param start_price: 初始价格
    :param max_daily_change: 每日最大波动幅度（百分比，0.1 表示10%）
    :return: 价格序列
    """
    prices = [start_price]  # 初始化价格序列
    for _ in range(1, n_points):
        # 随机生成相对于前一天的变化百分比
        daily_change = np.random.uniform(-max_daily_change, max_daily_change)
        # 计算当天价格
        new_price = prices[-1] * (1 + daily_change)
        prices.append(new_price)
    return np.array(prices)

# 生成示例数据
prices = generate_price_data_random(
    n_points=300,        # 数据点数量
    start_price=100,     # 起始价格
    max_daily_change=0.1 # 每日最大波动幅度（10%）
)

# 绘制生成的价格数据
plt.figure(figsize=(12, 6))
plt.plot(prices, label="Random Price Data", color="blue")
plt.title("Randomly Generated Price Data with Daily Variations")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()