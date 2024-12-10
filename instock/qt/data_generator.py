import numpy as np

def generate_single_point(prices, moving_averages, windows, windows_weights, P1, std_dev, strong, volatility_limit):
    """
    生成一个新的价格点和对应的移动平均值。

    参数:
    - prices: 当前已有的价格数据
    - moving_averages: 当前已有的移动平均值字典
    - windows: 移动平均窗口大小列表
    - windows_weights: 每个窗口对应的权重，用于斜率加权
    - P1: 初始价格（用于生成数据的基准价格）
    - std_dev: 标准差，用于价格波动
    - strong: 趋势强度
    - volatility_limit: 波动率限制

    返回:
    - prices: 原始的价格数据（没有改变的部分）
    - moving_averages: 原始的移动平均数据（没有改变的部分）
    - new_price: 新生成的价格
    - new_moving_averages: 新计算的移动平均数据
    """
    i = len(prices)

    # 计算各个窗口的斜率（基于已有的 moving_averages）
    slopes = {}
    for window in windows:
        # 获取上一移动平均值
        prev_ma = moving_averages[window][-1] if i > window else np.mean(prices[:i])  # 确保不会越界
        slopes[window] = (moving_averages[window][-1] - prev_ma) / P1 if i > window else 0

    # 计算加权斜率
    weighted_slope = sum(windows_weights[window] * slopes[window] for window in windows) / sum(windows_weights.values())

    # 生成新价格
    new_price = prices[-1] * (1 + np.random.normal(strong * weighted_slope, (1 + strong) * std_dev))
    new_price = np.clip(new_price, P1 * (1 - volatility_limit), P1 * (1 + volatility_limit))  # 限制价格波动范围

    # 将新价格添加到 prices 中
    prices.append(new_price)  # 直接修改原数组

    # 更新 moving_averages（基于更新后的 prices）
    new_moving_averages = {}
    for window in windows:
        # 计算当前窗口的移动平均（包括新生成的价格）
        new_moving_averages[window] = np.mean(prices[max(i - window, 0):i + 1])

    # 更新后的 moving_averages 也需要包含新的移动平均
    for window in windows:
        moving_averages[window].append(new_moving_averages[window])

    # 返回结果，顺序调整：原始数据放在前面
    return prices, moving_averages, new_price, new_moving_averages


def generate_prices(prices=None, moving_averages=None, windows=[3, 5, 10, 20, 60, 120, 240],
                    windows_weights={3: 0.1, 5: 0.1, 10: 0.1, 20: 0.1, 60: 0.1, 120: 0.2, 240: 0.3},
                    P1=100, std_dev=0.002, strong=1, volatility_limit=0.08):
    """
    根据是否有现有的价格数据来决定是生成 241 个价格，还是基于现有数据生成一个新的价格点。

    参数:
    - prices: 当前已有的价格数据。如果为空，则生成 241 个价格数据。
    - moving_averages: 当前已有的移动平均值字典。
    - windows: 移动平均窗口大小列表。
    - windows_weights: 每个窗口对应的权重，用于斜率加权。
    - P1: 初始价格（用于空数据生成数据的基准价格，一旦有了数据，第一个241周期是P1，后面都是前一个周期的收盘就）。
    - std_dev: 标准差，用于价格波动。
    - strong: 趋势强度。
    - volatility_limit: 波动率。

    返回:
    - prices: 原始的价格数据（没有改变的部分）。
    - moving_averages: 原始的移动平均数据（没有改变的部分）。
    - new_price: 新生成的价格。
    - new_moving_averages: 新计算的移动平均数据。
    - cur_p1:当前基准价
    """
    # 如果 prices 为空，则直接生成300个价格数据
    if prices is None:
        # 初始化 prices 和 moving_averages
        prices = [P1]
        moving_averages = {window: [] for window in windows}

        # 生成300个初始价格
        new_prices = []
        new_moving_averages = {window: [] for window in windows}
        for _ in range(241 - 1):  # 固定生成241个点
            prices, moving_averages, new_price, new_moving_avg = generate_single_point(
                prices, moving_averages, windows, windows_weights, P1, std_dev, strong, volatility_limit)
            new_prices.append(new_price)
            # 更新移动平均
            for window in windows:
                new_moving_averages[window].append(new_moving_avg[window])

        return prices, moving_averages,P1, new_prices, new_moving_averages

    # 如果 prices 不为空，则生成一个新的价格
    # 每天都有241个几个点，从242开始的新的241周期内，的P1等于上一个周期的收盘价
    cur_p1 = P1 if  len(prices) < 241 else prices[(len(prices) // 241 - 1) * 241 + 240]
    prices, moving_averages, new_price, new_moving_averages = generate_single_point(
        prices, moving_averages, windows, windows_weights, cur_p1, std_dev, strong, volatility_limit)

    # 返回原始的 prices 和 moving_averages，接着返回新生成的 new_price 和 new_moving_averages
    return prices, moving_averages, cur_p1, new_price, new_moving_averages
