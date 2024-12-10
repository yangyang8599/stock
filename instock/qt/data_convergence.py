import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


# 判断是否收敛
def check_convergence(indices, prices, min_points=4, max_points=10):
    """
    判断指定区间和极值点是否收敛。
    :param indices: 极值点的索引 (list)。
    :param prices: 极值点的价格 (list)。
    :param min_points: 最少极值数量 (int)。
    :param max_points: 最大极值数量 (int)。
    :return: 是否收敛 (bool), 上升趋势拟合 (dict), 下降趋势拟合 (dict), 收敛角度, 方差。
    """
    if len(indices) < min_points:
        return False, None, None, None, None

    # 分别提取高点和低点
    high_indices = indices[::2]
    high_prices = prices[::2]
    low_indices = indices[1::2]
    low_prices = prices[1::2]

    # 线性拟合上升趋势线
    slope_high, intercept_high, r_value_high, _, _ = linregress(high_indices, high_prices)
    variance_high = np.var(np.array(high_prices) - (slope_high * np.array(high_indices) + intercept_high))

    # 线性拟合下降趋势线
    slope_low, intercept_low, r_value_low, _, _ = linregress(low_indices, low_prices)
    variance_low = np.var(np.array(low_prices) - (slope_low * np.array(low_indices) + intercept_low))

    # 收敛角度
    convergence_angle = abs(slope_high - slope_low)

    # 收敛判定：方差小，收敛角度小
    is_converging = (len(indices) <= max_points and
                     variance_high < 0.01 and variance_low < 0.01 and
                     convergence_angle < 0.1)

    return is_converging, {"slope": slope_high, "intercept": intercept_high}, \
           {"slope": slope_low, "intercept": intercept_low}, convergence_angle, max(variance_high, variance_low)


# 移动窗口搜索
def moving_window_convergence(prices, maxima_indices, minima_indices, min_points=4, max_points=10):
    """
    移动窗口查找收敛点。
    :param prices: 原始价格数据。
    :param maxima_indices: 极大值索引。
    :param minima_indices: 极小值索引。
    :param min_points: 最少极值数量 (int)。
    :param max_points: 最大极值数量 (int)。
    :return: 收敛段结果 (list of dict)。
    """
    # 合并极值点并排序
    all_indices = np.array(maxima_indices + minima_indices)
    all_prices = np.array([prices[idx] for idx in maxima_indices + minima_indices])
    sorted_indices = np.argsort(all_indices)
    all_indices = all_indices[sorted_indices]
    all_prices = all_prices[sorted_indices]

    results = []
    start_idx = len(all_indices) - min_points  # 从最近的 4 个点开始

    while start_idx >= 0:
        for end_idx in range(start_idx + min_points, min(len(all_indices), start_idx + max_points) + 1):
            indices = all_indices[start_idx:end_idx]
            prices_segment = all_prices[start_idx:end_idx]

            # 判断是否收敛
            is_converging, up_trend, down_trend, angle, variance = check_convergence(
                indices, prices_segment, min_points, max_points
            )

            if is_converging:
                results.append({
                    "start": indices[0],
                    "end": indices[-1],
                    "up_trend": up_trend,
                    "down_trend": down_trend,
                    "convergence_angle": angle,
                    "variance": variance
                })
                start_idx = start_idx - 1  # 向前移动窗口
                break
        else:
            start_idx -= 1  # 当前窗口未找到收敛点，继续缩小窗口

    return results


# 动态绘图
def plot_convergence(prices, results, maxima_indices, minima_indices):
    """
    动态绘制收敛结果。
    :param prices: 原始价格数据。
    :param results: 收敛段结果 (list of dict)。
    :param maxima_indices: 极大值索引。
    :param minima_indices: 极小值索引。
    """
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Prices", alpha=0.5)
    plt.scatter(maxima_indices, [prices[i] for i in maxima_indices], color='red', label='Maxima')
    plt.scatter(minima_indices, [prices[i] for i in minima_indices], color='blue', label='Minima')

    for result in results:
        start, end = result["start"], result["end"]
        up_trend = result["up_trend"]
        down_trend = result["down_trend"]

        # 上升趋势线
        up_line_x = np.array([start, end])
        up_line_y = up_trend["slope"] * up_line_x + up_trend["intercept"]
        plt.plot(up_line_x, up_line_y, color='green', linestyle='--', label='Up Trend')

        # 下降趋势线
        down_line_x = np.array([start, end])
        down_line_y = down_trend["slope"] * down_line_x + down_trend["intercept"]
        plt.plot(down_line_x, down_line_y, color='orange', linestyle='--', label='Down Trend')

    plt.title("Convergence Analysis")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
