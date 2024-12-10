import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import linregress
import time


def moving_average(prices, window=5):
    """
    计算从左到右单向移动平均值，覆盖完整范围。
    :param prices: 原始价格序列
    :param window: 移动平均窗口大小
    :return: 移动平均序列，长度与原始序列相同
    """
    smoothed_prices = np.zeros(len(prices))
    for i in range(len(prices)):
        if i < window:
            smoothed_prices[i] = np.mean(prices[:i + 1])  # 边界处理：窗口不足时取已有数据的均值
        else:
            smoothed_prices[i] = np.mean(prices[i - window + 1:i + 1])  # 正常窗口
    return smoothed_prices



def find_turning_points(smoothed_prices):
    """找到平滑序列中的拐点（局部极值）"""
    turning_points = []
    for i in range(1, len(smoothed_prices) - 1):
        if smoothed_prices[i] > smoothed_prices[i - 1] and smoothed_prices[i] > smoothed_prices[i + 1]:
            turning_points.append(i)  # 局部最大值
        elif smoothed_prices[i] < smoothed_prices[i - 1] and smoothed_prices[i] < smoothed_prices[i + 1]:
            turning_points.append(i)  # 局部最小值
    return turning_points


def map_turning_points_to_raw_data(prices, turning_points, window=2):
    """从移动平均拐点映射到原始数据，并找到局部极值"""
    highs = []
    lows = []
    n = len(prices)

    for point in turning_points:
        if point < window or point >= n - window:
            continue

        start = max(0, point - window)
        end = min(n, point + window + 1)
        local_prices = prices[start:end]
        local_indices = np.arange(start, end)

        if len(local_prices) == 0:
            continue

        max_idx = np.argmax(local_prices)
        min_idx = np.argmin(local_prices)

        highs.append((local_indices[max_idx], local_prices[max_idx]))
        lows.append((local_indices[min_idx], local_prices[min_idx]))

    return highs, lows


def filter_cross_extremes(highs, lows):
    """确保高低点交替出现，并保留最高高点和最低低点"""
    if not highs or not lows:
        return highs, lows

    combined = sorted(highs + lows, key=lambda point: point[0])
    filtered_highs = []
    filtered_lows = []
    last_type = None

    for point in combined:
        if point in highs:
            if last_type == "high":
                if point[1] > filtered_highs[-1][1]:
                    filtered_highs[-1] = point
            else:
                filtered_highs.append(point)
                last_type = "high"
        elif point in lows:
            if last_type == "low":
                if point[1] < filtered_lows[-1][1]:
                    filtered_lows[-1] = point
            else:
                filtered_lows.append(point)
                last_type = "low"

    return filtered_highs, filtered_lows


def calculate_tolerance(prices):
    """计算当前片段的波动范围（标准差）"""
    return np.std(prices)


def is_converging(high_points, low_points, tolerance):
    """检查高低点是否符合收敛条件"""
    for i in range(1, len(high_points)):
        if high_points[i][1] > high_points[i - 1][1] + tolerance:
            return False

    for i in range(1, len(low_points)):
        if low_points[i][1] < low_points[i - 1][1] - tolerance:
            return False

    return True


def fit_line(points):
    """拟合直线 y = mx + c"""
    if len(points) < 2:
        return None, None
    x, y = zip(*points)
    if len(set(x)) == 1:
        return None, None
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, intercept


def plot_convergence(prices, smoothed_prices, highs, lows,turning_points, up_line, down_line, px, pn):
    """绘制价格图及收敛线段，并标记检测区间"""
    x = np.arange(len(prices))
    smoothed_x = np.arange(len(smoothed_prices))

    plt.clf()
    plt.plot(x, prices, label="Price", color="blue")
    plt.plot(smoothed_x, smoothed_prices, label="Moving Average", color="orange", linestyle="--")

    if highs:
        high_x, high_y = zip(*highs)
        plt.scatter(high_x, high_y, color="red", label="High Points")
    if lows:
        low_x, low_y = zip(*lows)
        plt.scatter(low_x, low_y, color="green", label="Low Points")

    if up_line and up_line[0] is not None:
        up_slope, up_intercept = up_line
        line_x = np.arange(px, pn + 1)
        line_y = up_slope * line_x + up_intercept
        plt.plot(line_x, line_y, color="red", linestyle="--", label="Upper Line")

    if down_line and down_line[0] is not None:
        down_slope, down_intercept = down_line
        line_x = np.arange(px, pn + 1)
        line_y = down_slope * line_x + down_intercept
        plt.plot(line_x, line_y, color="green", linestyle="--", label="Lower Line")

    # 标出平滑序列中的拐点
    if turning_points:
        turning_x = turning_points
        turning_y = smoothed_prices[turning_points]
        plt.scatter(turning_x, turning_y, color="purple", label="Turning Points",s=5)

    if pn - px >= 1:
        ax = plt.gca()
        rect = Rectangle((px, min(prices[px:pn + 1])),
                         pn - px,
                         max(prices[px:pn + 1]) - min(prices[px:pn + 1]),
                         linewidth=1.5, edgecolor='purple', facecolor='none', linestyle="--")
        ax.add_patch(rect)

    plt.title(f"Price Convergence (Px={px}, Pn={pn})")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.draw()
    plt.pause(0.2)


def detect_triangle_convergence(prices, ma_window=10, search_window=5, skip=5):
    """检测三角形收敛并绘制结果"""
    n = len(prices)
    smoothed_prices = moving_average(prices, ma_window)

    turning_points = find_turning_points(smoothed_prices)

    for px in range(n - 1, ma_window - 1, -skip):
        pn = n - 1
        if pn - px < 1:
            continue

        search_prices = prices[px:pn + 1]
        local_highs, local_lows = map_turning_points_to_raw_data(
            prices[px:pn + 1],
            [tp for tp in turning_points if px <= tp <= pn],
            window=search_window
        )
        local_highs, local_lows = filter_cross_extremes(local_highs, local_lows)
        tolerance = calculate_tolerance(search_prices)

        converging = is_converging(local_highs, local_lows, tolerance)
        if converging and len(local_highs) > 1 and len(local_lows) > 1:
            filtered_highs = [(x + px, y) for x, y in local_highs]
            filtered_lows = [(x + px, y) for x, y in local_lows]

            up_line = fit_line(filtered_highs)
            down_line = fit_line(filtered_lows)

            if up_line[0] is not None and down_line[0] is not None:
                print(f"Convergence detected at Px={px}, Pn={pn}")
                plot_convergence(prices, smoothed_prices, filtered_highs, filtered_lows,turning_points, up_line, down_line, px, pn)
                return

        plot_convergence(prices, smoothed_prices, local_highs, local_lows,turning_points, None, None, px, pn)


# 生成示例价格数据
prices = [3900]
for _ in range(300):
    prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))  # 假设每天波动最大±2%

detect_triangle_convergence(prices, ma_window=10, search_window=5, skip=5)
