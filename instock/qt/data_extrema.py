import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter


import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

class ExtremaFinder:
    def __init__(self, prices, savgol_filter_win_length=7, polyorder=2, sigma=1.5, buffer=20):
        self.prices = prices

        #下面都是要调优的参数
        self.savgol_filter_win_length = savgol_filter_win_length
        self.polyorder = polyorder
        self.sigma = sigma
        self.buffer = buffer
        self.last_detected_index = -1  # 初始值
        self.extrema_data = {'maxima': [], 'minima': [], 'maxima_prices': [], 'minima_prices': []}
        self.smoothed_curve = None  # 用于保存平滑曲线

    def symmetric_ma(self, prices_segment):
        """
        使用 Savitzky-Golay 滤波器实现对称平滑。
        :param prices_segment: 要平滑的数据片段
        :return: 对称平滑后的价格数据
        """
        if self.savgol_filter_win_length % 2 == 0:
            raise ValueError("window_length must be an odd number.")
        if self.savgol_filter_win_length > len(prices_segment):
            raise ValueError("window_length must not exceed the data length.")
        return savgol_filter(prices_segment, window_length=self.savgol_filter_win_length, polyorder=self.polyorder)

    def is_monotonic(self):
        """
        判断当前新增的价格变化趋势是单向的
        """
        if self.last_detected_index < 0:
            return False  # 没有上一个检测点，无法确定趋势

        last_price = self.prices[self.last_detected_index]
        trend = None  # 初始化趋势，None表示没有趋势

        for i in range(self.last_detected_index + 1, len(self.prices)):
            new_price = self.prices[i]
            if new_price > last_price:
                if trend is None:
                    trend = 'up'  # 第一个增加的价格，确定上升趋势
                elif trend == 'down':
                    return False  # 趋势发生了变化，返回False
            elif new_price < last_price:
                if trend is None:
                    trend = 'down'  # 第一个下降的价格，确定下降趋势
                elif trend == 'up':
                    return False  # 趋势发生了变化，返回False

            last_price = new_price  # 更新为最新的价格，用于下一次比较
        return True

    def get_nearest_extrema(self):
        """获取最近的极值点"""
        if self.last_detected_index >= 0:
            return (self.extrema_data['maxima'][-1], self.extrema_data['minima'][-1],
                    self.extrema_data['maxima_prices'][-1], self.extrema_data['minima_prices'][-1])
        return None

    def update_parameters(self, window_length=None, polyorder=None, sigma=None, buffer=None):
        """更新参数"""
        if window_length is not None:
            self.savgol_filter_win_length = window_length
        if polyorder is not None:
            self.polyorder = polyorder
        if sigma is not None:
            self.sigma = sigma
        if buffer is not None:
            self.buffer = buffer

    def update_extrema(self):
        """
        更新极值
        """
        # 如果新增的数据点是单向的，快速移动检测游标
        if self.last_detected_index >= 0 and self.is_monotonic():
            self.last_detected_index = len(self.prices) - 1
            return  # 没有新的数据点，跳过极值查找

        # 向前拓展buffer区域，避免重复计算
        start_idx = max(0, self.last_detected_index - self.buffer)
        smoothed_prices_window = self.prices[start_idx:]  # 截取新的小片段进行平滑

        # 判断是否需要重新计算平滑数据
        if self.last_detected_index is None:
            # 如果没有算过，需要重新计算平滑
            smoothed_prices = self.symmetric_ma(self.prices)
        elif self.last_detected_index < len(self.prices) - 1:
            # 计算新的平滑数据，防止越界
            smoothed_prices = self.symmetric_ma(smoothed_prices_window)
        elif self.last_detected_index == len(self.prices) - 1:
            # 已经算过切
            return
        else:
            raise ValueError("prices 数据和当前extraFinder数据不匹配.")

        # 高斯平滑
        gaussian_smoothed_prices = gaussian_filter(smoothed_prices, sigma=self.sigma)

        # 拼接高斯平滑数据到 smoothed_curve
        if self.smoothed_curve is not None:
            # 获取当前已经存储的数据大小
            current_length = len(self.smoothed_curve)
            new_length = len(self.prices)

            # 确保 smoothed_curve 的长度足够大
            if new_length > current_length:
                self.smoothed_curve.resize(new_length)

            # 直接拼接新增的数据
            self.smoothed_curve[current_length:] = gaussian_smoothed_prices[-(new_length - current_length):]
        else:
            self.smoothed_curve = gaussian_smoothed_prices

        # 查找局部极值
        local_maxima = argrelextrema(gaussian_smoothed_prices, np.greater)[0]
        local_minima = argrelextrema(gaussian_smoothed_prices, np.less)[0]

        # 合并并排序极值索引
        extrema_indices = np.sort(np.concatenate((local_maxima, local_minima)))
        extrema_types = ["maxima" if idx in local_maxima else "minima" for idx in extrema_indices]

        # 全局映射：将窗口内的索引映射回全局价格数据索引
        global_indices = [start_idx + idx for idx in extrema_indices]

        # 局部调整：在原始数据中保持性质一致性
        for idx, extrema_type in zip(global_indices, extrema_types):
            local_start = max(0, idx - 2)
            local_end = min(len(self.prices), idx + 3)  # 前 2 后 2 加上当前点
            local_window = self.prices[local_start:local_end]

            if extrema_type == "maxima":  # 仅查找局部极大值
                local_max_idx = np.argmax(local_window) + local_start
                if local_max_idx not in self.extrema_data['maxima']:  # 避免重复
                    self.extrema_data['maxima'].append(local_max_idx)
                    self.extrema_data['maxima_prices'].append(self.prices[local_max_idx])
            elif extrema_type == "minima":  # 仅查找局部极小值
                local_min_idx = np.argmin(local_window) + local_start
                if local_min_idx not in self.extrema_data['minima']:  # 避免重复
                    self.extrema_data['minima'].append(local_min_idx)
                    self.extrema_data['minima_prices'].append(self.prices[local_min_idx])

        # 更新检测游标
        self.last_detected_index = len(self.prices) - 1



def symmetric_ma(prices, window_length=5, polyorder=2):
    """
    使用 Savitzky-Golay 滤波器实现对称平滑。

    :param prices: 原始价格序列 (list or np.array)。
    :param window_length: 滑动窗口大小 (int)，必须是奇数。
    :param polyorder: 多项式阶数 (int)。
    :return: 对称平滑后的数据 (np.array)。
    """
    # 确保 window_length 为奇数，且不超过数据长度
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd number.")
    if window_length > len(prices):
        raise ValueError("window_length must not exceed the data length.")

    # 使用 Savitzky-Golay 滤波器
    return savgol_filter(prices, window_length=window_length, polyorder=polyorder)

def find_local_extrema_with_symmetric_gaussian(prices, window_length=5, polyorder=2, sigma=1.5, window=240, buffer=20):
    """
    使用对称平滑结合高斯平滑后的数据进行极值查找。

    :param prices: 原始价格序列 (list or np.array)。
    :param window_length: 对称平滑的滑动窗口大小 (int)，必须是奇数。
    :param polyorder: 对称平滑的多项式阶数 (int)。
    :param sigma: 高斯平滑的强度 (float)。
    :param window: 窗口大小 (int)，默认240。
    :param buffer: 额外保留的点数，用于提高开头计算精度 (int)，默认20。
    :return: (极大值索引, 极小值索引, 极大值价格, 极小值价格, 高斯平滑后的数据)
    """
    # 对称平滑
    smoothed_prices = symmetric_ma(prices, window_length=window_length, polyorder=polyorder)

    # 高斯平滑
    gaussian_smoothed_prices = gaussian_filter(smoothed_prices, sigma=sigma)

    # 数据长度不足窗口大小时，调整窗口到数据长度
    actual_window = min(window, len(gaussian_smoothed_prices))
    start_idx = max(0, len(gaussian_smoothed_prices) - actual_window - buffer)  # 多留 buffer 数据
    window_ma = gaussian_smoothed_prices[start_idx:]  # 从 start_idx 开始截取数据

    # 查找局部极值
    local_maxima = argrelextrema(window_ma, np.greater)[0]
    local_minima = argrelextrema(window_ma, np.less)[0]

    # 合并并排序极值索引
    extrema_indices = np.sort(np.concatenate((local_maxima, local_minima)))
    extrema_types = ["maxima" if idx in local_maxima else "minima" for idx in extrema_indices]

    # 全局映射：将窗口内的索引映射回全局价格数据索引
    global_indices = [start_idx + idx for idx in extrema_indices]

    # 局部调整：在原始数据中保持性质一致性
    maxima_indices, minima_indices = [], []
    maxima_prices, minima_prices = [], []

    for idx, extrema_type in zip(global_indices, extrema_types):
        local_start = max(0, idx - 3)
        local_end = min(len(prices), idx + 2)  # 前 3 后 1 加上当前点
        local_window = prices[local_start:local_end]

        if extrema_type == "maxima":  # 仅查找局部极大值
            local_max_idx = np.argmax(local_window) + local_start
            maxima_indices.append(local_max_idx)
            maxima_prices.append(prices[local_max_idx])
        elif extrema_type == "minima":  # 仅查找局部极小值
            local_min_idx = np.argmin(local_window) + local_start
            minima_indices.append(local_min_idx)
            minima_prices.append(prices[local_min_idx])

    return maxima_indices, minima_indices, maxima_prices, minima_prices, gaussian_smoothed_prices
