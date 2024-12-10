import sys

import matplotlib.pyplot as plt
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtWidgets import QStyle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMainWindow, QPushButton
from matplotlib.backend_tools import Cursors
from matplotlib.widgets import Cursor
from spyder_kernels.utils.lazymodules import numpy

from instock.qt.data_generator import generate_prices
from instock.qt.data_extrema import find_local_extrema_with_symmetric_gaussian, ExtremaFinder


class MyCoreWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Price Trend with Multiple Moving Averages")
        self.setGeometry(100, 100, 800, 600)

        self.timer = QTimer(self)  # 定时器
        self.timer.timeout.connect(self.auto_run)  # 定时器超时调用 auto_add_point 方法
        self.is_auto_running = False  # 标志，判断是否正在自动添加数据点

        #初始化存数据
        self.prices = None
        self.cur_p1 = None #相当于每天的开盘价。
        self.moving_averages = None
        self.ma_windows = [3, 5, 10]
        self.extrema_max_idx = None
        self.extrema_max_prices = None
        self.extrema_min_idx = None
        self.extrema_min_prices = None
        self.extrema_annotations = {
            'maxima_points': [],
            'minima_points': [],
            'maxima_texts': [],
            'minima_texts': [],
            'smoothed_curve': None  # 初始值为 None，类型为 Optional[Line2D]
        }
        self.convergence_min_points = 4 #收敛最少极值数量 (int)。
        self.convergence_man_points = 10 #收敛最大极值数量 (int)。

        #极值计算器
        self.prices_extrema_finder = None #在第一次初始化prices时同步初始化

        # 存储每个数据序列
        self.plot_lines = {}

        # 假设你有一个原始的数据点样式存储
        self.highlighted_point = None  # 用来保存当前高亮的点

        # 记录数据的显示范围
        self.current_xlim = None
        self.current_ylim = None

        # 变量用来绘制十字光标
        self.cursor_vline = None
        self.cursor_hline = None
        self.cursor_xy_text = None

        # 设置布局
        layout = QVBoxLayout()

        #==================== 创建按钮容器
        button_container = QWidget()
        # 设置按钮容器的固定宽度和高度（避免随着窗口变化拉伸）
        button_container.setFixedWidth(480)  # 固定宽度，避免按钮在窗口变化时拉伸
        button_container.setFixedHeight(50)  # 固定高度，保持按钮行的高度不变
        # 创建按钮的水平布局
        button_layout = QHBoxLayout(button_container)
        button_layout.setSpacing(5)  # 设置按钮之间的间距为5（减少间距）
        button_layout.setContentsMargins(0, 0, 0, 0)  # 去除布局的外部间距
        # 创建功能按钮
        self.zoom_in_button = self.create_button(self.zoom_in, button_layout, '➕', 'Zoom In')  # 使用 Unicode 字符 + 作为图标
        self.zoom_out_button = self.create_button(self.zoom_out, button_layout, '➖', 'Zoom Out')  # 使用 Unicode 字符 - 作为图标
        self.restore_button = self.create_button(self.restore_current_window, button_layout, 'current','current windows')
        self.global_windows_button = self.create_button(self.restore_global_windows, button_layout, 'global','global window')
        self.regenerate_button = self.create_button(self.regenerate_data, button_layout, QStyle.SP_FileDialogNewFolder,'Regenerate Data')  # 使用有效的图标
        self.pan_left_button = self.create_button(self.pan_left, button_layout, QStyle.SP_ArrowLeft, 'Pan Left')
        self.pan_right_button = self.create_button(self.pan_right, button_layout, QStyle.SP_ArrowRight, 'Pan Right')
        self.pan_up_button = self.create_button(self.pan_up, button_layout, QStyle.SP_ArrowUp, 'Pan Up')
        self.pan_down_button = self.create_button(self.pan_down, button_layout, QStyle.SP_ArrowDown, 'Pan Down')
        self.add_point_button = self.create_button(self.add_point, button_layout, QStyle.SP_DesktopIcon, 'Add Point')
        self.auto_add_point_button = self.create_button(self.start_or_stop_auto_runner, button_layout, "自动随机加数",'Auto Add Point')
        self.cal_extrema_button = self.create_button(self.do_update_extrema, button_layout, "计算极值", 'Cal Extrema')
        # 将按钮容器添加到主布局
        layout.addWidget(button_container)




        #================创建图形画布
        # 创建图形和画布
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        # 监听鼠标移动事件
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        # 监听鼠标点击事件
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        # 绑定图例的点击事件
        self.canvas.mpl_connect('pick_event', self.on_legend_click)
        # 绘制图表
        self.draw_plot()
        # 将画布添加到主布局
        layout.addWidget(self.canvas)

        # 创建一个容器并设置布局
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 设置矩形放大框
        self.rectangle_selector = RectangleSelector(self.ax, self.on_rectangle_select, useblit=True,
                                                    button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                                    interactive=True)

    def create_button(self,btn_click_fun, button_layout, icon, tooltip):
        """创建带有图标的按钮"""
        button = QPushButton()
        # 如果是字符图标，直接使用 QIcon 包装
        if isinstance(icon, str):
            button.setText(icon)  # 直接用 Unicode 字符
            button.setFixedHeight(40)  # 设置按钮固定大小
        else:
            button.setIcon(self.style().standardIcon(icon))  # 处理其他标准图标
            button.setIconSize(QSize(24, 24))  # 设置图标大小
            button.setFixedSize(40, 40)  # 设置按钮固定大小
        button.setToolTip(tooltip)  # 设置按钮的悬浮提示

        # 绑定按钮事件
        button.clicked.connect(btn_click_fun)
        button_layout.addWidget(button)
        return button

    def zoom_in(self):
        """基于当前十字光标位置放大图表"""
        if self.rectangle_selector.active:  # 如果存在放大矩形框，取消
            self.rectangle_selector.set_active(False)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # 获取十字光标位置
        cursor_x = self.cursor_vline.get_xdata()[0] if self.cursor_vline else (xlim[0] + xlim[1]) / 2
        cursor_y = self.cursor_hline.get_ydata()[0] if self.cursor_hline else (ylim[0] + ylim[1]) / 2

        scale_factor = 1.1
        self.ax.set_xlim(cursor_x - (cursor_x - xlim[0]) / scale_factor, cursor_x + (xlim[1] - cursor_x) / scale_factor)
        self.ax.set_ylim(cursor_y - (cursor_y - ylim[0]) / scale_factor, cursor_y + (ylim[1] - cursor_y) / scale_factor)
        self.canvas.draw()

    def zoom_out(self):
        """基于当前十字光标位置缩小图表"""
        if self.rectangle_selector.active:  # 如果存在放大矩形框，取消
            self.rectangle_selector.set_active(False)

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # 获取十字光标位置
        cursor_x = self.cursor_vline.get_xdata()[0] if self.cursor_vline else (xlim[0] + xlim[1]) / 2
        cursor_y = self.cursor_hline.get_ydata()[0] if self.cursor_hline else (ylim[0] + ylim[1]) / 2

        scale_factor = 0.9
        self.ax.set_xlim(cursor_x - (cursor_x - xlim[0]) / scale_factor, cursor_x + (xlim[1] - cursor_x) / scale_factor)
        self.ax.set_ylim(cursor_y - (cursor_y - ylim[0]) / scale_factor, cursor_y + (ylim[1] - cursor_y) / scale_factor)
        self.canvas.draw()

    def restore_current_window(self):
        """恢复到基于新数据的显示范围"""
        # 恢复时基于当前数据显示范围
        last_300_prices = self.prices[-300:]
        current_min_y = min(last_300_prices)
        current_max_y = max(last_300_prices)
        self.ax.set_ylim(current_min_y*0.999, current_max_y*1.001)
        self.current_ylim = self.ax.get_ylim()

        self.ax.set_xlim(len(self.prices)-300-5, len(self.prices)+5)
        self.current_xlim = self.ax.get_xlim()

        self.canvas.draw()

    def restore_global_windows(self):
        """恢复到基于全数据的显示范围"""
        # 恢复时基于当前数据显示范围
        current_min_y = min(self.prices)
        current_max_y = max(self.prices)
        self.ax.set_ylim(current_min_y*0.999, current_max_y*1.001)
        self.current_ylim = self.ax.get_ylim()

        self.ax.set_xlim(-5, len(self.prices)+5)
        self.current_xlim = self.ax.get_xlim()

        self.canvas.draw()

    def draw_plot(self):
        # 绘制价格线并存储
        if self.prices is not None:
            self.plot_lines["Price"] = self.ax.plot(self.prices, label="Price")[0]
        # 绘制移动平均线并存储
        for window in self.ma_windows:
            if self.moving_averages is not None and self.moving_averages[window] is not None:
                label = f"MA{window}"
                self.plot_lines[label] = self.ax.plot(self.moving_averages[window], label=label, linewidth=0.4)[0]
                # if window>5:
                #     self.plot_lines[label].set_visible(False)

        self.ax.set_title("Price with Trend Based on Multiple MAs")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=False)
        Cursor(self.ax, useblit=True, color='red', linewidth=1)

        # 设置图例标签的picker属性为True
        for legend_item in self.ax.get_legend().get_texts():
            legend_item.set_picker(True)

        # 初始化选中的点
        self.highlighted_point, = self.ax.plot([], [], 'ro', markersize=8)  # 用红色圆点标记高亮点

        # 记录数据的显示范围
        self.current_xlim = self.ax.get_xlim()
        self.current_ylim = self.ax.get_ylim()

        # 重置绘制十字光标
        self.cursor_vline = None
        self.cursor_hline = None
        self.cursor_xy_text = None

        self.update_ticks()
        # 更新画布
        self.canvas.draw()
    def add_point(self):
        # 新增数据点
        cur_price = self.generate_data()

        # 更新价格线
        self.plot_lines["Price"].set_xdata(range(len(self.prices)))  # 更新 x 数据
        self.plot_lines["Price"].set_ydata(self.prices)  # 更新 y 数据

        # 如果有移动平均线，也需要更新
        for window in self.ma_windows:
            self.plot_lines[f"MA{window}"].set_xdata(range(len(self.moving_averages[window])))  # 更新 x 数据
            self.plot_lines[f"MA{window}"].set_ydata(self.moving_averages[window])  # 更新 y 数据

        self.move_figure_window(x_offset=1)
        min_y, max_y = self.current_ylim
        if cur_price < min_y or cur_price > max_y :
            self.scale_y(y_min=(cur_price if cur_price<min_y else min_y),y_max=(cur_price if cur_price>max_y else max_y))

        # #更新刻度
        self.update_ticks()
        # 强制更新显示
        self.canvas.draw()

    def auto_run(self):
        self.add_point()

    def start_or_stop_auto_runner(self, t=100):
        """启动定时器，每隔 t 毫秒添加一个点"""
        if not self.is_auto_running:
            self.is_auto_running = True
            self.timer.start(t)  # 启动定时器，每 t 毫秒调用一次 auto_add_point
        else:
            self.is_auto_running = False
            self.timer.stop()

    def regenerate_data(self):
        """重新生成数据并更新图表"""
        self.prices = None
        self.moving_averages = None
        self.prices_extrema_finder = None
        self.generate_data()

        # 重新绘制图表
        self.ax.cla()
        self.draw_plot()
        # 监听鼠标移动事件
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def pan_left(self):
        """平移图表"""
        xlim = self.ax.get_xlim()
        self.ax.set_xlim(xlim[0] - 10, xlim[1] - 10)
        self.canvas.draw()

    def pan_right(self):
        """平移图表"""
        xlim = self.ax.get_xlim()
        self.ax.set_xlim(xlim[0] + 10, xlim[1] + 10)
        self.canvas.draw()

    def pan_up(self):
        """平移图表"""
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(ylim[0] + 10, ylim[1] + 10)
        self.canvas.draw()

    def pan_down(self):
        """平移图表"""
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(ylim[0] - 10, ylim[1] - 10)
        self.canvas.draw()

    def on_mouse_click(self, event):
        """鼠标点击时，取消矩形框（如果矩形框已经激活），否则绘制矩形框"""
        if event.inaxes != self.ax:
            return  # 确保点击发生在图形区域内

        if self.rectangle_selector.active:
            # 如果矩形框激活，点击取消矩形框
            self.rectangle_selector.set_active(False)
            self.canvas.draw()  # 重新绘制
        else:
            # 如果矩形框没有激活，不做任何操作，允许绘制矩形框
            pass

    def on_mouse_move(self, event):
        """鼠标移动时，更新十字光标位置并高亮当前数据点"""
        if event.inaxes != self.ax:
            return

        for legend_item in self.ax.get_legend().get_texts():
            if legend_item.contains(event)[0]:
                self.canvas.set_cursor(Cursors.HAND)  # 设置鼠标为手指形状
                return
        self.canvas.set_cursor(Cursors.POINTER)  # 恢复为默认的箭头形状

        # 获取鼠标的坐标
        x = event.xdata
        y = event.ydata  # 获取鼠标的y坐标
        if x is None or y is None:
            return

        # 找到最接近的点的索引
        index = int(round(x))  # 取最近的整数索引
        if index < 0 or index >= len(self.prices):
            # 如果当前索引无效，即没有对应的数据点，则不做任何处理
            if self.highlighted_point is not None:
                self.highlighted_point.remove()  # 如果有高亮点，移除
                self.highlighted_point = None  # 清空高亮点引用
            self.canvas.draw()
            return

        # 获取当前点的价格
        current_price = self.prices[index]

        # 更新十字光标的竖线
        if self.cursor_vline is None:
            self.cursor_vline = self.ax.axvline(x, color='green', linestyle='-', linewidth=1)
        else:
            self.cursor_vline.set_xdata([x, x])

        # 更新横线（横线始终与鼠标所在的价格同步）
        if self.cursor_hline is None:
            self.cursor_hline = self.ax.axhline(y, color='green', linestyle='-', linewidth=1)
        else:
            self.cursor_hline.set_ydata([y, y])  # 更新横线位置

        if self.cursor_xy_text is None:
            self.cursor_xy_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, verticalalignment='top')
        self.cursor_xy_text.set_text(f'x = {event.xdata:.2f}, y = {event.ydata:.2f}')


        # 删除之前的高亮点（确保它是正确的对象）
        if self.highlighted_point is not None:
            self.highlighted_point.remove()

        # 高亮当前数据点
        # self.highlighted_point, = self.ax.plot(x, current_price, 'ro', markersize=8)  # 用红色圆圈高亮显示当前点
        self.highlighted_point, = self.ax.plot(x, current_price, marker='o',
            markersize=12,
            markeredgecolor='red',  # 边框颜色
            markerfacecolor=(1, 0, 0, 0.3)  # 半透明红色填充，RGBA格式
        )
        self.ax.set_ylim(self.current_ylim)

        # 更新画布
        self.canvas.draw()

    def on_rectangle_select(self, eclick, erelease):
        """矩形选择框放大"""
        self.ax.set_xlim(eclick.xdata, erelease.xdata)
        self.ax.set_ylim(eclick.ydata, erelease.ydata)
        self.canvas.draw()

    def on_legend_click(self, event):
        # 确保是点击了图例标签
        label = event.artist.get_text()

        # 根据点击的标签控制对应线条的显示/隐藏
        if label in self.plot_lines:
            line = self.plot_lines[label]
            # 切换线条的可见性
            visible = not line.get_visible()
            line.set_visible(visible)

            # 调整标签的透明度，隐藏时更淡
            legend_item = event.artist
            legend_item.set_alpha(0.5 if not visible else 1.0)

            # 更新画布
            self.canvas.draw()

    def move_figure_window(self, x_offset=None, y_offset=None, need_draw=False):
        """更新图表的 x 轴和 y 轴范围"""
        if x_offset is not None:
            self.ax.set_xlim(self.current_xlim[0] + x_offset, self.current_xlim[1] + x_offset)
            self.current_xlim = self.ax.get_xlim()

        if y_offset is not None:
            self.ax.set_ylim(self.current_ylim[0] +  y_offset, self.current_ylim[1] + y_offset)
            self.current_ylim = self.ax.get_ylim()

        # 更新画布
        if need_draw:
            self.canvas.draw()
    def scale_y(self, y_min=None, y_max=None, need_draw=False):
        if y_min is None or y_max is None:
            return
        self.ax.set_ylim(y_min*0.999, y_max*1.001)
        self.current_ylim = self.ax.get_ylim()

        # 更新画布
        if need_draw:
            self.canvas.draw()

    def update_ticks(self):
        if self.current_xlim is not None:
            x_min, x_max = self.current_xlim
            # ---x刻度
            # 计算每 240 个点设置一个主刻度
            step = 240
            # 根据当前 x 轴范围设置刻度
            ticks = range(int(x_min // step) * step + step, int(x_max // step) * step + step, step)
            self.ax.set_xticks(ticks)
            # 设置次要刻度（每 60 个点设置一个次要刻度）
            self.ax.set_xticks(range(int(x_min //60 )*60 +60, int(x_max // 60)*60 + 60, 60), minor=True)
            # # 设置最小刻度（每 30 个点设置一个最小刻度）
            # self.ax.set_xticks(range(30, len(self.prices), 30), minor=True)
            # 显示主网格
            self.ax.grid(True, which='major', axis='x', linestyle='-', color='b', linewidth=1.5)
            # 显示次要网格
            self.ax.grid(True, which='minor', axis='x', linestyle='--', color='g', linewidth=1)

        # ---y刻度
        if self.current_ylim is not None:
            # 计算 P 的 1% 来设置 y 轴刻度间隔
            y_interval = self.cur_p1 * 0.01 if self.cur_p1 is not None else (self.current_ylim[1]-self.current_ylim[0]) * 0.1
            # 设置 y 轴刻度，间隔为 P 的 1%
            y_min, y_max = self.current_ylim
            y_ticks = numpy.arange(y_min, y_max, y_interval)
            # 更新 y 轴的刻度
            self.ax.set_yticks(y_ticks)
            # 显示网格
            self.ax.grid(True, which='both', axis='y', linestyle='--', color='g', linewidth=1)
    def do_update_extrema(self):
        self.update_extrema(True)

    def update_extrema(self, clear_existing_markers=True):
        if self.prices_extrema_finder is None:
            self.prices_extrema_finder = ExtremaFinder(self.prices)

        # 清除现有标注，如果需要
        if clear_existing_markers:
            """
                   清除现有的极值标注和曲线
                   """
            # 删除极大值和极小值标注点
            for marker in self.extrema_annotations['maxima_points']:
                marker.remove()
            for marker in self.extrema_annotations['minima_points']:
                marker.remove()

            # 删除极大值和极小值标注文本
            for text in self.extrema_annotations['maxima_texts']:
                text.remove()
            for text in self.extrema_annotations['minima_texts']:
                text.remove()

            # 删除平滑曲线
            if self.extrema_annotations['smoothed_curve']:
                self.extrema_annotations['smoothed_curve'].remove()

            # 清空存储的标注数据
            self.extrema_annotations['maxima_points'] = []
            self.extrema_annotations['minima_points'] = []
            self.extrema_annotations['maxima_texts'] = []
            self.extrema_annotations['minima_texts'] = []
            self.extrema_annotations['smoothed_curve'] = None

        #计算并更新极值（数据）
        self.prices_extrema_finder.update_extrema()

        # 获取极大值和极小值的索引和价格
        extrema_max_idx = self.prices_extrema_finder.extrema_data['maxima']
        extrema_max_prices = self.prices_extrema_finder.extrema_data['maxima_prices']
        extrema_min_idx = self.prices_extrema_finder.extrema_data['minima']
        extrema_min_prices = self.prices_extrema_finder.extrema_data['minima_prices']
        smoothed_curve = self.prices_extrema_finder.smoothed_curve

        # 标注极大值
        for idx, price in zip(extrema_max_idx, extrema_max_prices):
            # 添加极大值标注点
            marker = self.ax.scatter(idx, price, color='red', label='Maxima', zorder=5)
            self.extrema_annotations['maxima_points'].append(marker)
            # 添加极大值标注文本
            text = self.ax.annotate(f'{price:.2f}', (idx, price), textcoords="offset points", xytext=(0, 5),
                                    ha='center', fontsize=9)
            self.extrema_annotations['maxima_texts'].append(text)

        # 标注极小值
        for idx, price in zip(extrema_min_idx, extrema_min_prices):
            # 添加极小值标注点
            marker = self.ax.scatter(idx, price, color='blue', label='Minima', zorder=5)
            self.extrema_annotations['minima_points'].append(marker)
            # 添加极小值标注文本
            text = self.ax.annotate(f'{price:.2f}', (idx, price), textcoords="offset points", xytext=(0, -10),
                                    ha='center', fontsize=9)
            self.extrema_annotations['minima_texts'].append(text)

        # 绘制平滑曲线
        if smoothed_curve is not None:
            self.extrema_annotations['smoothed_curve'] = self.ax.plot(range(0, len(smoothed_curve)), smoothed_curve,
                                                label=f'Smoothed MA', linestyle='--', alpha=0.4, color='red')[0]

        # 刷新画布
        self.canvas.draw()



        # self.extrema_max_idx, self.extrema_min_idx, self.extrema_max_prices, self.extrema_min_prices, smoothed_ma  = find_local_extrema_with_symmetric_gaussian(self.prices)
        # # 标注极大值
        # self.ax.scatter(self.extrema_max_idx, self.extrema_max_prices, color='red', label='Maxima', zorder=5)
        # for idx, price in zip(self.extrema_max_idx, self.extrema_max_prices):
        #     self.ax.annotate(f'{price:.2f}', (idx, price), textcoords="offset points", xytext=(0, 5), ha='center',
        #                  fontsize=9)
        #
        # # 标注极小值
        # self.ax.scatter(self.extrema_min_idx, self.extrema_min_prices, color='blue', label='Minima', zorder=5)
        # for idx, price in zip(self.extrema_min_idx, self.extrema_min_prices):
        #     self.ax.annotate(f'{price:.2f}', (idx, price), textcoords="offset points", xytext=(0, -10), ha='center',
        #                  fontsize=9)
        #
        # # 高斯平滑后的均线
        # # 平滑数据对应的起始索引
        # start_idx = len(self.prices) - len(smoothed_ma)
        # # 在 x 轴上绘制平滑数据
        # self.ax.plot(range(start_idx, len(self.prices)), smoothed_ma, label=f'Smoothed MA{3}', linestyle='--',alpha=0.8)
        #
        # self.canvas.draw()

    def generate_data(self):
        self.prices, self.moving_averages, self.cur_p1, cur_price, *_ = generate_prices(windows=self.ma_windows,prices=self.prices,moving_averages=self.moving_averages)
        return cur_price

    # 移动窗口搜索
    def moving_window_convergence(self):

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










