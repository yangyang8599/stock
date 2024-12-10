import sys

from PyQt5.QtWidgets import QApplication

from instock.qt.window_lib import MyCoreWindow


class MyWindow(MyCoreWindow):
    def __init__(self):
        super().__init__()
        # 调用外部模块生成价格数据和移动平均数据
        self.regenerate_data()

# 启动应用
app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec_())
