#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import logging
import os.path
import signal
import sys
from abc import ABC

import tornado.escape
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import options
from tornado.log import enable_pretty_logging
from tornado import gen

# 在项目运行时，临时将项目路径添加到环境变量
cpath_current = os.path.dirname(os.path.dirname(__file__))
cpath = os.path.abspath(os.path.join(cpath_current, os.pardir))
sys.path.append(cpath)
log_path = os.path.join(cpath_current, 'log')
if not os.path.exists(log_path):
    os.makedirs(log_path)

# 配置日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置全局日志级别

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 控制台日志级别
console_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] (%(pathname)s:%(lineno)d - %(funcName)s) - %(message)s'
)  # 使用绝对路径，支持点击跳转
console_handler.setFormatter(console_formatter)

# 文件处理器
file_handler = logging.FileHandler(os.path.join(log_path, 'stock_web.log'), encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 文件日志级别
file_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d - %(funcName)s) - %(message)s'
)  # 文件中使用文件名，便于阅读
file_handler.setFormatter(file_formatter)

# 添加处理器到日志记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

import instock.lib.torndb as torndb
import instock.lib.database as mdb
import instock.lib.version as version
import instock.web.dataTableHandler as dataTableHandler
import instock.web.dataIndicatorsHandler as dataIndicatorsHandler
import instock.web.base as webBase

__author__ = 'myh '
__date__ = '2023/3/10 '


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            # 设置路由
            (r"/", HomeHandler),
            (r"/instock/", HomeHandler),
            # 使用datatable 展示报表数据模块。
            (r"/instock/api_data", dataTableHandler.GetStockDataHandler),
            (r"/instock/data", dataTableHandler.GetStockHtmlHandler),
            # 获得股票指标数据。
            (r"/instock/data/indicators", dataIndicatorsHandler.GetDataIndicatorsHandler),
            # 加入关注
            (r"/instock/control/attention", dataIndicatorsHandler.SaveCollectHandler),
        ]
        settings = dict(  # 配置
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=False,  # True,
            # cookie加密
            cookie_secret="027bb1b670eddf0392cdda8709268a17b58b7",
            debug=True,
        )
        super(Application, self).__init__(handlers, **settings)
        # Have one global connection to the blog DB across all handlers
        self.db = torndb.Connection(**mdb.MYSQL_CONN_TORNDB)


# 首页handler。
class HomeHandler(webBase.BaseHandler, ABC):
    @gen.coroutine
    def get(self):
        self.render("index.html",
                    stockVersion=version.__version__,
                    leftMenu=webBase.GetLeftMenu(self.request.uri))

def shutdown():
    print("Shutting down Tornado...")
    tornado.ioloop.IOLoop.current().stop()

def main():
    # tornado.options.parse_command_line()
    # tornado.options.options.logging = None

    http_server = tornado.httpserver.HTTPServer(Application())
    port = 9988
    http_server.listen(port)

    # Tornado 应用中捕捉信号，并在收到退出信号时停止事件循环：不然调试的时候不会退出
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown())  # 捕获 Ctrl+C 信号
    signal.signal(signal.SIGTERM, lambda sig, frame: shutdown())  # 捕获终止信号

    # enable_pretty_logging()
    # print("Logging level:", options.logging)
    print(f"服务已启动，web地址 : http://localhost:{port}/")
    logging.info(f"服务已启动，web地址 : http://localhost:{port}/")
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        shutdown()





if __name__ == "__main__":
    main()
