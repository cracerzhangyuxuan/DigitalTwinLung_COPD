#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志配置模块

统一管理项目的日志输出格式和目标
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 全局 logger 缓存
_loggers = {}


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    设置并返回一个 logger 实例
    
    Args:
        name: logger 名称
        log_dir: 日志文件目录
        level: 日志级别
        console: 是否输出到控制台
        file: 是否写入文件
        
    Returns:
        logger: 配置好的 logger 实例
    """
    # 如果已存在，直接返回
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # 清除已有 handlers
    
    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件输出
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已存在的 logger，如果不存在则创建一个默认的
    
    Args:
        name: logger 名称
        
    Returns:
        logger: logger 实例
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name, console=True, file=False)


class LoggerMixin:
    """
    为类提供 logger 属性的 Mixin
    
    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("message")
    """
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# 日志级别使用规范
"""
| 级别     | 使用场景                           | 示例                                    |
|----------|------------------------------------|-----------------------------------------|
| DEBUG    | 调试信息，变量值，中间状态          | logger.debug(f"Patch shape: {x.shape}") |
| INFO     | 正常流程里程碑                      | logger.info("模型加载完成")              |
| WARNING  | 可恢复异常，需要关注                | logger.warning("GPU 内存不足，切换 CPU") |
| ERROR    | 任务失败，需要处理                  | logger.error(f"文件不存在: {path}")      |
| CRITICAL | 系统级错误，程序无法继续            | logger.critical("数据库连接失败")        |
"""

