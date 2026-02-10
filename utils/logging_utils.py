# utils/logging_utils.py

import logging
import os

def setup_logging(exp_dir):
    """
    设置日志记录。

    Args:
        exp_dir (str): 实验目录
    """
    log_dir = os.path.join(exp_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, 'training.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging is set up.")

def log_memory_usage():
    """
    记录当前内存使用情况。
    """
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # 转换为MB
    logging.info(f"Current memory usage: {mem:.2f} MB")
