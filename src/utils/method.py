# src/utils/method.py
import os
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Tuple, Optional

__all__ = ["init_logger", "find_latest_checkpoint"]

def init_logger(log_dir: str, logger_name: str = "train") -> logging.Logger:
    """
    初始化日志系统：同时输出到控制台与文件。
    log 文件路径：{log_dir}/train_YYYYmmdd-HHMMSS.log
    """
    os.makedirs(log_dir, exist_ok=True)

    # Asia/Taipei (UTC+8)
    tz_taipei = timezone(timedelta(hours=8))
    ts = datetime.now(tz_taipei).strftime("%Y%m%d-%H%M")
    log_path = os.path.join(log_dir, f"{logger_name}_{ts}.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler（例如多次调用 init）
    if logger.handlers:
        # 如果你希望每次训练都新建文件，可清空旧 handlers
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 文件
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_path}")
    return logger


def find_latest_checkpoint(save_dir: str,
                           model_suffix: str = "net_PRSNet.pth",
                           optim_suffix: str = "optim.pth"
                           ) -> Tuple[Optional[str], Optional[str], int]:
    """
    在 save_dir 中查找形如 '00012_net_PRSNet.pth' 的最新 checkpoint。
    返回: (model_ckpt_path, optim_ckpt_path, last_epoch)
         如果未找到，返回 (None, None, -1)
    """
    if not os.path.isdir(save_dir):
        return None, None, -1

    # 匹配前缀为 5 位或任意位数字的 epoch 编号
    pattern = re.compile(r"^(\d+)_{}".format(re.escape(model_suffix)))
    candidates = []
    for fname in os.listdir(save_dir):
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            candidates.append((epoch, fname))

    if not candidates:
        return None, None, -1

    candidates.sort(key=lambda x: x[0])
    last_epoch, last_fname = candidates[-1]
    model_ckpt = os.path.join(save_dir, last_fname)
    optim_ckpt = os.path.join(save_dir, f"{last_epoch:05d}_{optim_suffix}")
    return model_ckpt, optim_ckpt, last_epoch
