# alicia_duo_sdk/utils/config.py

import os
from omegaconf import OmegaConf

def get_planning_config(name: str):
    """
    加载 config/lqt.yaml 等规划参数配置
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "config", f"{name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    return OmegaConf.load(config_path)

