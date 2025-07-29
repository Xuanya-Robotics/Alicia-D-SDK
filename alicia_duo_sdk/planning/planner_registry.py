# planner_registry.py

from importlib import import_module

# 名称映射到模块路径（不包含 .py）
_PLANNERS = {
    "linear": "alicia_duo_sdk.planning.planners.linear",
    "cartesian": "alicia_duo_sdk.planning.planners.cartesian",
    # 未来支持："lspb": "planners.lspb", "moveit": "planners.moveit_planner"
}


def get_planner(name: str):
    if name not in _PLANNERS:
        raise ValueError(f"未注册的 planner 名称: {name}\n可用选项: {list(_PLANNERS.keys())}")
    try:
        module = import_module(_PLANNERS[name])
        return module.LinearPlanner()  # 或者动态查找类
    except Exception as e:
        raise ImportError(f"无法导入 planner 模块 '{_PLANNERS[name]}': {e}")



def list_available_planners():
    return list(_PLANNERS.keys())
