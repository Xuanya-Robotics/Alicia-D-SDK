# planner_registry.py

from importlib import import_module
from alicia_d_sdk.utils.logger import beauty_print

# 名称映射到模块路径（不包含 .py）
_PLANNERS = {
    "linear": "alicia_d_sdk.planning.planners.linear",
    "cartesian": "alicia_d_sdk.planning.planners.cartesian",
    "lqt": "alicia_d_sdk.planning.planners.lqt"
    # 未来支持："lspb": "planners.lspb", "moveit": "planners.moveit_planner"
}


def get_planner(name: str):
    if name not in _PLANNERS:
        raise ValueError(f"未注册的 planner 名称: {name}\n可用选项: {list(_PLANNERS.keys())}")
    try:
        module = import_module(_PLANNERS[name])
        beauty_print(f"加载 planner: {name}", type="module")

        # 默认类名规则：LinearPlanner / CartesianPlanner / LQT
        class_name = {
            "linear": "LinearPlanner",
            "cartesian": "CartesianPlanner",
            "lqt": "LQT"
        }.get(name, None)

        if class_name is None or not hasattr(module, class_name):
            raise AttributeError(f"模块中未找到 {class_name} 类")

        return getattr(module, class_name)()

    except Exception as e:
        raise ImportError(f"无法导入 planner 模块 '{_PLANNERS[name]}': {e}")


def list_available_planners():
    return list(_PLANNERS.keys())


def list_available_planners():
    return list(_PLANNERS.keys())

if __name__ == "__main__":
    planner = get_planner("lqt")
    print(f"获取的 planner 类型: {type(planner)}")