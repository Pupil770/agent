def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    mock_data = {
        "北京": "晴天，气温 25°C，空气质量良好",
        "上海": "多云，气温 28°C，有轻微雾霾",
        "广州": "雷阵雨，气温 32°C，湿度 85%",
    }
    return mock_data.get(city, f"{city}：暂无天气数据")


def calculate(expression: str) -> str:
    """计算数学表达式的结果，例如 '123*456' 或 '2+3*4'"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"
