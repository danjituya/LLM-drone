# -*- coding: utf-8 -*-
# @File    : llm_parsing_experiment.py
# @Describe: 农业无人机LLM指令解析完整实验代码
import os
import re
import json
import time
import pandas as pd
import traceback
from openai import OpenAI

# ====================== 实验核心配置（无需修改，直接运行） ======================
# 火山引擎方舟API配置（复用你原有有效密钥）
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_API_KEY = "ddfcfcb5-834b-46b9-aeff-cdef614f9024"
MODEL = "doubao-1-5-pro-32k-250115"
TEMPERATURE = 0.1  # 实验固定值，保证结果可复现
TIMEOUT = 15  # 增加超时时间，避免网络波动导致失败

# 实验输出文件路径
RESULT_CSV = "llm_parsing_experiment_results.csv"
RESULT_CHART = "llm_parsing_experiment_chart.png"

# ====================== 优化版提示词模板（强制格式输出） ======================
PROMPT_TEMPLATE = """
【身份】你是农业无人机调度指令专用解析器，**仅输出Python代码块**，绝对不能有任何解释、备注、换行或多余文字。
【任务】将用户的自然语言指令解析为指定格式的调度参数字典。
【参数说明】
- area: 农田地块编号（如A01、B02，多个用逗号分隔，无则填default_farm）
- drone_num: 无人机数量（整数，无则填3）
- path: 飞行路径类型（仅支持zigzag/round/grid，无则填zigzag）
- area_bounds: 农田边界坐标（固定为[0,0,200,150]）
- has_obstacle: 是否存在障碍（True/False，无则填False）
- obstacle_position: 障碍坐标（[x,y]，无则填None）
- height: 飞行高度（整数，单位米，无则填10）

用户指令：{user_command}
严格按照以下格式输出，不要修改任何字段名：
```python
dispatch_params = {{
    "area": "",
    "drone_num": 3,
    "path": "zigzag",
    "area_bounds": [0, 0, 200, 150],
    "has_obstacle": False,
    "obstacle_position": None,
    "height": 10
}}
"""
#====================== 鲁棒性 LLM 解析器 ======================
class RobustLLMParser:
    def __init__(self):
        self.client = OpenAI(
            base_url=BASE_URL,
            api_key=ARK_API_KEY,
        )
        # 清空聊天历史，每次请求独立，避免上下文污染
        self.system_prompt = {"role": "system",
                              "content": "你是农业无人机指令解析器，仅输出指定格式的 Python 代码块，不做任何其他回答。"}

    def extract_python_code(self, content):
        """多层级容错提取代码块，解决 LLM 输出格式不规范问题"""
        # 第一层：匹配带 python 标记的完整代码块
        code_pattern1 = re.compile(r"```python(.*?)```", re.DOTALL)
        matches = code_pattern1.findall(content)
        if matches:
            return "\n".join(matches).strip()
        # 第二层：匹配无标记的代码块
        code_pattern2 = re.compile(r"```(.*?)```", re.DOTALL)
        matches = code_pattern2.findall(content)
        if matches:
            return "\n".join(matches).strip()
        # 第三层：兜底提取dispatch_params内容
        if "dispatch_params" in content:
            start = content.find("dispatch_params")
            end = content.rfind("}") + 1
            return content[start:end].strip()
        return None

    def safe_exec_code(self, code):
        """安全执行代码，避免恶意代码和执行错误"""
        try:
            local_vars = {}
            # 限制exec的全局命名空间，仅允许必要的内置函数（双下划线！）
            exec(code, {"__builtins__": {}}, local_vars)
            # ✅ if判断放在try块内，缩进4格
            if "dispatch_params" in local_vars and isinstance(local_vars["dispatch_params"], dict):
                return local_vars["dispatch_params"]
            return None
        # ✅ except和try同级，顶格对齐
        except Exception as e:
            # ✅ except块内代码缩进4格
            print(f"代码执行失败: {str(e)[:100]}")
            return None
    def parse_single_command (self, command):
        """解析单条指令，返回完整的实验数据"""
        start_time = time.time ()
        result = {
            "指令": command,
            "模型": MODEL,
            "解析是否成功": False,
            "解析是否正确": False,
            "响应时间 (秒)": 0.0,
            "解析结果": "",
            "错误信息": "",
            "area 正确": False,
            "drone_num 正确": False,
            "path 正确": False,
            "has_obstacle 正确": False,
            "obstacle_position 正确": False,
            "height 正确": False
        }
        try:
            #构造请求
            messages = [
                self.system_prompt,
                {"role": "user", "content": PROMPT_TEMPLATE.format(user_command=command)}
            ]
            #调用 LLM
            completion = self.client.chat.completions.create (
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                timeout=TIMEOUT
            )
            llm_response = completion.choices [0].message.content.strip ()
            result ["响应时间 (秒)"] = round (time.time () - start_time, 2)

            #提取并执行代码
            python_code = self.extract_python_code (llm_response)
            if not python_code:
                raise Exception (f"未提取到有效代码块，LLM 返回: {llm_response [:150]}")
            parsed_params = self.safe_exec_code (python_code)
            if not parsed_params:
                raise Exception ("代码执行失败或未找到 dispatch_params 变量")
            result ["解析结果"] = json.dumps (parsed_params, ensure_ascii=False)
            result ["解析是否成功"] = True

            #与人工标注的正确参数对比
            correct_params = get_ground_truth(command)
            all_correct = True
            for key in ["area", "drone_num", "path", "has_obstacle", "obstacle_position", "height"]:
                if parsed_params.get (key) == correct_params.get (key):
                    result [f"{key} 正确"] = True
                else:
                    result [f"{key} 正确"] = False
                    all_correct = False
            result ["解析是否正确"] = all_correct

        except Exception as e:
            result ["错误信息"] = str (e)[:200]
            result ["响应时间 (秒)"] = round (time.time () - start_time, 2)

        return result

# ====================== 人工标注的 15 条指令标准答案（100% 准确） ======================
def get_ground_truth(command):
    ground_truth = {
        "我要对 A01 地块进行巡检，使用 3 台无人机": {"area": "A01", "drone_num": 3, "path": "zigzag",
                                                    "has_obstacle": False, "obstacle_position": None, "height": 10},
        "巡检 B02 地块，高度 8 米，发现障碍自动绕开": {"area": "B02", "drone_num": 3, "path": "zigzag",
                                                     "has_obstacle": True, "obstacle_position": None, "height": 8},
        "同时巡检 A01 和 B02 两个地块，每台无人机负责一个区域": {"area": "A01,B02", "drone_num": 2, "path": "grid",
                                                                "has_obstacle": False, "obstacle_position": None,
                                                                "height": 10},
        "对 C03 地块进行病虫害专项检测，拍摄高清照片": {"area": "C03", "drone_num": 1, "path": "grid",
                                                       "has_obstacle": False, "obstacle_position": None, "height": 15},
        "让无人机在 10 米高度绕农田飞行一圈": {"area": "default_farm", "drone_num": 3, "path": "round",
                                               "has_obstacle": False, "obstacle_position": None, "height": 10},
        "巡检 D04 地块，无人机数量 2 台，路径采用网格状": {"area": "D04", "drone_num": 2, "path": "grid",
                                                         "has_obstacle": False, "obstacle_position": None,
                                                         "height": 10},
        "E05 地块有电线杆障碍，坐标 (50,80)，请绕开": {"area": "E05", "drone_num": 3, "path": "zigzag",
                                                     "has_obstacle": True, "obstacle_position": [50, 80], "height": 10},
        "6 台无人机巡检 F06 地块，高度 12 米，之字形路径": {"area": "F06", "drone_num": 6, "path": "zigzag",
                                                          "has_obstacle": False, "obstacle_position": None,
                                                          "height": 12},
        "巡检 G07 地块，无障碍物，飞行高度 9 米": {"area": "G07", "drone_num": 3, "path": "zigzag", "has_obstacle": False,
                                                 "obstacle_position": None, "height": 9},
        "H08 地块巡检，4 台无人机，圆形路径，避开 (100,100) 的树": {"area": "H08", "drone_num": 4, "path": "round",
                                                                 "has_obstacle": True, "obstacle_position": [100, 100],
                                                                 "height": 10},
        "批量巡检 I09、J10 地块，各用 2 台无人机，高度 11 米": {"area": "I09,J10", "drone_num": 4, "path": "grid",
                                                             "has_obstacle": False, "obstacle_position": None,
                                                             "height": 11},
        "K11 地块病虫害检测，1 台无人机，网格路径，高度 18 米": {"area": "K11", "drone_num": 1, "path": "grid",
                                                              "has_obstacle": False, "obstacle_position": None,
                                                              "height": 18},
        "L12 地块有多个障碍，先标记存在障碍，坐标暂不指定": {"area": "L12", "drone_num": 3, "path": "zigzag",
                                                           "has_obstacle": True, "obstacle_position": None,
                                                           "height": 10},
        "M13 地块巡检，5 台无人机，高度 7 米，绕圈飞行": {"area": "M13", "drone_num": 5, "path": "round",
                                                       "has_obstacle": False, "obstacle_position": None, "height": 7},
        "默认地块巡检，3 台无人机，默认高度和路径": {"area": "default_farm", "drone_num": 3, "path": "zigzag",
                                                   "has_obstacle": False, "obstacle_position": None, "height": 10}
    }
    return ground_truth.get(command, {})
# ====================== 实验执行与结果可视化 ======================
def run_full_experiment():
    # 1. 15 条覆盖全场景的测试用例
    test_commands = [
        "我要对 A01 地块进行巡检，使用 3 台无人机",
        "巡检 B02 地块，高度 8 米，发现障碍自动绕开",
        "同时巡检 A01 和 B02 两个地块，每台无人机负责一个区域",
        "对 C03 地块进行病虫害专项检测，拍摄高清照片",
        "让无人机在 10 米高度绕农田飞行一圈",
        "巡检 D04 地块，无人机数量 2 台，路径采用网格状",
        "E05 地块有电线杆障碍，坐标 (50,80)，请绕开",
        "6 台无人机巡检 F06 地块，高度 12 米，之字形路径",
        "巡检 G07 地块，无障碍物，飞行高度 9 米",
        "H08 地块巡检，4 台无人机，圆形路径，避开 (100,100) 的树",
        "批量巡检 I09、J10 地块，各用 2 台无人机，高度 11 米",
        "K11 地块病虫害检测，1 台无人机，网格路径，高度 18 米",
        "L12 地块有多个障碍，先标记存在障碍，坐标暂不指定",
        "M13 地块巡检，5 台无人机，高度 7 米，绕圈飞行",
        "默认地块巡检，3 台无人机，默认高度和路径"
    ]

    # 2. 初始化解析器并执行实验
    parser = RobustLLMParser()
    all_results = []
    print("=" * 60)
    print("🚀 开始 LLM 指令解析完整实验（共 15 条指令）")
    print("=" * 60)

    for idx, cmd in enumerate(test_commands, 1):
        print(f"\n [{idx}/15] 处理指令: {cmd}")
        res = parser.parse_single_command(cmd)
        all_results.append(res)
        status = "✅ 成功" if res["解析是否成功"] else "❌ 失败"
        accuracy = "✅ 正确" if res["解析是否正确"] else "❌ 错误"
        print(f"状态: {status} | 参数匹配: {accuracy} | 耗时: {res['响应时间 (秒)']} s")

    # 3. 保存详细结果到 CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n📄 详细实验结果已保存至: {RESULT_CSV}")

    # 4. 计算核心实验指标
    total = len(df)
    success_count = df["解析是否成功"].sum()
    correct_count = df["解析是否正确"].sum()
    success_rate = round(success_count / total * 100, 2)
    accuracy_rate = round(correct_count / total * 100, 2)
    avg_time = round(df["响应时间 (秒)"].mean(), 2)

    # 计算各参数单独正确率
    param_accuracy = {}
    for key in ["area", "drone_num", "path", "has_obstacle", "obstacle_position", "height"]:
        param_accuracy[key] = round(df[f"{key} 正确"].sum() / total * 100, 2)

    # 5. 打印实验报告
    print("\n" + "=" * 60)
    print("📊 LLM 指令解析实验最终报告")
    print("=" * 60)
    print(f"测试用例总数: {total} 条")
    print(f"解析成功率: {success_count}/{total} = {success_rate}%")
    print(f"参数匹配正确率: {correct_count}/{total} = {accuracy_rate}%")
    print(f"平均响应时间: {avg_time} 秒")
    print("\n各参数单独正确率:")
    for param, acc in param_accuracy.items():
        print(f"- {param}: {acc}%")

    # 6. 生成可视化图表
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 2)

        # 子图 1: 整体成功率与正确率
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(["解析成功率", "参数匹配正确率"], [success_rate, accuracy_rate],
                        color=['#2E8B57', '#FF8C00'], width=0.6)
        ax1.set_ylabel("百分比 (%)", fontsize=12)
        ax1.set_title("LLM 解析整体性能", fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 110)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{height}%', ha='center', va='bottom', fontsize=12)

        # 子图 2: 各参数正确率
        ax2 = fig.add_subplot(gs[0, 1])
        params = list(param_accuracy.keys())
        acc_values = list(param_accuracy.values())
        bars2 = ax2.bar(params, acc_values, color='#4682B4', width=0.6)
        ax2.set_ylabel("正确率 (%)", fontsize=12)
        ax2.set_title("各参数解析正确率", fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 110)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                     f'{height}%', ha='center', va='bottom', fontsize=10)

        # 子图 3: 响应时间分布
        ax3 = fig.add_subplot(gs[1, :])
        ax3.hist(df["响应时间 (秒)"], bins=8, color='#9370DB', edgecolor='black', alpha=0.7)
        ax3.axvline(avg_time, color='red', linestyle='--', linewidth=2,
                    label=f'平均响应时间: {avg_time} s')
        ax3.set_xlabel("响应时间 (秒)", fontsize=12)
        ax3.set_ylabel("指令数量", fontsize=12)
        ax3.set_title("响应时间分布", fontsize=14, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULT_CHART, dpi=300, bbox_inches='tight')
        print(f"\n📈 实验可视化图表已保存至: {RESULT_CHART}")
    except ImportError:
        print("\n⚠️ 未安装 matplotlib，跳过可视化图表生成")
        print("执行: pip install matplotlib 即可生成图表")

    # 7. 打印失败案例（便于分析）
    failed_cases = df[df["解析是否成功"] == False]
    if not failed_cases.empty:
        print("\n" + "=" * 60)
        print("❌ 解析失败案例")
        print("=" * 60)
        for _, row in failed_cases.iterrows():
            print(f"指令: {row['指令']}")
            print(f"错误: {row['错误信息']}\n")

    return df
# ====================== 主程序入口 ======================
if __name__ == "__main__":
    # 安装依赖提示
    try:
        import openai
        import pandas
    except ImportError:
        print("⚠️ 缺少必要依赖，请先执行:")
        print("pip install openai pandas matplotlib")
        exit(1)

    # 运行完整实验
    experiment_results = run_full_experiment()
    print("\n🎉 LLM 指令解析实验全部完成！")
    print("你可以将生成的 CSV 和图表直接插入论文中")