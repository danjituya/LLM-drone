# -*- coding: utf-8 -*-
"""
农业无人机LLM指令解析性能验证实验（Windows专属版）
期刊适配：IEEE Transactions on Cybernetics
运行环境：Windows 10/11 + Python 3.8-3.11
依赖：pip install openai pandas numpy matplotlib seaborn scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import os
import yaml
from yaml import SafeDumper  # 新增这行，用于YAML序列化
import re
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from pathlib import Path
from openai import OpenAI


# ====================== 1. Windows专属配置（自动处理路径和编码） ======================
@dataclass
class LLMExperimentConfig:
    # 实验元数据
    seed: int = 42
    repeat_times: int = 5  # 每条指令重复5次（统计显著性要求）
    timeout: float = 15.0
    temperature: float = 0.1

    # 国内可用模型配置（只保留你能直接用的豆包，通义千问可选）
    models: Dict[str, Dict[str, str]] = None

    # 顶刊级提示词模板
    prompt_template: str = None

    # 测试用例（包含你已有的15条+5条鲁棒性测试）
    test_commands: List[str] = None

    # 指令复杂度分级
    command_complexity: Dict[str, str] = None

    # Windows路径自动处理
    result_dir: str = str(Path("./llm_experiment_results").resolve())
    log_dir: str = str(Path("./llm_experiment_logs").resolve())
    config_path: str = str(Path("./llm_experiment_config.yaml").resolve())

    def __post_init__(self):
        # 火山方舟三模型对比（请通过环境变量设置 API Key）
        import os
        self.models = self.models or {
            "doubao-1-5-pro-32k-250115": {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "api_key": os.environ.get("DOUBAO_API_KEY", "YOUR_API_KEY_HERE")
            },
            "deepseek-v3-2-251201": {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "api_key": os.environ.get("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
            },
            "glm-4-7-251222": {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "api_key": os.environ.get("GLM_API_KEY", "YOUR_API_KEY_HERE")
            }
        }

        # 优化版提示词模板（强制格式输出）
        self.prompt_template = self.prompt_template or """
        【身份】你是农业无人机调度指令专用解析器，**仅输出Python代码块**，绝对不能有任何解释、备注、换行或多余文字。
        【任务】将用户的自然语言指令解析为指定格式的调度参数字典。
        【参数说明】
        - area: 农田地块编号（如A01、B02，多个用逗号分隔，无则填default_farm）
        - drone_num: 无人机**总数量**（整数，多地块时为所有地块无人机数量之和，无则填3）
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
        """.strip()
        self.test_commands = self.test_commands or [
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
            "默认地块巡检，3 台无人机，默认高度和路径",
            "我要对 A01 地块巡检！！！使用 3 台无人机（紧急）",
            "巡检 B02 地块 高度大概 8 米左右 有障碍 自动绕开",
            "A01 B02 地块 各 2 台 网格路径 11 米 有障碍 (50,80)",
            "巡检 C03 地块 高度 十米 无人机 三台 之字形",
            "E05 地块有电线杆 坐标 50 和 80 绕开 高度 10 米"
        ]
        self.command_complexity = self.command_complexity or {
            "我要对 A01 地块进行巡检，使用 3 台无人机": "简单",
            "巡检 B02 地块，高度 8 米，发现障碍自动绕开": "简单",
            "同时巡检 A01 和 B02 两个地块，每台无人机负责一个区域": "中等",
            "对 C03 地块进行病虫害专项检测，拍摄高清照片": "简单",
            "让无人机在 10 米高度绕农田飞行一圈": "简单",
            "巡检 D04 地块，无人机数量 2 台，路径采用网格状": "简单",
            "E05 地块有电线杆障碍，坐标 (50,80)，请绕开": "中等",
            "6 台无人机巡检 F06 地块，高度 12 米，之字形路径": "简单",
            "巡检 G07 地块，无障碍物，飞行高度 9 米": "简单",
            "H08 地块巡检，4 台无人机，圆形路径，避开 (100,100) 的树": "中等",
            "批量巡检 I09、J10 地块，各用 2 台无人机，高度 11 米": "复杂",
            "K11 地块病虫害检测，1 台无人机，网格路径，高度 18 米": "中等",
            "L12 地块有多个障碍，先标记存在障碍，坐标暂不指定": "中等",
            "M13 地块巡检，5 台无人机，高度 7 米，绕圈飞行": "简单",
            "默认地块巡检，3 台无人机，默认高度和路径": "简单",
            "我要对 A01 地块巡检！！！使用 3 台无人机（紧急）": "噪声",
            "巡检 B02 地块 高度大概 8 米左右 有障碍 自动绕开": "噪声",
            "A01 B02 地块 各 2 台 网格路径 11 米 有障碍 (50,80)": "复杂 + 噪声",
            "巡检 C03 地块 高度 十米 无人机 三台 之字形": "噪声",
            "E05 地块有电线杆 坐标 50 和 80 绕开 高度 10 米": "歧义"
        }

def load_llm_config() -> LLMExperimentConfig:
    """加载/保存配置（Windows自动处理路径）"""
    config = LLMExperimentConfig()
    # 自动创建目录
    Path(config.result_dir).mkdir(exist_ok=True)
    Path(config.log_dir).mkdir(exist_ok=True)
    # 保存默认配置
    with open(config.config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            asdict(config),
            f,
            Dumper=SafeDumper,  # 指定安全序列化器
            allow_unicode=True,  # 替代ensure_ascii=False，支持中文
            indent=4,
            sort_keys=False  # 可选：保持字典原有顺序，避免字段乱序
        )
    return config
#====================== 2. Windows 兼容日志系统 ======================
def setup_llm_logger(config: LLMExperimentConfig) -> logging.Logger:
    logger = logging.getLogger("LLM_Drone_Command_Parsing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 避免重复日志

    # 文件日志（Windows编码兼容）
    log_file = Path(config.log_dir) / f"llm_experiment_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    ))

    # 控制台日志
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    ))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
#====================== 3. LLM 解析器核心（Windows 兼容） ======================
class OpenAICompatibleParser:
    """OpenAI兼容接口解析器（支持豆包、通义千问等所有国内模型）"""
    def __init__(self, model_name: str, base_url: str, api_key: str,
                 prompt_template: str, logger: logging.Logger):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.prompt_template = prompt_template
        self.logger = logger
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=15.0
        )

    def extract_python_code(self, content: str) -> Optional[str]:
        """多层级容错代码提取（解决Windows换行问题）"""
        # 层级1：标准Python代码块
        pattern1 = re.compile(r"```python(.*?)```", re.DOTALL)
        matches = pattern1.findall(content)
        if matches:
            return "\n".join(matches).strip().replace("\r\n", "\n")

        # 层级2：无语言标记的代码块
        pattern2 = re.compile(r"```(.*?)```", re.DOTALL)
        matches = pattern2.findall(content)
        if matches:
            return "\n".join(matches).strip().replace("\r\n", "\n")

        # 层级3：兜底提取dispatch_params
        if "dispatch_params" in content:
            start = content.find("dispatch_params")
            end = content.rfind("}") + 1
            return content[start:end].strip().replace("\r\n", "\n")

        self.logger.warning(f"未提取到有效代码: {content[:200]}")
        return None


    def safe_exec_code(self, code: str) -> Optional[Dict]:
        """安全执行代码（沙箱环境）"""
        try:
            local_vars = {}
            # 严格限制执行环境，禁止危险操作
            safe_globals = {
                "__builtins__": {
                    "dict": dict, "list": list, "int": int, "str": str, "bool": bool,
                    "None": None, "True": True, "False": False
                }
            }
            exec(code, safe_globals, local_vars)
            if "dispatch_params" in local_vars and isinstance(local_vars["dispatch_params"], dict):
                return local_vars["dispatch_params"]
            self.logger.warning("执行结果无dispatch_params字典")
            return None
        except Exception as e:
            self.logger.error(f"代码执行失败: {str(e)[:100]}")
            return None


    def classify_error_type(self, parsed_params: Dict, ground_truth: Dict) -> str:
        """错误类型分类（顶刊级错误分析）"""
        if not parsed_params:
            return "解析失败"

        # 字段缺失
        missing_fields = [k for k in ground_truth.keys() if k not in parsed_params]
        if missing_fields:
            return f"字段缺失: {','.join(missing_fields)}"

        # 类型错误
        type_errors = []
        type_map = {
            "area": str, "drone_num": int, "path": str,
            "has_obstacle": bool, "obstacle_position": (list, type(None)),
            "height": int
        }
        for k, expected_type in type_map.items():
            if k not in parsed_params:
                continue
            val = parsed_params[k]
            if not isinstance(val, expected_type):
                type_errors.append(f"{k}({type(val).__name__}≠{expected_type.__name__})")
        if type_errors:
            return f"类型错误: {','.join(type_errors)}"

        # 数值错误
        value_errors = [k for k in ground_truth.keys() if parsed_params.get(k) != ground_truth.get(k)]
        if value_errors:
            return f"数值错误: {','.join(value_errors)}"

        return "无错误"

    def parse_command(self, command: str) -> Dict:
        """解析单条指令，返回顶刊级实验数据"""
        start_time = time.time()
        result = {
            "指令": command,
            "模型名称": self.model_name,
            "解析成功率": False,
            "解析正确率": False,
            "响应时间(s)": 0.0,
            "解析结果": "",
            "错误信息": "",
            "错误类型": "",
            "复杂度分级": "",
            "area_正确率": False,
            "drone_num_正确率": False,
            "path_正确率": False,
            "has_obstacle_正确率": False,
            "obstacle_position_正确率": False,
            "height_正确率": False,
            "参数匹配F1": 0.0
        }

        try:
            # 构造请求
            messages = [
                {"role": "system", "content": "你是农业无人机指令解析器，仅输出指定格式Python代码块"},
                {"role": "user", "content": self.prompt_template.format(user_command=command)}
            ]

            # 调用LLM（真实API调用，生成真实数据）
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                timeout=15.0
            )
            response = completion.choices[0].message.content.strip()
            result["响应时间(s)"] = round(time.time() - start_time, 4)

            # 提取并执行代码
            code = self.extract_python_code(response)
            if not code:
                raise Exception("未提取到有效代码")
            parsed_params = self.safe_exec_code(code)
            if not parsed_params:
                raise Exception("代码执行失败")

            result["解析结果"] = json.dumps(parsed_params, ensure_ascii=False)
            result["解析成功率"] = True

            # 对比标准答案
            ground_truth = get_ground_truth(command)
            result["复杂度分级"] = get_command_complexity(command)

            # 计算各参数正确率和F1-Score（顶刊指标）
            all_correct = True
            true_positive = 0
            false_positive = 0
            false_negative = 0
            for k in ["area", "drone_num", "path", "has_obstacle", "obstacle_position", "height"]:
                parsed_val = parsed_params.get(k)
                true_val = ground_truth.get(k)
                correct = parsed_val == true_val
                result[f"{k}_正确率"] = correct
                if not correct:
                    all_correct = False

                # F1-Score计算
                if parsed_val is not None and true_val is not None:
                    if parsed_val == true_val:
                        true_positive += 1
                    else:
                        false_positive += 1
                        false_negative += 1
                elif parsed_val is not None:
                    false_positive += 1
                elif true_val is not None:
                    false_negative += 1

            # 计算F1-Score
            if true_positive + false_positive + false_negative == 0:
                f1 = 1.0
            else:
                precision = true_positive / (true_positive + false_positive) if (
                                                                                            true_positive + false_positive) > 0 else 0.0
                recall = true_positive / (true_positive + false_negative) if (
                                                                                         true_positive + false_negative) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            result["参数匹配F1"] = round(f1, 4)
            result["解析正确率"] = all_correct

            # 错误类型分类
            if not all_correct:
                result["错误类型"] = self.classify_error_type(parsed_params, ground_truth)

        except Exception as e:
            result["错误信息"] = str(e)[:200]
            result["响应时间(s)"] = round(time.time() - start_time, 4)
            result["错误类型"] = "API调用失败"

        return result

#====================== 4. 标准答案与复杂度分级 ======================
def get_ground_truth(command: str) -> Dict:
    """人工标注标准答案（包含你已有的15条）"""
    ground_truth = {
        "我要对 A01 地块进行巡检，使用 3 台无人机": {
            "area": "A01", "drone_num": 3, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "巡检 B02 地块，高度 8 米，发现障碍自动绕开": {
            "area": "B02", "drone_num": 3, "path": "zigzag",
            "has_obstacle": True, "obstacle_position": None, "height": 8
        },
        "同时巡检 A01 和 B02 两个地块，每台无人机负责一个区域": {
            "area": "A01,B02", "drone_num": 2, "path": "grid",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "对 C03 地块进行病虫害专项检测，拍摄高清照片": {
            "area": "C03", "drone_num": 1, "path": "grid",
            "has_obstacle": False, "obstacle_position": None, "height": 15
        },
        "让无人机在 10 米高度绕农田飞行一圈": {
            "area": "default_farm", "drone_num": 3, "path": "round",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "巡检 D04 地块，无人机数量 2 台，路径采用网格状": {
            "area": "D04", "drone_num": 2, "path": "grid",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "E05 地块有电线杆障碍，坐标 (50,80)，请绕开": {
            "area": "E05", "drone_num": 3, "path": "zigzag",
            "has_obstacle": True, "obstacle_position": [50,80], "height": 10
        },
        "6 台无人机巡检 F06 地块，高度 12 米，之字形路径": {
            "area": "F06", "drone_num": 6, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 12
        },
        "巡检 G07 地块，无障碍物，飞行高度 9 米": {
            "area": "G07", "drone_num": 3, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 9
        },
        "H08 地块巡检，4 台无人机，圆形路径，避开 (100,100) 的树": {
            "area": "H08", "drone_num": 4, "path": "round",
            "has_obstacle": True, "obstacle_position": [100,100], "height": 10
        },
        "批量巡检 I09、J10 地块，各用 2 台无人机，高度 11 米": {
            "area": "I09,J10", "drone_num": 4, "path": "grid",
            "has_obstacle": False, "obstacle_position": None, "height": 11
        },
        "K11 地块病虫害检测，1 台无人机，网格路径，高度 18 米": {
            "area": "K11", "drone_num": 1, "path": "grid",
            "has_obstacle": False, "obstacle_position": None, "height": 18
        },
        "L12 地块有多个障碍，先标记存在障碍，坐标暂不指定": {
            "area": "L12", "drone_num": 3, "path": "zigzag",
            "has_obstacle": True, "obstacle_position": None, "height": 10
        },
        "M13 地块巡检，5 台无人机，高度 7 米，绕圈飞行": {
            "area": "M13", "drone_num": 5, "path": "round",
            "has_obstacle": False, "obstacle_position": None, "height": 7
        },
        "默认地块巡检，3 台无人机，默认高度和路径": {
            "area": "default_farm", "drone_num": 3, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        # 鲁棒性测试用例
        "我要对 A01 地块巡检！！！使用 3 台无人机（紧急）": {
            "area": "A01", "drone_num": 3, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "巡检 B02 地块 高度大概 8 米左右 有障碍 自动绕开": {
            "area": "B02", "drone_num": 3, "path": "zigzag",
            "has_obstacle": True, "obstacle_position": None, "height": 8
        },
        "A01 B02 地块 各 2 台 网格路径 11 米 有障碍 (50,80)": {
            "area": "A01,B02", "drone_num": 4, "path": "grid",
            "has_obstacle": True, "obstacle_position": [50,80], "height": 11
        },
        "巡检 C03 地块 高度 十米 无人机 三台 之字形": {
            "area": "C03", "drone_num": 3, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "E05 地块有电线杆 坐标 50 和 80 绕开 高度 10 米": {
            "area": "E05", "drone_num": 3, "path": "zigzag",
            "has_obstacle": True, "obstacle_position": [50,80], "height": 10
        }
    }
    return ground_truth.get(command, {})
def get_command_complexity(command: str) -> str:
    """获取指令的复杂度分级"""
    config = load_llm_config()
    return config.command_complexity.get(command, "未知")
#====================== 5. 实验执行与统计分析 ======================
def run_llm_experiment(config: LLMExperimentConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行顶刊级LLM实验（Windows纯本地运行）"""
    # 初始化解析器
    parsers = {}
    for model_name, model_config in config.models.items():
        try:
            parser = OpenAICompatibleParser(
                model_name=model_name,
                base_url=model_config["base_url"],
                api_key=model_config["api_key"],
                prompt_template=config.prompt_template,
                logger=logger
            )
            parsers[model_name] = parser
            logger.info(f"✅ 初始化{model_name}解析器成功")
        except Exception as e:
            logger.error(f"❌ 初始化{model_name}解析器失败: {str(e)}")
            continue

    if not parsers:
        raise RuntimeError("无可用LLM解析器，请检查API密钥和网络连接")

    # 执行实验（真实API调用，生成真实数据）
    all_results = []
    total_commands = len(config.test_commands) * config.repeat_times * len(parsers)
    current = 0
    logger.info(
        f"🚀 开始LLM实验 | 模型数: {len(parsers)} | 指令数: {len(config.test_commands)} | 重复次数: {config.repeat_times} | 总请求数: {total_commands}")

    for model_name, parser in parsers.items():
        for cmd in config.test_commands:
            for i in range(config.repeat_times):
                current += 1
                logger.info(
                    f"[{current}/{total_commands}] [{model_name}] 处理指令({i + 1}/{config.repeat_times}): {cmd[:50]}...")
                result = parser.parse_command(cmd)
                result["实验轮次"] = i + 1
                all_results.append(result)
                time.sleep(0.5)  # 避免API限流

    # 保存原始数据（CSV格式，可直接插入Excel）
    df = pd.DataFrame(all_results)
    raw_data_path = Path(config.result_dir) / "llm_experiment_raw_data.csv"
    df.to_csv(raw_data_path, index=False, encoding="utf-8-sig")
    logger.info(f"📄 原始实验数据已保存至: {raw_data_path}")

    # 统计分析（顶刊级）
    stat_df = df.groupby("模型名称").agg({
        "解析成功率": ["mean", "std", stats.sem],
        "解析正确率": ["mean", "std", stats.sem],
        "响应时间(s)": ["mean", "std", stats.sem],  # 与parse_command字段名完全一致
        "参数匹配F1": ["mean", "std", stats.sem]  # 与parse_command字段名完全一致
    }).round(4)

    # 按复杂度分级统计
    complexity_stat = df.groupby(["模型名称", "复杂度分级"])["解析正确率"].mean().unstack().round(4) * 100

    # 错误类型统计
    error_stat = df[df["解析正确率"] == False].groupby(["模型名称", "错误类型"]).size().reset_index(name="次数")

    # 保存统计结果
    stat_df.to_csv(Path(config.result_dir) / "llm_experiment_statistics.csv", encoding="utf-8-sig")
    complexity_stat.to_csv(Path(config.result_dir) / "llm_complexity_statistics.csv", encoding="utf-8-sig")
    error_stat.to_csv(Path(config.result_dir) / "llm_error_statistics.csv", encoding="utf-8-sig")

    logger.info(f"📊 统计结果已保存至: {config.result_dir}")
    return df, stat_df
#====================== 6. 顶刊级可视化（Windows 兼容） ======================
def plot_llm_results(df: pd.DataFrame, stat_df: pd.DataFrame, config: LLMExperimentConfig, logger: logging.Logger):
    """生成符合IEEE Transactions格式的专业图表（Windows中文显示正常）"""
    # Windows中文显示配置
    plt.rcParams.update({
        'font.family': ['SimHei', 'Times New Roman'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.unicode_minus': False  # 解决负号显示问题
    })

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 子图1：整体解析性能
    ax1 = fig.add_subplot(gs[0, 0])
    success_mean = stat_df["解析成功率"]["mean"] * 100
    accuracy_mean = stat_df["解析正确率"]["mean"] * 100
    x = np.arange(len(success_mean.index))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, success_mean.values, width, label='解析成功率', color='#2E8B57')
    bars2 = ax1.bar(x + width / 2, accuracy_mean.values, width, label='参数匹配正确率', color='#FF8C00')

    ax1.set_ylabel('百分比 (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) LLM解析整体性能', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(success_mean.index, rotation=15)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=9)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9
            )

    # 子图2：参数匹配F1-Score
    ax2 = fig.add_subplot(gs[0, 1])
    f1_mean = stat_df["参数匹配F1"]["mean"]  # 与parse_command字段名完全一致
    bars3 = ax2.bar(f1_mean.index, f1_mean.values, color='#4682B4', width=0.6)
    ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 参数匹配F1-Score', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=15)

    for bar, mean in zip(bars3, f1_mean.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9
        )

    # 子图3：响应时间分布
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(
        x="模型名称",
        y="响应时间(s)",
        data=df,
        ax=ax3,
        hue="模型名称",
        palette="Set2",
        linewidth=1.2,
        legend=False
    )
    ax3.set_xlabel('模型', fontsize=11, fontweight='bold')
    ax3.set_ylabel('响应时间 (秒)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 响应时间分布', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=15)

    # 子图4：各参数正确率对比
    ax4 = fig.add_subplot(gs[1, 0])
    params = ["area", "drone_num", "path", "has_obstacle", "obstacle_position", "height"]
    param_accuracy = []
    for model in stat_df.index:
        model_df = df[df["模型名称"] == model]
        acc = [model_df[f"{p}_正确率"].mean() * 100 for p in params]
        param_accuracy.append(acc)

    x = np.arange(len(params))
    width = 0.25
    for i, (model, acc) in enumerate(zip(stat_df.index, param_accuracy)):
        ax4.bar(x + i * width, acc, width, label=model)

    ax4.set_ylabel('正确率 (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) 各参数解析正确率', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(params, rotation=45)
    ax4.set_ylim(0, 110)
    ax4.legend(fontsize=8)

    # 子图5：不同复杂度指令的性能
    ax5 = fig.add_subplot(gs[1, 1])
    complexity_df = df.groupby(["模型名称", "复杂度分级"])["解析正确率"].mean().unstack() * 100
    complexity_df.plot(kind='bar', ax=ax5, rot=15, colormap='viridis', width=0.6)
    ax5.set_ylabel('正确率 (%)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) 不同复杂度指令性能', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 110)
    ax5.legend(fontsize=8)

    # 子图6：错误类型分布
    ax6 = fig.add_subplot(gs[1, 2])
    error_counts = df[df["解析正确率"] == False]["错误类型"].value_counts()
    ax6.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%',
            startangle=90, colors=sns.color_palette("pastel"))
    ax6.set_title('(f) 错误类型分布', fontsize=12, fontweight='bold')

    # 保存图表
    chart_path = Path(config.result_dir) / "llm_parsing_results.png"
    plt.savefig(chart_path)
    logger.info(f"🖼️ 顶刊级可视化图表已保存至: {chart_path}")
    plt.close()
#====================== 主程序（Windows 直接运行） ======================
if __name__ == "__main__":
    # 初始化
    config = load_llm_config()
    logger = setup_llm_logger(config)
    np.random.seed(config.seed)  # 固定随机种子（可复现性）

    # 打印实验信息
    logger.info("=" * 80)
    logger.info("🚀 LLM指令解析顶刊级验证实验启动（Windows版）")
    logger.info(f"📌 测试指令数: {len(config.test_commands)} 条")
    logger.info(f"🔄 每条指令重复次数: {config.repeat_times} 次")
    logger.info(f"🤖 测试模型: {list(config.models.keys())}")
    logger.info(f"📂 结果保存目录: {config.result_dir}")
    logger.info("=" * 80)

    # 执行实验
    try:
        raw_df, stat_df = run_llm_experiment(config, logger)
        plot_llm_results(raw_df, stat_df, config, logger)  # 补全logger参数

        # 打印最终统计报告
        logger.info("\n" + "=" * 80)
        logger.info("📊 LLM指令解析实验最终统计报告（顶刊级）")
        logger.info("=" * 80)
        logger.info("\n1. 整体性能统计（均值±标准差）:")
        logger.info(stat_df.to_string())

        logger.info("\n2. 各参数单独正确率(%):")
        for model in stat_df.index:
            model_df = raw_df[raw_df["模型名称"] == model]
            acc = [model_df[f"{p}_正确率"].mean() * 100 for p in
                   ["area", "drone_num", "path", "has_obstacle", "obstacle_position", "height"]]
            logger.info(
                f"   {model}: {dict(zip(['area', 'drone_num', 'path', 'has_obstacle', 'obstacle_position', 'height'], [round(a, 1) for a in acc]))}")

        logger.info("\n3. 不同复杂度指令正确率(%):")
        complexity_stat = raw_df.groupby(["模型名称", "复杂度分级"])["解析正确率"].mean().unstack() * 100
        logger.info(complexity_stat.to_string())

        logger.info("\n✅ 顶刊级LLM实验执行完成！")
        logger.info("📁 原始数据: llm_experiment_results/llm_experiment_raw_data.csv")
        logger.info("📊 统计结果: llm_experiment_results/llm_experiment_statistics.csv")
        logger.info("🖼️  可视化图表: llm_experiment_results/llm_parsing_results.png")
        logger.info("📜 实验日志: llm_experiment_logs/")
    except Exception as e:
        logger.error(f"❌ 实验执行异常: {str(e)}", exc_info=True)
        raise


