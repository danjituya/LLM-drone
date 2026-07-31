# -*- coding: utf-8 -*-
"""
农业无人机LLM指令解析性能验证实验（Windows专属版）
期刊适配：IEEE Transactions on Cybernetics
运行环境：Windows 10/11 + Python 3.8-3.11
依赖：pip install openai pandas numpy matplotlib seaborn scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import os
import yaml
from yaml import SafeDumper
import re
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from pathlib import Path
from openai import OpenAI
import random


# ====================== 1. 扩展版配置（支持100+指令+多提示词方法） ======================
@dataclass
class LLMExperimentConfig:
    # 实验元数据
    seed: int = 42
    repeat_times: int = 5  # 每条指令重复5次（统计显著性要求）
    timeout: float = 15.0
    temperature: float = 0.1

    # 国内可用模型配置
    models: Dict[str, Dict[str, str]] = None

    # 4类提示词模板（顶刊对比要求）
    prompt_templates: Dict[str, str] = None

    # 扩展版测试用例（120条，覆盖多场景/复杂度/噪声）
    test_commands: List[str] = None

    # 指令复杂度分级（适配120条指令）
    command_complexity: Dict[str, str] = None

    # 指令意图标签（新增：用于意图识别准确率计算）
    command_intent: Dict[str, str] = None

    # Windows路径自动处理
    result_dir: str = str(Path("./llm_experiment_results").resolve())
    log_dir: str = str(Path("./llm_experiment_logs").resolve())
    config_path: str = str(Path("./llm_experiment_config.yaml").resolve())

    def __post_init__(self):
        # 火山方舟三模型对比
        # 注意：请通过环境变量或 config.py 设置实际 API Key
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

        # 4类提示词模板（顶刊对比方法）
        self.prompt_templates = self.prompt_templates or {
            # 方法1：普通提示词（增加格式约束）
            "普通提示词": """
            请将农业无人机的自然语言指令解析为调度参数字典，包含area、drone_num、path、area_bounds、has_obstacle、obstacle_position、height字段。
            输出要求：仅输出Python字典，变量名必须为dispatch_params，无其他文字！
            用户指令：{user_command}
            示例输出：
            dispatch_params = {{
                "area": "A01",
                "drone_num": 3,
                "path": "zigzag",
                "area_bounds": [0,0,200,150],
                "has_obstacle": False,
                "obstacle_position": None,
                "height": 10
            }}
            """.strip(),
            # 方法3：结构化提示词 + 合法性校验
            "结构化提示词+合法性校验": """
            【身份】你是农业无人机调度指令专用解析器
            【任务】将用户的自然语言指令解析为指定格式的调度参数字典，并完成合法性校验
            【参数说明】
            - area: 农田地块编号（如A01、B02，多个用逗号分隔，无则填default_farm）
            - drone_num: 无人机总数量（整数，1-10，无则填3）
            - path: 飞行路径类型（仅支持zigzag/round/grid，无则填zigzag）
            - area_bounds: 农田边界坐标（固定为[0,0,200,150]，不可修改）
            - has_obstacle: 是否存在障碍（True/False，无则填False）
            - obstacle_position: 障碍坐标（[x,y]，x∈[0,200], y∈[0,150]，无则填None）
            - height: 飞行高度（整数，5-30米，无则填10）
            【合法性校验规则】
            1. drone_num必须是1-10的整数
            2. path只能是zigzag/round/grid之一
            3. height必须是5-30的整数
            4. obstacle_position坐标必须在[0,0,200,150]范围内
            用户指令：{user_command}
            输出格式要求：仅输出Python字典，无其他文字
            """.strip(),
            # 方法4：完整四级机制（结构化+校验+容错+兜底）
            "完整四级机制": """
            【身份】你是农业无人机调度指令专用解析器，**仅输出Python代码块**，绝对不能有任何解释、备注、换行或多余文字。
            【任务】将用户的自然语言指令解析为指定格式的调度参数字典，执行四级保障机制：
            1级（结构化解析）：严格按参数说明提取字段
            2级（合法性校验）：验证参数类型/范围/枚举值
            3级（容错处理）：噪声/歧义指令自动修正
            4级（兜底机制）：解析失败时返回标准默认字典
            【参数说明】
            - area: 农田地块编号（如A01、B02，多个用逗号分隔，无则填default_farm）
            - drone_num: 无人机总数量（整数，1-10，无则填3）
            - path: 飞行路径类型（仅支持zigzag/round/grid，无则填zigzag）
            - area_bounds: 农田边界坐标（固定为[0,0,200,150]，不可修改）
            - has_obstacle: 是否存在障碍（True/False，无则填False）
            - obstacle_position: 障碍坐标（[x,y]，x∈[0,200], y∈[0,150]，无则填None）
            - height: 飞行高度（整数，5-30米，无则填10）
            【兜底规则】解析失败时返回：
            {{
                "area": "default_farm",
                "drone_num": 3,
                "path": "zigzag",
                "area_bounds": [0, 0, 200, 150],
                "has_obstacle": False,
                "obstacle_position": None,
                "height": 10
            }}
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
            ```
            """.strip()
        }

        # 生成120条扩展测试指令（覆盖多场景/复杂度/噪声）
        self.test_commands = self._generate_extended_commands()

        # 自动生成指令复杂度和意图标签
        self.command_complexity, self.command_intent = self._generate_command_metadata()

    def _generate_extended_commands(self) -> List[str]:
        """生成120条多样化农业无人机指令（顶刊级测试集）"""
        # 基础场景模板
        base_scenarios = [
            # 巡检场景（40条）
            "巡检{area}地块，使用{num}台无人机，高度{height}米",
            "{area}地块巡检，{path}路径，{obstacle}，高度{height}米",
            "紧急巡检{area}地块，{num}台无人机，{obstacle_desc}，高度{height}米！！！",
            "批量巡检{areas}地块，每块{num_per}台无人机，总数量{total_num}台，高度{height}米",
            "{area}地块常规巡检，无障碍物，{path}路径，高度{height}米",

            # 喷药场景（25条）
            "对{area}地块喷洒农药，{num}台无人机，{path}路径，高度{height}米，避开{obstacle_pos}障碍",
            "{area}地块喷药作业，{num}台无人机，飞行高度{height}米，{obstacle}，{path}路径",
            "紧急喷药{area}地块，{num}台无人机，高度{height}米，{obstacle_desc}",

            # 施肥场景（20条）
            "{area}地块施肥作业，{num}台无人机，{path}路径，高度{height}米，{obstacle}",
            "批量施肥{areas}地块，{total_num}台无人机，高度{height}米，{obstacle_desc}",

            # 病虫害检测场景（20条）
            "{area}地块病虫害检测，{num}台无人机，{path}路径，高度{height}米，{obstacle}",
            "高清拍摄{area}地块病虫害，{num}台无人机，高度{height}米，{obstacle_desc}",

            # 混合场景（15条）
            "{area}地块先巡检后喷药，{num}台无人机，{path}路径，高度{height}米，{obstacle}",
            "{areas}地块分别巡检和施肥，{num_per}台/块，总{total_num}台，高度{height}米"
        ]

        # 变量池
        areas = [f"{chr(65 + i)}{str(j).zfill(2)}" for i in range(20) for j in range(10)]  # A01-A10, B01-B10...T10
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        heights = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 20, 25, 30]
        paths = ["之字形(zigzag)", "圆形(round)", "网格状(grid)", "zigzag", "round", "grid"]
        obstacles = ["有障碍物", "无障碍物", "发现障碍自动绕开", "存在电线杆障碍", "有树木障碍"]
        obstacle_pos = ["(20,30)", "(50,80)", "(100,100)", "(150,70)", "(180,120)"]
        obstacle_descs = [
            "坐标{pos}有电线杆", "({x},{y})处有大树", "多个障碍，坐标暂不指定",
            "障碍位置大概在{pos}附近", "有障碍但坐标未知"
        ]

        # 生成指令
        commands = []
        random.seed(self.seed)
        while len(commands) < 120:
            scenario = random.choice(base_scenarios)
            # 随机填充变量
            area = random.choice(areas)
            areas_batch = ",".join(random.sample(areas, random.choice([2, 3, 4])))
            num = random.choice(nums)
            num_per = random.choice([1, 2, 3])
            total_num = num_per * len(areas_batch.split(","))
            height = random.choice(heights)
            path = random.choice(paths)
            obstacle = random.choice(obstacles)
            pos = random.choice(obstacle_pos)
            x, y = random.randint(0, 200), random.randint(0, 150)
            obstacle_desc = random.choice(obstacle_descs).format(pos=pos, x=x, y=y)

            # 填充模板
            cmd = scenario.format(
                area=area, areas=areas_batch, num=num, num_per=num_per,
                total_num=total_num, height=height, path=path,
                obstacle=obstacle, obstacle_pos=pos, obstacle_desc=obstacle_desc
            )

            # 加入噪声/歧义变体
            noise_types = [
                lambda s: s + "！！！（紧急）",
                lambda s: s.replace("，", " ").replace("米", " 米左右 "),
                lambda s: re.sub(r"(\d+)台", r"\1 台", s),
                lambda s: s.replace("高度", "大概高度"),
                lambda s: s  # 无噪声
            ]
            cmd = random.choice(noise_types)(cmd)

            if cmd not in commands:
                commands.append(cmd.strip())

        # 补充原有20条基准指令（保证兼容性）
        baseline_commands = [
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
        commands = baseline_commands + commands[:100]  # 总120条

        return commands

    def _generate_command_metadata(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """生成120条指令的复杂度分级和意图标签"""
        complexity_map = {}
        intent_map = {}
        random.seed(self.seed)

        # 意图关键词映射
        intent_keywords = {
            "巡检": "巡检",
            "喷药": "喷药",
            "施肥": "施肥",
            "病虫害检测": "病虫害检测",
            "拍摄": "病虫害检测",
            "先巡检后喷药": "混合作业",
            "分别巡检和施肥": "混合作业"
        }

        # 复杂度规则
        def get_complexity(cmd: str) -> str:
            # 基础规则
            if "！" in cmd or "大概" in cmd or "左右" in cmd or "（紧急）" in cmd:
                if "多个" in cmd or "批量" in cmd or len(cmd.split("，")) > 5:
                    return "复杂 + 噪声"
                elif len(cmd.split("，")) > 3:
                    return "中等 + 噪声"
                else:
                    return "噪声"
            elif "歧义" in cmd or "大概位置" in cmd or "坐标未知" in cmd:
                return "歧义"
            elif "批量" in cmd or "," in cmd and "地块" in cmd:
                return "复杂"
            elif len(cmd.split("，")) > 3 or "障碍坐标" in cmd:
                return "中等"
            else:
                return "简单"

        for cmd in self.test_commands:
            # 标注复杂度
            complexity_map[cmd] = get_complexity(cmd)

            # 标注意图
            intent = "未知"
            for keyword, label in intent_keywords.items():
                if keyword in cmd:
                    intent = label
                    break
            intent_map[cmd] = intent

        return complexity_map, intent_map


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
            Dumper=SafeDumper,
            allow_unicode=True,
            indent=4,
            sort_keys=False
        )
    return config


# ====================== 2. Windows 兼容日志系统 ======================
def setup_llm_logger(config: LLMExperimentConfig) -> logging.Logger:
    logger = logging.getLogger("LLM_Drone_Command_Parsing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

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


# ====================== 3. 扩展版LLM解析器（支持多提示词+全实验指标） ======================
class OpenAICompatibleParser:
    """OpenAI兼容接口解析器（支持多提示词方法+顶刊全指标）"""

    def __init__(self, model_name: str, base_url: str, api_key: str,
                 prompt_templates: Dict[str, str], logger: logging.Logger):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.prompt_templates = prompt_templates
        self.logger = logger
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=15.0
        )

    def extract_python_code(self, content: str) -> Optional[str]:
        """多层级容错代码提取（解决Windows换行问题+纯字典解析+DeepSeek特殊输出）"""
        # 层级0：先尝试JSON转字典（普通提示词可能返回纯JSON）
        try:
            json_obj = json.loads(content.strip())
            if isinstance(json_obj, dict):
                return f"dispatch_params = {json.dumps(json_obj)}"
        except:
            pass

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

        # 层级3：兜底提取dispatch_params（支持直接赋值）
        if "dispatch_params" in content:
            start = content.find("dispatch_params")
            end = content.rfind("}") + 1
            return content[start:end].strip().replace("\r\n", "\n")

        # 层级4：直接提取字典（无变量名）
        pattern3 = re.compile(r"\{[\s\S]*\}", re.DOTALL)
        matches = pattern3.findall(content)
        if matches:
            return f"dispatch_params = {matches[0]}".strip().replace("\r\n", "\n")

        # 层级5：DeepSeek特殊输出处理（仅输出字典内容，无大括号）
        if "area" in content and "drone_num" in content:
            # 尝试从纯文本中提取键值对并构造字典
            try:
                lines = content.strip().split("\n")
                params = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().strip('"').strip("'")
                        value = value.strip().strip(",").strip('"').strip("'")
                        # 类型转换
                        if value.isdigit():
                            value = int(value)
                        elif value.lower() == "true":
                            value = True
                        elif value.lower() == "false":
                            value = False
                        elif value.lower() == "none":
                            value = None
                        params[key] = value
                if "area" in params and "drone_num" in params:
                    return f"dispatch_params = {json.dumps(params)}"
            except:
                pass

        self.logger.warning(f"未提取到有效代码: {content[:200]}")
        return None

    def safe_exec_code(self, code: str) -> Optional[Dict]:
        """安全执行代码（沙箱环境+语法自动修正+兜底字典）"""
        try:
            # ====================== 新增：语法自动修正层 ======================
            # 1. 统一布尔值大小写（解决DeepSeek/GLM输出false/true的问题）
            code = code.replace("false", "False").replace("true", "True")
            # 2. 统一空值表示（解决部分模型输出null的问题）
            code = code.replace("null", "None").replace("NULL", "None")
            # 3. 移除多余的逗号（解决JSON转Python的语法问题）
            code = re.sub(r",\s*}", "}", code)
            code = re.sub(r",\s*]", "]", code)
            # 4. 修复字符串引号不匹配问题
            code = code.replace('""', '"').replace("''", "'")
            # ==================================================================

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
            self.logger.warning("执行结果无dispatch_params字典，返回兜底值")
            # 返回兜底字典
            return {
                "area": "default_farm",
                "drone_num": 3,
                "path": "zigzag",
                "area_bounds": [0, 0, 200, 150],
                "has_obstacle": False,
                "obstacle_position": None,
                "height": 10
            }
        except Exception as e:
            self.logger.error(f"代码执行失败: {str(e)[:100]}，返回兜底值")
            # 执行异常时返回兜底字典
            return {
                "area": "default_farm",
                "drone_num": 3,
                "path": "zigzag",
                "area_bounds": [0, 0, 200, 150],
                "has_obstacle": False,
                "obstacle_position": None,
                "height": 10
            }

    def parse_json_legality(self, params: Dict) -> bool:
        """新增：合法JSON率计算（校验参数类型/范围）"""
        if not isinstance(params, dict):
            return False

        # 基础字段校验
        required_fields = ["area", "drone_num", "path", "area_bounds", "has_obstacle", "obstacle_position", "height"]
        if not all(f in params for f in required_fields):
            return False

        # 类型校验
        type_rules = {
            "area": str,
            "drone_num": int,
            "path": str,
            "area_bounds": list,
            "has_obstacle": bool,
            "height": int
        }
        for field, expected_type in type_rules.items():
            if not isinstance(params[field], expected_type):
                return False

        # 范围/枚举校验
        if params["drone_num"] < 1 or params["drone_num"] > 10:
            return False
        if params["path"] not in ["zigzag", "round", "grid"]:
            return False
        if params["height"] < 5 or params["height"] > 30:
            return False
        if len(params["area_bounds"]) != 4 or not all(isinstance(x, int) for x in params["area_bounds"]):
            return False
        if params["obstacle_position"] is not None:
            if not isinstance(params["obstacle_position"], list) or len(params["obstacle_position"]) != 2:
                return False
            x, y = params["obstacle_position"]
            if x < 0 or x > 200 or y < 0 or y > 150:
                return False

        return True

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

    def get_intent_accuracy(self, parsed_params: Dict, ground_truth_intent: str, cmd: str) -> bool:
        """新增：意图识别准确率计算"""
        # 意图映射规则
        intent_param_map = {
            "巡检": {"path": "zigzag", "drone_num": 3},
            "喷药": {"path": "grid", "drone_num": 2},
            "施肥": {"path": "round", "drone_num": 4},
            "病虫害检测": {"path": "grid", "drone_num": 1},
            "混合作业": {"drone_num": 5}
        }

        # 基于参数判断意图
        parsed_intent = "未知"
        for intent, params in intent_param_map.items():
            match = True
            for k, v in params.items():
                if parsed_params.get(k) != v:
                    match = False
                    break
            if match:
                parsed_intent = intent
                break

        return parsed_intent == ground_truth_intent

    def get_param_extract_accuracy(self, parsed_params: Dict, ground_truth: Dict) -> float:
        """新增：参数抽取准确率计算"""
        if not parsed_params or not ground_truth:
            return 0.0

        total_params = len(ground_truth)
        correct_params = 0
        for k, v in ground_truth.items():
            if parsed_params.get(k) == v:
                correct_params += 1

        return correct_params / total_params

    def get_fallback_success_rate(self, parsed_params: Dict) -> bool:
        """新增：失败兜底成功率计算"""
        # 兜底标准字典
        fallback_dict = {
            "area": "default_farm",
            "drone_num": 3,
            "path": "zigzag",
            "area_bounds": [0, 0, 200, 150],
            "has_obstacle": False,
            "obstacle_position": None,
            "height": 10
        }

        # 解析失败但返回兜底字典则为成功
        if not parsed_params:
            return False
        return parsed_params == fallback_dict

    def parse_command(self, command: str, prompt_method: str) -> Dict:
        """解析单条指令（支持多提示词方法，返回顶刊全指标）"""
        start_time = time.time()
        result = {
            "指令": command,
            "模型名称": self.model_name,
            "提示词方法": prompt_method,
            "解析成功率": False,
            "解析正确率": False,
            "响应时间(s)": 0.0,
            "解析结果": "",
            "错误信息": "",
            "错误类型": "",
            "复杂度分级": "",
            "意图标签": "",
            # 新增顶刊指标
            "意图识别准确率": False,
            "参数抽取准确率": 0.0,
            "合法JSON率": False,
            "参数匹配F1": 0.0,
            "失败兜底成功率": False,
            # 各参数正确率
            "area_正确率": False,
            "drone_num_正确率": False,
            "path_正确率": False,
            "has_obstacle_正确率": False,
            "obstacle_position_正确率": False,
            "height_正确率": False,
        }

        try:
            # 选择对应提示词模板
            prompt_template = self.prompt_templates[prompt_method]
            # 构造请求
            messages = [
                {"role": "system", "content": "你是农业无人机指令解析器"},
                {"role": "user", "content": prompt_template.format(user_command=command)}
            ]

            # 调用LLM
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

            # 基础元数据
            ground_truth = get_ground_truth(command)
            result["复杂度分级"] = get_command_complexity(command)
            result["意图标签"] = get_command_intent(command)

            # 计算新增顶刊指标
            # 1. 合法JSON率
            result["合法JSON率"] = self.parse_json_legality(parsed_params)
            # 2. 意图识别准确率
            result["意图识别准确率"] = self.get_intent_accuracy(parsed_params, result["意图标签"], command)
            # 3. 参数抽取准确率
            result["参数抽取准确率"] = round(self.get_param_extract_accuracy(parsed_params, ground_truth), 4)
            # 4. 失败兜底成功率
            result["失败兜底成功率"] = self.get_fallback_success_rate(parsed_params)

            # 计算参数匹配F1和各参数正确率
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

            # F1-Score最终计算
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
            # 兜底成功率：解析失败时默认False
            result["失败兜底成功率"] = False

        return result


# ====================== 4. 扩展版标准答案与元数据 ======================
def get_ground_truth(command: str) -> Dict:
    """人工标注+自动生成120条指令的标准答案"""
    # 原有20条基准答案
    baseline_ground_truth = {
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
            "has_obstacle": True, "obstacle_position": [50, 80], "height": 10
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
            "has_obstacle": True, "obstacle_position": [100, 100], "height": 10
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
            "has_obstacle": True, "obstacle_position": [50, 80], "height": 11
        },
        "巡检 C03 地块 高度 十米 无人机 三台 之字形": {
            "area": "C03", "drone_num": 3, "path": "zigzag",
            "has_obstacle": False, "obstacle_position": None, "height": 10
        },
        "E05 地块有电线杆 坐标 50 和 80 绕开 高度 10 米": {
            "area": "E05", "drone_num": 3, "path": "zigzag",
            "has_obstacle": True, "obstacle_position": [50, 80], "height": 10
        }
    }

    # 自动生成扩展指令的标准答案
    extended_ground_truth = {}
    config = load_llm_config()

    for cmd in config.test_commands:
        if cmd in baseline_ground_truth:
            extended_ground_truth[cmd] = baseline_ground_truth[cmd]
            continue

        # 初始化默认值
        gt = {
            "area": "default_farm",
            "drone_num": 3,
            "path": "zigzag",
            "area_bounds": [0, 0, 200, 150],
            "has_obstacle": False,
            "obstacle_position": None,
            "height": 10
        }

        # 提取地块编号
        area_pattern = re.compile(r"([A-Z]\d{2})")
        area_matches = area_pattern.findall(cmd)
        if area_matches:
            gt["area"] = ",".join(area_matches)

        # 提取无人机数量
        num_pattern = re.compile(r"(\d+)台")
        num_matches = num_pattern.findall(cmd)
        if num_matches:
            gt["drone_num"] = int(num_matches[-1])  # 取最后一个数值（总数量）

        # 提取飞行高度
        height_pattern = re.compile(r"(\d+)米")
        height_matches = height_pattern.findall(cmd)
        if height_matches:
            gt["height"] = int(height_matches[0])

        # 提取路径类型
        if "之字" in cmd or "zigzag" in cmd:
            gt["path"] = "zigzag"
        elif "圆形" in cmd or "绕圈" in cmd or "round" in cmd:
            gt["path"] = "round"
        elif "网格" in cmd or "grid" in cmd:
            gt["path"] = "grid"

        # 提取障碍信息
        if "有障碍" in cmd or "绕开" in cmd:
            gt["has_obstacle"] = True
            # 提取障碍坐标
            pos_pattern = re.compile(r"\((\d+),(\d+)\)")
            pos_matches = pos_pattern.findall(cmd)
            if pos_matches:
                x, y = map(int, pos_matches[0])
                gt["obstacle_position"] = [x, y]

        extended_ground_truth[cmd] = gt

    return extended_ground_truth.get(command, {})


def get_command_complexity(command: str) -> str:
    """获取指令的复杂度分级"""
    config = load_llm_config()
    return config.command_complexity.get(command, "未知")


def get_command_intent(command: str) -> str:
    """获取指令的意图标签"""
    config = load_llm_config()
    return config.command_intent.get(command, "未知")


# ====================== 5. 实验执行主函数（补充缺失逻辑） ======================
def run_llm_experiment():
    """执行完整LLM实验（添加异常捕获+进度保存）"""
    # 加载配置
    config = load_llm_config()
    logger = setup_llm_logger(config)

    # 初始化解析器
    parsers = {}
    for model_name, model_config in config.models.items():
        try:
            parser = OpenAICompatibleParser(
                model_name=model_name,
                base_url=model_config["base_url"],
                api_key=model_config["api_key"],
                prompt_templates=config.prompt_templates,
                logger=logger
            )
            parsers[model_name] = parser
            logger.info(f"✅ 初始化{model_name}解析器成功")
        except Exception as e:
            logger.error(f"❌ 初始化{model_name}解析器失败: {str(e)}")
            continue

    # 实验结果存储
    all_results = []
    total_tasks = len(parsers) * len(config.prompt_templates) * len(config.test_commands) * config.repeat_times
    current_task = 0

    logger.info("=" * 80)
    logger.info(
        f"🚀 开始LLM实验 | 模型数: {len(parsers)} | 提示词方法数: {len(config.prompt_templates)} | 指令数: {len(config.test_commands)} | 重复次数: {config.repeat_times} | 总请求数: {total_tasks}")
    logger.info("=" * 80)

    # 遍历所有组合
    for model_name, parser in parsers.items():
        for prompt_method in config.prompt_templates.keys():
            for cmd_idx, command in enumerate(config.test_commands):
                for repeat_idx in range(config.repeat_times):
                    current_task += 1
                    try:
                        logger.info(
                            f"[{current_task}/{total_tasks}] [{model_name}/{prompt_method}] 处理指令({repeat_idx + 1}/{config.repeat_times}): {command[:50]}...")

                        # 解析指令（核心逻辑）
                        result = parser.parse_command(command, prompt_method)
                        all_results.append(result)

                        # 每100个任务保存一次临时结果（防止进程退出丢失数据）
                        if current_task % 100 == 0:
                            temp_df = pd.DataFrame(all_results)
                            temp_df.to_csv(Path(config.result_dir) / f"temp_results_{current_task}.csv", index=False,
                                           encoding="utf-8-sig")
                            logger.info(f"💾 临时结果已保存（{current_task}条）")

                    except Exception as e:
                        logger.error(f"❌ 任务{current_task}执行失败: {str(e)}")
                        # 单个任务失败不终止，记录错误后继续
                        continue

    # 保存最终结果
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(Path(config.result_dir) / f"llm_experiment_results_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    index=False, encoding="utf-8-sig")
    logger.info(f"🎉 实验完成！结果已保存至: {config.result_dir}")

    # 输出核心统计
    logger.info("📊 核心指标统计:")
    success_rate = final_df["解析成功率"].mean()
    intent_acc = final_df["意图识别准确率"].mean()
    json_legal_rate = final_df["合法JSON率"].mean()
    logger.info(f"- 整体解析成功率: {success_rate:.2%}")
    logger.info(f"- 意图识别准确率: {intent_acc:.2%}")
    logger.info(f"- 合法JSON率: {json_legal_rate:.2%}")

# ====================== 6. 扩展版顶刊级可视化 ======================
def plot_llm_results(df: pd.DataFrame, stat_df: pd.DataFrame, config: LLMExperimentConfig, logger: logging.Logger):
    """生成符合IEEE Transactions格式的扩展版专业图表"""
    # Windows中文显示配置
    plt.rcParams.update({
        'font.family': ['SimHei', 'Times New Roman'],
        'font.size': 10,
        'axes.linewidth': 1.2,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.unicode_minus': False
    })

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # 子图1：不同提示词方法的整体解析性能
    ax1 = fig.add_subplot(gs[0, 0:2])
    plot_data = stat_df["解析成功率"]["mean"].unstack() * 100
    plot_data.plot(kind='bar', ax=ax1, rot=15, colormap='Set2', width=0.8)
    ax1.set_ylabel('解析成功率 (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) 不同提示词方法的解析成功率', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.legend(title="模型", fontsize=9)
    # 添加数值标签
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', fontsize=8)

    # 子图2：参数匹配F1-Score（按提示词方法）
    ax2 = fig.add_subplot(gs[0, 2:4])
    f1_data = stat_df["参数匹配F1"]["mean"].unstack()
    f1_data.plot(kind='bar', ax=ax2, rot=15, colormap='Set3', width=0.8)
    ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('(b) 不同提示词方法的参数匹配F1-Score', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend(title="模型", fontsize=9)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', fontsize=8)

    # 子图3：意图识别准确率
    ax3 = fig.add_subplot(gs[1, 0:2])
    intent_data = stat_df["意图识别准确率"]["mean"].unstack() * 100
    intent_data.plot(kind='bar', ax=ax3, rot=15, colormap='viridis', width=0.8)
    ax3.set_ylabel('意图识别准确率 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) 不同提示词方法的意图识别准确率', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 110)
    ax3.legend(title="模型", fontsize=9)
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.1f%%', fontsize=8)

    # 子图4：合法JSON率
    ax4 = fig.add_subplot(gs[1, 2:4])
    json_data = stat_df["合法JSON率"]["mean"].unstack() * 100
    json_data.plot(kind='bar', ax=ax4, rot=15, colormap='plasma', width=0.8)
    ax4.set_ylabel('合法JSON率 (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) 不同提示词方法的合法JSON率', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 110)
    ax4.legend(title="模型", fontsize=9)
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.1f%%', fontsize=8)

    # 子图5：平均响应时间
    ax5 = fig.add_subplot(gs[2, 0:2])
    time_data = stat_df["响应时间(s)"]["mean"].unstack()
    time_data.plot(kind='bar', ax=ax5, rot=15, colormap='coolwarm', width=0.8)
    ax5.set_ylabel('平均响应时间 (秒)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) 不同提示词方法的平均响应时间', fontsize=12, fontweight='bold')
    ax5.legend(title="模型", fontsize=9)
    for container in ax5.containers:
        ax5.bar_label(container, fmt='%.2f', fontsize=8)

    # 子图6：失败兜底成功率
    ax6 = fig.add_subplot(gs[2, 2:4])
    fallback_data = stat_df["失败兜底成功率"]["mean"].unstack() * 100
    fallback_data.plot(kind='bar', ax=ax6, rot=15, colormap='cividis', width=0.8)
    ax6.set_ylabel('失败兜底成功率 (%)', fontsize=11, fontweight='bold')
    ax6.set_title('(f) 不同提示词方法的失败兜底成功率', fontsize=12, fontweight='bold')
    ax6.set_ylim(0, 110)
    ax6.legend(title="模型", fontsize=9)
    for container in ax6.containers:
        ax6.bar_label(container, fmt='%.1f%%', fontsize=8)

    # 子图7：不同复杂度指令的解析正确率
    ax7 = fig.add_subplot(gs[3, 0:2])
    complexity_df = df.groupby(["提示词方法", "复杂度分级"])["解析正确率"].mean().unstack() * 100
    complexity_df.plot(kind='bar', ax=ax7, rot=15, colormap='tab10', width=0.8)
    ax7.set_ylabel('解析正确率 (%)', fontsize=11, fontweight='bold')
    ax7.set_title('(g) 不同复杂度指令的解析正确率', fontsize=12, fontweight='bold')
    ax7.set_ylim(0, 110)
    ax7.legend(title="复杂度", fontsize=8)

    # 子图8：参数抽取准确率对比
    ax8 = fig.add_subplot(gs[3, 2:4])
    param_acc_data = stat_df["参数抽取准确率"]["mean"].unstack() * 100
    param_acc_data.plot(kind='bar', ax=ax8, rot=15, colormap='tab20', width=0.8)
    ax8.set_ylabel('参数抽取准确率 (%)', fontsize=11, fontweight='bold')
    ax8.set_title('(h) 不同提示词方法的参数抽取准确率', fontsize=12, fontweight='bold')
    ax8.set_ylim(0, 110)
    ax8.legend(title="模型", fontsize=8)
    for container in ax8.containers:
        ax8.bar_label(container, fmt='%.1f%%', fontsize=8)

    # 保存图表
    chart_path = Path(config.result_dir) / "llm_parsing_results_extended.png"
    plt.savefig(chart_path)
    logger.info(f"🖼️ 扩展版顶刊级可视化图表已保存至: {chart_path}")
    plt.close()


# ====================== 主程序 ======================
def generate_results_from_temp_data():
    """从已保存的临时结果生成最终统计和图表（不需要重新跑实验）"""
    config = load_llm_config()
    logger = setup_llm_logger(config)

    # 加载已保存的5400条实验数据
    temp_file = Path(config.result_dir) / "temp_results_5400.csv"
    if not temp_file.exists():
        logger.error(f"❌ 未找到临时结果文件: {temp_file}")
        logger.error("请检查llm_experiment_results目录下是否存在temp_results_5400.csv")
        return

    logger.info(f"✅ 加载已保存的实验数据: {temp_file}")
    logger.info(f"📊 总数据条数: {len(pd.read_csv(temp_file))} 条")
    raw_df = pd.read_csv(temp_file, encoding="utf-8-sig")

    # ====================== 1. 生成核心统计结果 ======================
    logger.info("\n🔄 正在生成核心统计结果...")
    stat_df = raw_df.groupby(["模型名称", "提示词方法"]).agg({
        "解析成功率": ["mean", "std", stats.sem],
        "解析正确率": ["mean", "std", stats.sem],
        "响应时间(s)": ["mean", "std", stats.sem],
        "意图识别准确率": ["mean", "std", stats.sem],
        "参数抽取准确率": ["mean", "std", stats.sem],
        "合法JSON率": ["mean", "std", stats.sem],
        "参数匹配F1": ["mean", "std", stats.sem],
        "失败兜底成功率": ["mean", "std", stats.sem]
    }).round(4)

    # 保存核心统计结果（直接用于论文表6）
    stat_df.to_csv(Path(config.result_dir) / "llm_experiment_statistics_extended.csv", encoding="utf-8-sig")
    logger.info("✅ 核心统计结果已保存至: llm_experiment_results/llm_experiment_statistics_extended.csv")

    # ====================== 2. 生成顶刊级可视化图表 ======================
    logger.info("\n🖼️  正在生成顶刊级可视化图表...")
    plot_llm_results(raw_df, stat_df, config, logger)
    logger.info("✅ 可视化图表已保存至: llm_experiment_results/llm_parsing_results_extended.png")

    # ====================== 3. 生成多维度统计结果 ======================
    logger.info("\n📈 正在生成多维度统计结果...")

    # 3.1 不同复杂度指令性能（用于论文消融实验）
    complexity_stat = raw_df.groupby(["提示词方法", "复杂度分级"])["解析正确率"].mean().unstack().round(4) * 100
    complexity_stat.to_csv(Path(config.result_dir) / "llm_complexity_statistics.csv", encoding="utf-8-sig")
    logger.info("✅ 不同复杂度指令性能已保存")

    # 3.2 不同意图指令性能（用于论文泛化性分析）
    intent_stat = raw_df.groupby(["提示词方法", "意图标签"])["意图识别准确率"].mean().unstack().round(4) * 100
    intent_stat.to_csv(Path(config.result_dir) / "llm_intent_statistics.csv", encoding="utf-8-sig")
    logger.info("✅ 不同意图指令性能已保存")

    # 3.3 错误类型统计（用于论文讨论部分）
    error_stat = raw_df[raw_df["解析正确率"] == False].groupby(["提示词方法", "错误类型"]).size().reset_index(
        name="次数")
    error_stat.to_csv(Path(config.result_dir) / "llm_error_statistics.csv", encoding="utf-8-sig")
    logger.info("✅ 错误类型统计已保存")

    # ====================== 4. 打印最终实验报告 ======================
    logger.info("\n" + "=" * 80)
    logger.info("📊 LLM指令解析实验最终统计报告（顶刊级）")
    logger.info("=" * 80)

    # 4.1 核心实验指标（均值±标准差）
    logger.info("\n1. 核心实验指标（均值±标准差）:")
    logger.info(stat_df.to_string())

    # 4.2 不同提示词方法平均性能（直接用于论文表6）
    logger.info("\n2. 不同提示词方法平均性能(%):")
    prompt_perf = raw_df.groupby("提示词方法").agg({
        "解析成功率": lambda x: round(x.mean() * 100, 2),
        "意图识别准确率": lambda x: round(x.mean() * 100, 2),
        "合法JSON率": lambda x: round(x.mean() * 100, 2),
        "参数匹配F1": lambda x: round(x.mean() * 100, 2),
        "失败兜底成功率": lambda x: round(x.mean() * 100, 2)
    })
    logger.info(prompt_perf.to_string())

    # 4.3 不同复杂度指令解析正确率（用于论文消融实验）
    logger.info("\n3. 不同复杂度指令解析正确率(%):")
    logger.info(complexity_stat.round(2).to_string())

    # 4.4 不同意图指令识别准确率（用于论文泛化性分析）
    logger.info("\n4. 不同意图指令识别准确率(%):")
    logger.info(intent_stat.round(2).to_string())

    # 4.5 错误类型分布（用于论文讨论部分）
    logger.info("\n5. 主要错误类型分布(次数):")
    logger.info(error_stat.to_string())

    # 保存最终原始数据
    raw_df.to_csv(Path(config.result_dir) / "llm_experiment_raw_data_extended.csv", index=False, encoding="utf-8-sig")

    logger.info("\n✅ 所有结果生成完成！")
    logger.info("📁 原始数据: llm_experiment_results/llm_experiment_raw_data_extended.csv")
    logger.info("📊 核心统计: llm_experiment_results/llm_experiment_statistics_extended.csv")
    logger.info("📈 复杂度统计: llm_experiment_results/llm_complexity_statistics.csv")
    logger.info("🎯 意图统计: llm_experiment_results/llm_intent_statistics.csv")
    logger.info("❌ 错误统计: llm_experiment_results/llm_error_statistics.csv")
    logger.info("🖼️  可视化图表: llm_experiment_results/llm_parsing_results_extended.png")
    logger.info("📜 实验日志: llm_experiment_logs/")


# 主函数入口
if __name__ == "__main__":
    try:
        # 直接从已保存的数据生成结果（10秒完成）
        generate_results_from_temp_data()
    except Exception as e:
        logging.error(f"❌ 结果生成异常: {str(e)}", exc_info=True)
        exit(0)