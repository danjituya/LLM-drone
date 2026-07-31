# -*- coding: utf-8 -*-
# @File    : agriculture_drone_agent.py
# @Describe: 多智能体农业无人机核心Agent - 优化版
import os
import re
import json
import time
from openai import OpenAI
from drone_scheduler import DroneScheduler
from disease_detector import DiseaseDetector
from vla_controller import Vlacontroller
from config import (
    BASE_URL, ARK_API_KEY, MODEL, VLA_MODEL,
    PROMPTS_DIR, REPORT_FOLDER,
    MAX_CHAT_HISTORY_LENGTH, LLM_TIMEOUT
)

# ====================== matplotlib导入加容错 ======================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  未安装matplotlib，部分可视化功能不可用")


# ====================== Web可视化模块 ======================
class WebVisualizer:
    def __init__(self):
        self.output_dir = REPORT_FOLDER
        os.makedirs(self.output_dir, exist_ok=True)

    def show(self, drone_data, disease_result):
        timestamp = int(time.time())
        output_file = os.path.join(self.output_dir, f"report_{timestamp}.json")
        total_images = disease_result["total_images"]
        disease_images = disease_result["disease_images"]
        disease_rate = round(disease_images / total_images * 100, 2) if total_images > 0 else 0
        viz_data = {
            "patrol_id": drone_data["patrol_id"],
            "area": drone_data["area"],
            "drone_num": drone_data["drone_num"],
            "path_viz_path": drone_data["path_viz_path"],
            "disease_summary": {
                "total_images": total_images,
                "disease_images": disease_images,
                "disease_rate": disease_rate
            },
            "disease_detail": disease_result["detail"]
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)
        web_url = f"http://localhost:54321/report/{timestamp}"
        return web_url, output_file


# ====================== 全局模块初始化 ======================
drone_sch = None
disease_detector = None
web_viz = None
vla_ctrl = None

def init_core_modules():
    """统一初始化核心模块"""
    global drone_sch, disease_detector, web_viz, vla_ctrl
    try:
        drone_sch = DroneScheduler(drone_num=3)
    except Exception:
        drone_sch = None
    try:
        disease_detector = DiseaseDetector()
    except Exception:
        disease_detector = None
    try:
        web_viz = WebVisualizer()
    except Exception:
        web_viz = None
    try:
        vla_ctrl = Vlacontroller(
            model_api_key=ARK_API_KEY,
            model_endpoint=BASE_URL,
            vla_model=VLA_MODEL
        )
    except Exception:
        vla_ctrl = None


# ====================== AgricultureDroneAgent 类 ======================
class AgricultureDroneAgent:
    def __init__(self, vla_ctrl=None,
                 system_prompts=None,
                 knowledge_prompt=None,
                 auto_load_knowledge=False):
        self.vla_ctrl = vla_ctrl
        self.client = OpenAI(base_url=BASE_URL, api_key=ARK_API_KEY)
        self.chat_history = []

        if system_prompts is None:
            system_prompts = os.path.join(PROMPTS_DIR, "agri_drone_cn.txt")
        if knowledge_prompt is None:
            knowledge_prompt = os.path.join(PROMPTS_DIR, "agri_drone_knowledge.txt")

        if os.path.exists(system_prompts):
            try:
                with open(system_prompts, "r", encoding="utf8") as f:
                    sys_prompt = f.read()
                self.chat_history.append({"role": "system", "content": sys_prompt})
            except Exception:
                pass

        if auto_load_knowledge and os.path.exists(knowledge_prompt):
            try:
                with open(knowledge_prompt, "r", encoding="utf8") as f:
                    kg_prompt = f.read()
                self.ask(kg_prompt)
            except Exception:
                pass

    def _trim_chat_history(self):
        """修剪对话历史"""
        system_messages = [msg for msg in self.chat_history if msg["role"] == "system"]
        other_messages = [msg for msg in self.chat_history if msg["role"] != "system"]
        if len(other_messages) > MAX_CHAT_HISTORY_LENGTH * 2:
            other_messages = other_messages[-MAX_CHAT_HISTORY_LENGTH * 2:]
            self.chat_history = system_messages + other_messages

    def ask(self, prompt):
        try:
            self.chat_history.append({"role": "user", "content": prompt})
            self._trim_chat_history()
            completion = self.client.chat.completions.create(
                model=MODEL,
                messages=self.chat_history,
                temperature=0.1,
                timeout=LLM_TIMEOUT
            )
            content = completion.choices[0].message.content
            self.chat_history.append({"role": "assistant", "content": content})
            return content
        except Exception as e:
            if len(self.chat_history) > 0 and self.chat_history[-1]["role"] == "user":
                self.chat_history.pop()
            raise e

    def extract_python_code(self, content):
        """多层级安全代码提取"""
        # 优先尝试JSON解析
        try:
            json_obj = json.loads(content.strip())
            if isinstance(json_obj, dict):
                return f"dispatch_params = {json.dumps(json_obj)}"
        except Exception:
            pass

        # 标准Python代码块
        pattern = re.compile(r"```python(.*?)```", re.DOTALL)
        matches = pattern.findall(content)
        if matches:
            return "\n".join(matches).strip().replace("\r\n", "\n")

        # 无语言标记的代码块
        pattern2 = re.compile(r"```(.*?)```", re.DOTALL)
        matches = pattern2.findall(content)
        if matches:
            return "\n".join(matches).strip().replace("\r\n", "\n")

        # 提取dispatch_params
        if "dispatch_params" in content:
            start = content.find("dispatch_params")
            end = content.rfind("}") + 1
            return content[start:end].strip().replace("\r\n", "\n")

        # 兜底提取字典
        pattern3 = re.compile(r"\{[\s\S]*\}", re.DOTALL)
        matches = pattern3.findall(content)
        if matches:
            return f"dispatch_params = {matches[0]}".strip().replace("\r\n", "\n")

        return None

    def safe_exec_code(self, code):
        """安全执行代码"""
        try:
            code = code.replace("false", "False").replace("true", "True")
            code = code.replace("null", "None").replace("NULL", "None")
            code = re.sub(r",\s*}", "}", code)
            code = re.sub(r",\s*]", "]", code)
            code = code.replace('""', '"').replace("''", "'")

            local_vars = {}
            safe_globals = {
                "__builtins__": {
                    "dict": dict, "list": list, "int": int, "str": str,
                    "bool": bool, "None": None, "True": True, "False": False
                }
            }
            exec(code, safe_globals, local_vars)
            if "dispatch_params" in local_vars and isinstance(local_vars["dispatch_params"], dict):
                return local_vars["dispatch_params"]
            return self._get_default_dispatch_params()
        except Exception:
            return self._get_default_dispatch_params()

    def _get_default_dispatch_params(self):
        """获取默认调度参数"""
        return {
            "area": "default_farm",
            "drone_num": 3,
            "path": "zigzag",
            "area_bounds": [0, 0, 200, 150],
            "has_obstacle": False,
            "obstacle_position": None
        }

    def process(self, command, run_python_code=True):
        if not drone_sch:
            raise Exception("无人机调度器不可用，无法执行巡检")
        if not disease_detector:
            raise Exception("病虫害识别器不可用，无法执行识别")

        llm_response = None
        try:
            enhanced_command = f"""用户指令：{command}

请直接输出以下格式的Python代码，不要任何其他文字：
```python
dispatch_params = {{
    "area": "农田区域",
    "drone_num": 3,
    "path": "zigzag",
    "area_bounds": [0, 0, 200, 150],
    "has_obstacle": True,
    "obstacle_position": [100, 75]
}}
```"""
            llm_response = self.ask(enhanced_command)
        except Exception as e:
            llm_response = None

        dispatch_params = None
        if llm_response and run_python_code:
            python_code = self.extract_python_code(llm_response)
            if python_code:
                dispatch_params = self.safe_exec_code(python_code)

        if not dispatch_params:
            dispatch_params = self._get_default_dispatch_params()

        drone_result = drone_sch.start_patrol(
            area=dispatch_params.get("area", "default_farm"),
            drone_num=dispatch_params.get("drone_num", 3),
            path=dispatch_params.get("path", "zigzag"),
            area_bounds=dispatch_params.get("area_bounds", [0, 0, 200, 150]),
            vla_controller=self.vla_ctrl
        )
        if drone_result["status"] != "success":
            raise Exception(f"无人机巡检失败: {drone_result.get('error', '未知错误')}")

        disease_result = disease_detector.predict(drone_result["data_path"])
        if disease_result["status"] != "success":
            raise Exception(f"病虫害识别失败: {disease_result.get('error', '未知错误')}")

        web_url, report_file = None, None
        if web_viz:
            web_url, report_file = web_viz.show(drone_result, disease_result)

        return {
            "llm_response": llm_response,
            "dispatch_params": dispatch_params,
            "drone_result": drone_result,
            "disease_result": disease_result,
            "web_url": web_url,
            "report_file": report_file
        }

