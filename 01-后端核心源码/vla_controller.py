# -*- coding: utf-8 -*-
import base64
import re
import json
import os
import io
import time
from config import (BASE_URL, ARK_API_KEY, VLA_MODEL,
                    VLA_MAX_IMAGE_SIZE, VLA_QUALITY, VLA_MAX_BASE64_LENGTH)
from openai import OpenAI


class Vlacontroller:
    def __init__(self, model_api_key=ARK_API_KEY, model_endpoint=BASE_URL,
                 vla_model=VLA_MODEL):
        self.client = OpenAI(base_url=model_endpoint, api_key=model_api_key)
        self.vla_model = vla_model
        self._init_pillow()

    def _init_pillow(self):
        """初始化Pillow用于图片压缩"""
        try:
            from PIL import Image
            self._pillow_available = True
        except ImportError:
            print("⚠️  未安装Pillow，图片压缩功能不可用，建议执行: pip install Pillow")
            self._pillow_available = False

    def process_visual_input(self, visual_data, current_state,
                              language_instruction=None):
        """处理视觉数据并返回控制信号"""
        start_time = time.time()

        img_base64 = self._process_visual_data(visual_data)
        if not img_base64:
            return {"action": "continue", "reason": "视觉数据为空或加载失败"}

        prompt = self._build_vla_prompt(current_state, language_instruction)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.vla_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                timeout=60
            )
            raw_output = completion.choices[0].message.content
            elapsed = time.time() - start_time

            control_signal = self._parse_output(raw_output)
            control_signal["processing_time"] = round(elapsed, 2)
            return control_signal

        except Exception as e:
            elapsed = time.time() - start_time
            return {"action": "continue", "reason": f"模型处理失败：{str(e)}"}

    def _process_visual_data(self, visual_data):
        """处理视觉数据，支持压缩优化"""
        if not visual_data:
            return ""

        if isinstance(visual_data, str) and len(visual_data) > 100:
            try:
                base64.b64decode(visual_data[:100])
                if len(visual_data) > VLA_MAX_BASE64_LENGTH and self._pillow_available:
                    return self._compress_base64_image(visual_data)
                return visual_data
            except Exception:
                pass

        if isinstance(visual_data, str) and os.path.exists(visual_data):
            try:
                if self._pillow_available:
                    return self._compress_file_image(visual_data)
                else:
                    with open(visual_data, "rb") as f:
                        return base64.b64encode(f.read()).decode('utf-8')
            except Exception:
                return ""

        if isinstance(visual_data, bytes):
            return self._compress_bytes_image(visual_data) if self._pillow_available else base64.b64encode(visual_data).decode('utf-8')

        return ""

    def _compress_base64_image(self, base64_str):
        """压缩base64图片"""
        try:
            from PIL import Image
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes))
            compressed = self._resize_and_compress(img)
            buffer = io.BytesIO()
            compressed.save(buffer, format='JPEG', quality=VLA_QUALITY)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception:
            return base64_str

    def _compress_file_image(self, file_path):
        """压缩文件图片"""
        try:
            from PIL import Image
            img = Image.open(file_path)
            compressed = self._resize_and_compress(img)
            buffer = io.BytesIO()
            compressed.save(buffer, format='JPEG', quality=VLA_QUALITY)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception:
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')

    def _compress_bytes_image(self, img_bytes):
        """压缩bytes图片"""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes))
            compressed = self._resize_and_compress(img)
            buffer = io.BytesIO()
            compressed.save(buffer, format='JPEG', quality=VLA_QUALITY)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception:
            return base64.b64encode(img_bytes).decode('utf-8')

    def _resize_and_compress(self, img):
        """调整图片大小"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        if max(w, h) > VLA_MAX_IMAGE_SIZE:
            ratio = VLA_MAX_IMAGE_SIZE / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img

    def _parse_output(self, raw_output):
        """解析模型输出 - 更健壮的解析"""
        # 尝试多种JSON匹配方式
        json_data = None
        
        # 方式1：直接尝试解析（可能是纯JSON）
        try:
            json_data = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            pass
        
        # 方式2：用正则提取JSON
        if json_data is None:
            # 尝试匹配最外层的JSON对象
            json_match = re.search(r'\{[\s\S]*\}', raw_output)
            if json_match:
                try:
                    json_str = json_match.group()
                    json_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # 尝试修复常见JSON问题
                    fixed_json = self._fix_json_string(json_str)
                    try:
                        json_data = json.loads(fixed_json)
                    except json.JSONDecodeError:
                        print(f"⚠️ JSON解析失败，使用兜底值")
        
        # 方式3：兜底值
        if json_data is None:
            print(f"⚠️ 未匹配到有效JSON，使用兜底值")
            json_data = {
                "action": "continue",
                "reason": "模型未返回标准JSON格式",
                "safety_level": "safe",
                "obstacle_position": [0, 0],
                "path_correction": [],
                "obstacle_type": "无"
            }
        
        # 补全默认值
        result = {
            "action": json_data.get("action", "continue"),
            "reason": json_data.get("reason", "模型未返回原因"),
            "safety_level": json_data.get("safety_level", "safe"),
            "obstacle_position": json_data.get("obstacle_position", [0, 0]),
            "path_correction": json_data.get("path_correction", []),
            "obstacle_type": json_data.get("obstacle_type", "无")
        }
        
        # 验证action合法性
        allowed_actions = {"avoid_obstacle", "continue", "stop"}
        if result["action"] not in allowed_actions:
            print(f"⚠️ 非法action「{result['action']}」，兜底为continue")
            result["action"] = "continue"
            result["reason"] = "非法action兜底"
        
        # 验证obstacle_position是数字数组
        if not isinstance(result["obstacle_position"], list) or len(result["obstacle_position"]) != 2:
            result["obstacle_position"] = [0, 0]
        
        return result

    def _fix_json_string(self, json_str):
        """修复常见JSON问题"""
        # 移除可能的markdown代码块标记
        json_str = re.sub(r'```json\s*|\s*```', '', json_str)
        # 修复单引号为双引号
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        # 移除尾随逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        return json_str

    def _build_vla_prompt(self, current_state, language_instruction):
        """构建优化的VLA提示词"""
        base_prompt = """你是无人机的视觉系统，负责检测飞行路径上的障碍物。

【任务】
分析这张航拍图像，检测是否存在影响无人机飞行的障碍物。

【障碍物类型】
- 电线杆、电线
- 树木、树枝
- 建筑物、围墙
- 其他高出地面的物体

【判断规则】
- 若检测到障碍物：action = "avoid_obstacle"，标注障碍物位置
- 若未检测到障碍物：action = "continue"
- 若图像模糊无法判断：action = "continue"，reason说明原因

【输出格式 - 严格JSON，不允许其他文字】
```json
{
    "action": "avoid_obstacle" 或 "continue",
    "reason": "检测到电线杆" 或 "未检测到障碍物" 或 "图像模糊",
    "safety_level": "danger" 或 "safe",
    "obstacle_position": [x坐标, y坐标],
    "path_correction": [],
    "obstacle_type": "障碍物类型" 或 "无"
}
```

【注意】
1. 只输出JSON，不要任何解释文字
2. obstacle_position是[0-100]的相对坐标
3. 保持JSON格式正确"""
        
        if language_instruction:
            base_prompt += f"\n\n【额外指令】{language_instruction}"
        
        return base_prompt
