# -*- coding: utf-8 -*-
"""
统一配置模块 - 示例文件
使用说明：将此文件复制为 config.py 并填入实际配置
"""
import os

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = os.path.dirname(BASE_DIR)
PROJECT_ROOT = os.path.dirname(SOURCE_ROOT)

# 前端页面路径
STATIC_FOLDER = os.path.join(SOURCE_ROOT, "02-前端源码", "前端页面")
REPORT_FOLDER = os.path.join(STATIC_FOLDER, "reports")

# 提示词文件夹路径
PROMPTS_DIR = os.path.join(SOURCE_ROOT, "03-大模型提示词", "prompts")

# 模型文件路径
MODEL_PATH = os.path.join(SOURCE_ROOT, "04-模型文件", "best.pt")

# 数据集配置路径
YAML_PATH = os.path.join(SOURCE_ROOT, "05-数据集样本", "dataset_sample", "plantvillage.yaml")

# 示例图片源目录
SOURCE_IMAGE_DIR = os.path.join(SOURCE_ROOT, "05-数据集样本", "dataset_sample")

# 采集数据保存目录
BASE_DATA_PATH = os.path.join(SOURCE_ROOT, "06-辅助数据", "drone_collected_data")

# 临时图片目录
TEMP_IMAGE_DIR = os.path.join(SOURCE_ROOT, "06-辅助数据", "temp_images")

# 持久化存储路径
PATROL_DATA_FILE = os.path.join(BASE_DIR, "patrol_data.json")

# ====================== API 配置 ======================
# 请填入实际的API地址和密钥
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_API_KEY = "在此填入你的API密钥"

# ====================== 模型配置 ======================
# 文本大模型（用于对话、指令解析）
# 性能优先：deepseek-v3-2-251201（1800次实测均速2.42s，最稳定）
MODEL = "deepseek-v3-2-251201"

# VLA视觉语言模型（用于航拍照片障碍物识别+避障决策）
# 性能优先：doubao-seed-1-6-vision-250815（seed系列最新视觉能力）
VLA_MODEL = "doubao-seed-1-6-vision-250815"

# VLA图片压缩配置
VLA_MAX_IMAGE_SIZE = 800
VLA_QUALITY = 85
VLA_MAX_BASE64_LENGTH = 500000

# ====================== 服务配置 ======================
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 54321

# ====================== 无人机运行模式配置 ======================
# 运行模式: 'simulation' | 'px4_sitl' | 'real_flight'
FLIGHT_MODE = os.environ.get('DRONE_FLIGHT_MODE', 'simulation')

# PX4连接配置
PX4_CONNECTION_STRING = os.environ.get('PX4_CONNECTION', 'udp:127.0.0.1:14550')
PX4_CONNECTION_BAUD = int(os.environ.get('PX4_BAUD', '57600'))

# 真实无人机连接配置
REAL_DRONE_CONNECTION = os.environ.get('REAL_DRONE_CONNECTION', '')

# 基础无人机参数
DEFAULT_DRONE_NUM = 3
DEFAULT_ALTITUDE = 10
DEFAULT_AREA_BOUNDS = [0, 0, 200, 150]
DEFAULT_TAKEOFF_ALTITUDE = 10
DEFAULT_FLIGHT_SPEED = 5.0

# 飞行安全限制
MAX_FLIGHT_ALTITUDE = 120
MAX_FLIGHT_DISTANCE = 500
GEO_FENCE_ENABLED = True

# ====================== 安全配置 ======================
MAX_CHAT_HISTORY_LENGTH = 20
LLM_TIMEOUT = 10
DISEASE_CONFIDENCE_THRESHOLD = 0.7
