# -*- coding: utf-8 -*-
import os
import json
import time
import threading
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import agriculture_drone_agent
from agriculture_drone_agent import init_core_modules, AgricultureDroneAgent
from config import (
    STATIC_FOLDER, REPORT_FOLDER, PATROL_DATA_FILE,
    PROMPTS_DIR, SERVER_HOST, SERVER_PORT
)

# 自动创建目录
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Flask应用
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='')
CORS(app, supports_credentials=True)

# 全局状态
AGENT_AVAILABLE = False
agri_agent = None
PATROL_DATA = {}
_lock = threading.Lock()

# ====================== 持久化管理 ======================
def load_patrol_data():
    """从文件加载巡检任务数据"""
    global PATROL_DATA
    if os.path.exists(PATROL_DATA_FILE):
        try:
            with open(PATROL_DATA_FILE, 'r', encoding='utf-8') as f:
                PATROL_DATA = json.load(f)
            print(f"📂 已加载 {len(PATROL_DATA)} 条历史巡检记录")
        except Exception as e:
            print(f"⚠️  加载历史数据失败: {e}")
            PATROL_DATA = {}

def save_patrol_data():
    """保存巡检任务数据到文件"""
    try:
        with open(PATROL_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(PATROL_DATA, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  保存数据失败: {e}")

# ====================== 意图识别（关键词快速判断） ======================
PATROL_KEYWORDS = [
    '巡检', '巡查', '巡逻', '飞行', '无人机',
    '喷药', '施肥', '播种', '病虫害', '病害',
    '农田', '地块', '作业', '任务', '航线',
    '多少台', '几台', '几个无人机'
]

def is_patrol_command(message):
    """快速判断是否为巡检指令（关键词匹配，避免LLM调用）"""
    message_lower = message.lower()
    for keyword in PATROL_KEYWORDS:
        if keyword in message_lower:
            return True
    return False

# ====================== Agent初始化 ======================
def init_agent():
    """初始化Agent服务"""
    global AGENT_AVAILABLE, agri_agent
    try:
        init_core_modules()
        print("正在初始化农业无人机Agent...")
        agri_agent = AgricultureDroneAgent(
            system_prompts=os.path.join(PROMPTS_DIR, "agri_drone_cn.txt"),
            knowledge_prompt=os.path.join(PROMPTS_DIR, "agri_drone_knowledge.txt"),
            vla_ctrl=agriculture_drone_agent.vla_ctrl,
            auto_load_knowledge=False
        )
        AGENT_AVAILABLE = True
        print("✅ Agent初始化成功")
    except Exception as e:
        print(f"⚠️ Agent模块初始化失败: {e}")
        AGENT_AVAILABLE = False

# ====================== 路由接口 ======================
@app.route('/')
def index():
    """首页路由"""
    index_file = os.path.join(STATIC_FOLDER, 'index.html')
    if os.path.exists(index_file):
        return send_from_directory(STATIC_FOLDER, 'index.html')
    return """
    <html>
    <head><meta charset="UTF-8"><title>系统提示</title></head>
    <body style="text-align:center;margin-top:100px;font-family:Microsoft YaHei;">
    <h1>✅ 后端服务运行正常</h1>
    <h3>❌ 前端 index.html 未找到</h3>
    <p>请将 index.html 放到以下路径：</p>
    <p style="color:red; font-weight:bold;">{}</p>
    <p>然后重启服务再访问 http://127.0.0.1:{}</p>
    </body>
    </html>
    """.format(STATIC_FOLDER, SERVER_PORT), 200

@app.route('/api/task/list', methods=['GET'])
def get_task_list():
    """获取巡检任务列表"""
    try:
        task_list = []
        with _lock:
            for patrol_id, task_data in PATROL_DATA.items():
                timestamp = int(patrol_id.split('_')[1]) if '_' in patrol_id else int(time.time())
                create_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                total_images = task_data['disease_result']['total_images']
                disease_images = task_data['disease_result']['disease_images']
                disease_rate = round(disease_images / total_images * 100, 2) if total_images > 0 else 0
                task_list.append({
                    "patrol_id": patrol_id,
                    "create_time": create_time,
                    "area": task_data['dispatch_params'].get('area', '默认农田'),
                    "drone_num": task_data['drone_result']['drone_num'],
                    "disease_rate": disease_rate,
                    "dispatch_params": task_data['dispatch_params'],
                    "drone_status": list(task_data['drone_result'].get('drone_status', {}).values()),
                    "disease_summary": {
                        "total_images": total_images,
                        "disease_images": disease_images,
                        "disease_rate": disease_rate,
                        "class_names": list(task_data['disease_result'].get('class_names', {}).values())
                    },
                    "path_viz_path": task_data['drone_result'].get('path_viz_path', ''),
                    "drone_result": task_data['drone_result'],
                    "disease_result": task_data['disease_result']
                })
        task_list.sort(key=lambda x: x['create_time'], reverse=True)
        return jsonify({"code": 200, "msg": "success", "data": task_list})
    except Exception as e:
        print(f"❌ 获取任务列表失败: {e}")
        return jsonify({"code": 500, "msg": f"获取任务列表失败: {str(e)}", "data": []}), 500

@app.route('/api/task/detail/<patrol_id>', methods=['GET'])
def get_task_detail(patrol_id):
    """获取单个任务详情"""
    with _lock:
        if patrol_id not in PATROL_DATA:
            return jsonify({"code": 404, "msg": "任务不存在", "data": None}), 404
        return jsonify({"code": 200, "msg": "success", "data": PATROL_DATA[patrol_id]})

@app.route('/report/<timestamp>')
def get_report(timestamp):
    """获取可视化报告"""
    report_file = os.path.join(REPORT_FOLDER, f"report_{timestamp}.json")
    if os.path.exists(report_file):
        with open(report_file, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        return jsonify({"code": 200, "data": report_data})
    return jsonify({"code": 404, "msg": "报告不存在"}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    """智能对话接口"""
    try:
        if not request.is_json:
            return jsonify({'code': 400, 'msg': '请求格式必须为JSON', 'data': None}), 400
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'code': 400, 'msg': '请输入消息内容', 'data': None}), 400

        print(f"\n👤 用户: {user_message}")

        if not AGENT_AVAILABLE or not agri_agent:
            return jsonify({
                'code': 503,
                'msg': 'Agent服务暂不可用',
                'data': {'type': 'chat', 'response': '抱歉，核心服务初始化失败'}
            }), 503

        try:
            # 使用关键词快速判断是否为巡检指令（避免LLM分类调用）
            if is_patrol_command(user_message):
                result = agri_agent.process(user_message)
                patrol_id = result['drone_result']['patrol_id']
                with _lock:
                    PATROL_DATA[patrol_id] = result
                save_patrol_data()
                return jsonify({
                    'code': 200,
                    'msg': '巡检任务已完成',
                    'data': {
                        'type': 'patrol',
                        'patrol_id': patrol_id,
                        'result': result
                    }
                })
            else:
                # 普通对话直接调用LLM
                chat_response = agri_agent.ask(user_message)
                print(f"🤖 Agent: {chat_response}")
                return jsonify({
                    'code': 200,
                    'msg': 'success',
                    'data': {'type': 'chat', 'response': chat_response}
                })
        except Exception as e:
            print(f"❌ Agent处理失败: {e}")
            return jsonify({
                'code': 500,
                'msg': f'任务处理失败: {str(e)}',
                'data': None
            }), 500
    except Exception as e:
        print(f"❌ 接口错误: {e}")
        return jsonify({
            'code': 500,
            'msg': f'服务器内部错误: {str(e)}',
            'data': None
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "code": 200,
        "status": "running",
        "msg": "后端服务运行正常",
        "agent_available": AGENT_AVAILABLE,
        "patrol_count": len(PATROL_DATA)
    })

# ====================== 服务启动 ======================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 智慧农业无人机巡检系统 - 后端服务")
    # 加载历史数据
    load_patrol_data()
    # 初始化Agent
    init_agent()
    print(f"📂 前端静态文件路径: {STATIC_FOLDER}")
    print(f"🌐 系统访问地址: http://127.0.0.1:{SERVER_PORT}")
    print(f"🔌 健康检查接口: http://127.0.0.1:{SERVER_PORT}/api/health")
    print(f"🔌 任务列表接口: http://127.0.0.1:{SERVER_PORT}/api/task/list")
    print("=" * 60 + "\n")
    # 启动服务
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, use_reloader=False, threaded=True)
