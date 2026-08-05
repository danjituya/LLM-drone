# -*- coding: utf-8 -*-
import os
import json
import time
import threading
import hashlib
import queue
from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
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

# LLM结果缓存（避免重复调用）
_llm_cache = {}
_llm_cache_lock = threading.Lock()
LLM_CACHE_TTL = 3600  # 缓存有效期1小时

# 异步任务队列和状态
_task_queue = queue.Queue()
_task_status = {}
_task_status_lock = threading.Lock()
_task_counter = 0

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

# ====================== LLM缓存管理 ======================
def get_llm_cache(key):
    """获取LLM缓存"""
    with _llm_cache_lock:
        if key in _llm_cache:
            entry = _llm_cache[key]
            if time.time() - entry['time'] < LLM_CACHE_TTL:
                return entry['value']
            else:
                del _llm_cache[key]
    return None

def set_llm_cache(key, value):
    """设置LLM缓存"""
    with _llm_cache_lock:
        _llm_cache[key] = {'value': value, 'time': time.time()}

def clear_llm_cache():
    """清除LLM缓存"""
    with _llm_cache_lock:
        _llm_cache.clear()

def make_cache_key(text):
    """生成缓存key"""
    return hashlib.md5(text.encode()).hexdigest()

# ====================== 异步任务管理 ======================
def add_task_status(task_id, status, message='', data=None):
    """更新任务状态"""
    with _task_status_lock:
        _task_status[task_id] = {
            'task_id': task_id,
            'status': status,  # pending, processing, completed, failed
            'message': message,
            'data': data,
            'updated_at': time.time()
        }

def get_task_status(task_id):
    """获取任务状态"""
    with _task_status_lock:
        return _task_status.get(task_id)

def generate_task_id():
    """生成任务ID"""
    global _task_counter
    _task_counter += 1
    return f"task_{int(time.time())}_{_task_counter}"

def process_async_task(task_id, message, is_patrol):
    """处理异步任务"""
    try:
        add_task_status(task_id, 'processing', '正在处理您的指令...')

        if not AGENT_AVAILABLE or not agri_agent:
            add_task_status(task_id, 'failed', 'Agent服务暂不可用')
            return

        # 检查LLM缓存
        cache_key = make_cache_key(message)
        cached = get_llm_cache(cache_key)

        if is_patrol:
            # 巡检任务：不缓存（每次结果都不同）
            add_task_status(task_id, 'processing', '正在解析指令...')
            result = agri_agent.process(message)

            patrol_id = result['drone_result']['patrol_id']
            with _lock:
                PATROL_DATA[patrol_id] = result
            save_patrol_data()

            add_task_status(task_id, 'completed', '巡检任务完成', {
                'type': 'patrol',
                'patrol_id': patrol_id,
                'result': result
            })
        else:
            # 普通对话：使用缓存
            if cached:
                add_task_status(task_id, 'completed', '对话完成', {
                    'type': 'chat',
                    'response': cached,
                    'cached': True
                })
                return

            add_task_status(task_id, 'processing', '正在思考...')
            response = agri_agent.ask(message)

            # 存入缓存
            set_llm_cache(cache_key, response)

            add_task_status(task_id, 'completed', '对话完成', {
                'type': 'chat',
                'response': response,
                'cached': False
            })

    except Exception as e:
        add_task_status(task_id, 'failed', f'任务处理失败: {str(e)}')

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

@app.route('/api/chat/async', methods=['POST'])
def chat_async():
    """异步对话接口（立即返回任务ID，后台处理）"""
    try:
        if not request.is_json:
            return jsonify({'code': 400, 'msg': '请求格式必须为JSON'}), 400

        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'code': 400, 'msg': '请输入消息内容'}), 400

        is_patrol = is_patrol_command(user_message)
        task_id = generate_task_id()
        add_task_status(task_id, 'pending', '任务已创建，等待处理')

        # 异步启动任务
        thread = threading.Thread(
            target=process_async_task,
            args=(task_id, user_message, is_patrol),
            daemon=True
        )
        thread.start()

        return jsonify({
            'code': 200,
            'msg': '任务已提交',
            'data': {
                'task_id': task_id,
                'is_patrol': is_patrol,
                'message': user_message
            }
        })
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'提交失败: {str(e)}'}), 500

@app.route('/api/task/status/<task_id>', methods=['GET'])
def get_async_task_status(task_id):
    """查询异步任务状态"""
    task = get_task_status(task_id)
    if not task:
        return jsonify({'code': 404, 'msg': '任务不存在'}), 404

    return jsonify({
        'code': 200,
        'msg': 'success',
        'data': task
    })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """SSE流式对话接口（实时推送处理进度）"""
    try:
        if not request.is_json:
            return jsonify({'code': 400, 'msg': '请求格式必须为JSON'}), 400

        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'code': 400, 'msg': '请输入消息内容'}), 400

        is_patrol = is_patrol_command(user_message)
        task_id = generate_task_id()

        add_task_status(task_id, 'pending', '任务已创建')

        def generate():
            try:
                yield f"data: {json.dumps({'event': 'start', 'task_id': task_id, 'message': '任务开始'})}\n\n"

                if not AGENT_AVAILABLE or not agri_agent:
                    yield f"data: {json.dumps({'event': 'error', 'message': 'Agent服务暂不可用'})}\n\n"
                    return

                cache_key = make_cache_key(user_message)
                cached = get_llm_cache(cache_key)

                if is_patrol:
                    # 巡检任务
                    yield f"data: {json.dumps({'event': 'progress', 'step': 'parsing', 'message': '正在解析指令...'})}\n\n"
                    add_task_status(task_id, 'processing', '正在解析指令...')

                    yield f"data: {json.dumps({'event': 'progress', 'step': 'vla', 'message': 'VLA视觉避障分析中...'})}\n\n"
                    add_task_status(task_id, 'processing', 'VLA视觉避障分析中...')

                    yield f"data: {json.dumps({'event': 'progress', 'step': 'flying', 'message': '无人机飞行巡检中...'})}\n\n"
                    add_task_status(task_id, 'processing', '无人机飞行巡检中...')

                    yield f"data: {json.dumps({'event': 'progress', 'step': 'detecting', 'message': '病虫害识别中...'})}\n\n"
                    add_task_status(task_id, 'processing', '病虫害识别中...')

                    result = agri_agent.process(user_message)
                    patrol_id = result['drone_result']['patrol_id']
                    with _lock:
                        PATROL_DATA[patrol_id] = result
                    save_patrol_data()

                    add_task_status(task_id, 'completed', '巡检完成', result)
                    yield f"data: {json.dumps({'event': 'done', 'result': {'type': 'patrol', 'patrol_id': patrol_id, 'result': result}})}\n\n"
                else:
                    # 普通对话
                    if cached:
                        add_task_status(task_id, 'completed', '对话完成', {'type': 'chat', 'response': cached, 'cached': True})
                        yield f"data: {json.dumps({'event': 'cached', 'message': '使用缓存结果', 'response': cached})}\n\n"
                        yield f"data: {json.dumps({'event': 'done', 'result': {'type': 'chat', 'response': cached, 'cached': True}})}\n\n"
                        return

                    yield f"data: {json.dumps({'event': 'progress', 'step': 'thinking', 'message': 'AI正在思考中...'})}\n\n"
                    add_task_status(task_id, 'processing', 'AI正在思考中...')

                    response = agri_agent.ask(user_message)
                    set_llm_cache(cache_key, response)

                    add_task_status(task_id, 'completed', '对话完成', {'type': 'chat', 'response': response})
                    yield f"data: {json.dumps({'event': 'done', 'result': {'type': 'chat', 'response': response, 'cached': False}})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
            finally:
                yield "data: [DONE]\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'服务器错误: {str(e)}'}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """清除LLM缓存"""
    clear_llm_cache()
    return jsonify({'code': 200, 'msg': '缓存已清除'})

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "code": 200,
        "status": "running",
        "msg": "后端服务运行正常",
        "agent_available": AGENT_AVAILABLE,
        "patrol_count": len(PATROL_DATA),
        "llm_cache_size": len(_llm_cache),
        "pending_tasks": len([t for t in _task_status.values() if t['status'] == 'pending'])
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
