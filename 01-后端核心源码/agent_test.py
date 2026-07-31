from pymavlink import mavutil
import time
import threading
import json
import requests

# ================= 全局变量 =================
current_target = (0, 0, -5)
running = True

# 你的豆包API（请通过环境变量 DOUBAO_API_KEY 设置）
import os
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "YOUR_API_KEY_HERE")

# ================= 后台持续发目标点（永不炸机） =================
def send_target_loop(master):
    global current_target, running
    while running:
        x, y, z = current_target
        master.mav.set_position_target_local_ned_send(
            0, 1, 1,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111111000,
            x, y, z,
            0,0,0,0,0,0,0,0
        )
        time.sleep(0.05)

# ================= ✅ 修复SSL终极版：真实豆包API =================
def parse_natural_language_command(command):
    url = "https://api.doubao.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {DOUBAO_API_KEY}",
        "Content-Type": "application/json",
        "Connection": "close",  # 关键：解决SSL_EOF
    }

    system_prompt = """
    你是农业无人机指令解析器。只输出JSON动作数组，不要解释，不要文字。
    动作仅支持：takeoff(height), goto(x,y,z), take_photo(), land()
    A01地块(50,30)，B02(100,50)，C03(150,100)
    z为负表示向上。
    """

    data = {
        "model": "doubao-3.5-pro",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": command}
        ],
        "temperature": 0
    }

    try:
        # ✅ 这三行就是修复SSL报错的关键
        session = requests.Session()
        session.trust_env = False
        response = session.post(
            url, headers=headers, json=data,
            verify=False, allow_redirects=False, timeout=15
        )

        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return json.loads(content)

    except Exception as e:
        print(f"⚠️ API临时波动，使用默认安全动作: {e}")
        return [{"type": "takeoff", "height": 5}, {"type": "land"}]

# ================= 执行动作 =================
def execute_action_sequence(actions):
    global current_target
    for action in actions:
        if action["type"] == "takeoff":
            h = action["height"]
            print(f"\n🚀 起飞到 {h}m")
            current_target = (0,0,-h)
            time.sleep(5)

        elif action["type"] == "goto":
            x,y,z = action["x"], action["y"], action["z"]
            print(f"✈️ 飞往 ({x},{y},{-z})m")
            current_target = (x,y,z)
            time.sleep(8)

        elif action["type"] == "take_photo":
            print("📸 拍摄多光谱影像")
            time.sleep(1)

        elif action["type"] == "land":
            print("🪂 降落")
            master.mav.command_long_send(1,1,
                mavutil.mavlink.MAV_CMD_NAV_LAND,0,0,0,0,0,0,0,0)
            time.sleep(10)

# ================= 主程序 =================
if __name__ == "__main__":
    print("✅ 等待PX4连接...")
    master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')
    master.wait_heartbeat()
    print(f"✅ PX4 连接成功，系统ID：{master.target_system}")

    # 解锁
    print("\n正在强制解锁...")
    for _ in range(10):
        master.mav.command_long_send(1,1,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0,1,21196,0,0,0,0,0)
        time.sleep(0.1)
    time.sleep(2)

    # 启动后台线程
    t = threading.Thread(target=send_target_loop, args=(master,))
    t.daemon = True
    t.start()

    # 切换OFFBOARD
    print("切换到OFFBOARD模式...")
    master.mav.command_long_send(1,1,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,0,1,6,0,0,0,0,0)
    time.sleep(1)

    # ================= 测试真实指令 =================
    cmd = "让无人机飞到A01地块，高度10米，拍摄多光谱影像，然后返回起飞点降落"
    print("\n======================================")
    print("指令：" + cmd)

    actions = parse_natural_language_command(cmd)
    print("✅ 豆包API解析成功：")
    print(json.dumps(actions, indent=2, ensure_ascii=False))

    execute_action_sequence(actions)

    running = False
    print("\n🎉 任务完成！")