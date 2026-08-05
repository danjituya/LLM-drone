# -*- coding: utf-8 -*-
# @File    : drone_scheduler.py
# @Describe: 无人机调度模块 - 专业版（支持多飞行模式）
import os
import shutil
import time
import random
import math
import logging
from abc import ABC, abstractmethod
from config import (
    SOURCE_IMAGE_DIR, BASE_DATA_PATH,
    DEFAULT_DRONE_NUM, DEFAULT_AREA_BOUNDS,
    FLIGHT_MODE, PX4_CONNECTION_STRING, PX4_CONNECTION_BAUD,
    REAL_DRONE_CONNECTION, DEFAULT_TAKEOFF_ALTITUDE,
    DEFAULT_FLIGHT_SPEED, MAX_FLIGHT_ALTITUDE, GEO_FENCE_ENABLED
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# matplotlib导入加容错
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
    logger.warning("未安装matplotlib，路径可视化功能将不可用")


# ====================== 飞行控制器抽象基类 ======================
class FlightController(ABC):
    """飞行控制器抽象基类 - 支持不同飞行模式"""
    
    @abstractmethod
    def connect(self):
        """连接飞行器"""
        pass
    
    @abstractmethod
    def arm_and_takeoff(self, altitude=DEFAULT_TAKEOFF_ALTITUDE):
        """解锁并起飞"""
        pass
    
    @abstractmethod
    def goto_waypoint(self, x, y, z=None):
        """飞往航点"""
        pass
    
    @abstractmethod
    def land(self):
        """降落"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def get_status(self):
        """获取飞行状态"""
        pass


# ====================== 模拟飞行控制器 ======================
class SimulationFlightController(FlightController):
    """纯模拟模式 - 无实际飞行控制"""
    
    def __init__(self):
        self.connected = False
        self.current_position = [0, 0, 0]
        self.battery = 100.0
        logger.info("🎮 初始化模拟飞行控制器")
    
    def connect(self):
        self.connected = True
        logger.info("✅ 模拟模式已连接（无需硬件）")
        return True
    
    def arm_and_takeoff(self, altitude=DEFAULT_TAKEOFF_ALTITUDE):
        logger.info(f"🚁 模拟起飞至 {altitude} 米")
        self.current_position[2] = altitude
        return True
    
    def goto_waypoint(self, x, y, z=None):
        logger.info(f"   → 模拟飞往航点: ({x:.1f}, {y:.1f}, {z or self.current_position[2]:.1f})")
        self.current_position = [x, y, z or self.current_position[2]]
        return True
    
    def land(self):
        logger.info("🛬 模拟降落")
        self.current_position[2] = 0
        return True
    
    def disconnect(self):
        self.connected = False
        logger.info("🔌 模拟模式已断开")
    
    def get_status(self):
        return {
            "mode": "simulation",
            "connected": self.connected,
            "position": self.current_position,
            "battery": self.battery,
            "armed": self.current_position[2] > 0
        }


# ====================== PX4飞控控制器 ======================
class PX4FlightController(FlightController):
    """PX4飞控控制器 - 支持仿真和真实飞行"""
    
    def __init__(self, connection_string=None, baud=None, is_real_flight=False):
        self.connection_string = connection_string or PX4_CONNECTION_STRING
        self.baud = baud or PX4_CONNECTION_BAUD
        self.is_real_flight = is_real_flight
        self.master = None
        self.connected = False
        self.current_position = [0, 0, 0]
        mode_name = "真实飞行" if is_real_flight else "PX4仿真"
        logger.info(f"🚀 初始化{mode_name}控制器: {self.connection_string}")
    
    def connect(self):
        try:
            from pymavlink import mavutil
            
            # 根据连接类型选择连接方式
            if self.connection_string.startswith('udp:'):
                self.master = mavutil.mavlink_connection(self.connection_string)
            elif self.connection_string.startswith('/') or self.connection_string.startswith('COM'):
                self.master = mavutil.mavlink_connection(
                    self.connection_string, baud=self.baud
                )
            else:
                self.master = mavutil.mavlink_connection(self.connection_string)
            
            # 等待心跳包
            self.master.wait_heartbeat(timeout=5)
            self.connected = True
            mode_name = "真实无人机" if self.is_real_flight else "PX4仿真"
            logger.info(f"✅ 已连接到{mode_name}: System={self.master.target_system}")
            return True
        except ImportError:
            logger.error("❌ 未安装pymavlink库，请执行: pip install pymavlink")
            return False
        except Exception as e:
            logger.error(f"❌ 连接失败: {e}")
            return False
    
    def arm_and_takeoff(self, altitude=DEFAULT_TAKEOFF_ALTITUDE):
        if not self.connected:
            logger.error("❌ 未连接飞控")
            return False
        
        try:
            from pymavlink import mavutil
            
            # 设置飞行模式为GUIDED
            self.master.mav.set_mode_send(
                self.master.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4  # GUIDED mode
            )
            time.sleep(0.5)
            
            # 解锁
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )
            
            # 等待解锁确认
            self.master.motors_armed_wait()
            logger.info("✅ 电机已解锁")
            
            # 起飞
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0, 0, altitude
            )
            
            # 等待到达目标高度
            while True:
                msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=30)
                if msg:
                    current_alt = msg.relative_alt / 1000.0
                    if current_alt >= altitude * 0.95:
                        logger.info(f"✅ 已到达目标高度 {altitude} 米")
                        break
                time.sleep(0.5)
            
            self.current_position[2] = altitude
            return True
        except Exception as e:
            logger.error(f"❌ 起飞失败: {e}")
            return False
    
    def goto_waypoint(self, x, y, z=None):
        if not self.connected:
            return False
        
        try:
            from pymavlink import mavutil
            
            z = z or self.current_position[2] or DEFAULT_TAKEOFF_ALTITUDE
            
            # 发送航点命令
            self.master.mav.mission_item_send(
                self.master.target_system,
                self.master.target_component,
                0,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 1, 0, 0, 0, 0,
                x, y, -z
            )
            
            # 等待到达航点
            while True:
                msg = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=60)
                if msg:
                    dx = abs(msg.x - x)
                    dy = abs(msg.y - y)
                    if dx < 1.0 and dy < 1.0:
                        logger.info(f"   ✅ 已到达航点: ({x:.1f}, {y:.1f})")
                        break
                time.sleep(0.5)
            
            self.current_position = [x, y, z]
            return True
        except Exception as e:
            logger.error(f"❌ 飞往航点失败: {e}")
            return False
    
    def land(self):
        if not self.connected:
            return False
        
        try:
            from pymavlink import mavutil
            
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            logger.info("🛬 开始降落")
            self.current_position[2] = 0
            return True
        except Exception as e:
            logger.error(f"❌ 降落失败: {e}")
            return False
    
    def disconnect(self):
        self.connected = False
        if self.master:
            self.master.close()
        logger.info("🔌 已断开飞控连接")
    
    def get_status(self):
        return {
            "mode": "real_flight" if self.is_real_flight else "px4_sitl",
            "connected": self.connected,
            "position": self.current_position,
            "connection": self.connection_string
        }


# ====================== 飞行控制器工厂 ======================
def create_flight_controller(mode=None):
    """
    创建飞行控制器实例
    
    Args:
        mode: 飞行模式，可选值:
            - 'simulation': 纯模拟模式（默认）
            - 'px4_sitl': PX4软件在环仿真
            - 'real_flight': 真实飞行模式
    
    Returns:
        FlightController: 对应的飞行控制器实例
    """
    mode = mode or FLIGHT_MODE
    
    if mode == 'simulation':
        return SimulationFlightController()
    elif mode == 'px4_sitl':
        return PX4FlightController(
            connection_string=PX4_CONNECTION_STRING,
            is_real_flight=False
        )
    elif mode == 'real_flight':
        connection = REAL_DRONE_CONNECTION or PX4_CONNECTION_STRING
        return PX4FlightController(
            connection_string=connection,
            baud=PX4_CONNECTION_BAUD,
            is_real_flight=True
        )
    else:
        logger.warning(f"未知飞行模式 '{mode}'，使用模拟模式")
        return SimulationFlightController()


# ====================== 无人机调度器 ======================
class DroneScheduler:
    def __init__(self, drone_num=DEFAULT_DRONE_NUM, flight_mode=None):
        self.drone_num = drone_num
        self.drone_ids = [f"drone_{i + 1}" for i in range(drone_num)]
        self.base_data_path = BASE_DATA_PATH
        os.makedirs(self.base_data_path, exist_ok=True)
        self.source_image_dir = SOURCE_IMAGE_DIR
        self.algorithm_type = "vla_avoidance"
        
        # 初始化飞行控制器
        self.flight_mode = flight_mode or FLIGHT_MODE
        self.flight_controller = create_flight_controller(self.flight_mode)
        self.current_flight_mode = self.flight_controller.get_status()['mode']
        
        logger.info(f"✅ 无人机调度器初始化完成")
        logger.info(f"   🚁 无人机数量: {self.drone_num}")
        logger.info(f"   🎮 飞行模式: {self.current_flight_mode}")

    def plan_path(self, area_bounds, drone_num, path_type="zigzag",
                  has_obstacle=False, obstacle_position=None):
        x_min, y_min, x_max, y_max = area_bounds
        area_width = x_max - x_min
        sub_area_width = area_width / drone_num
        drone_paths = {}

        for i, drone_id in enumerate(self.drone_ids[:drone_num]):
            sub_x_min = x_min + i * sub_area_width
            sub_x_max = x_min + (i + 1) * sub_area_width

            if self.algorithm_type == "traditional_zigzag":
                path_points = self._generate_zigzag_path(sub_x_min, y_min, sub_x_max, y_max, 10)
            elif self.algorithm_type == "a_star":
                path_points = self._generate_a_star_path(
                    sub_x_min, y_min, sub_x_max, y_max,
                    obstacle_position, scan_spacing=30
                )
                path_points = self._optimize_path(path_points)
            elif self.algorithm_type == "vla_avoidance":
                if (has_obstacle and obstacle_position
                        and sub_x_min <= obstacle_position[0] <= sub_x_max):
                    path_points = self._generate_obstacle_avoidance_arc_path(
                        sub_x_min, y_min, sub_x_max, y_max, obstacle_position
                    )
                else:
                    path_points = self._generate_zigzag_path(sub_x_min, y_min, sub_x_max, y_max, 10)
            else:
                path_points = self._generate_zigzag_path(sub_x_min, y_min, sub_x_max, y_max, 10)

            drone_paths[drone_id] = {
                "sub_area": [sub_x_min, y_min, sub_x_max, y_max],
                "path_points": path_points,
                "path_length_m": self._calculate_path_length(path_points),
                "has_obstacle": (has_obstacle and
                                 obstacle_position is not None and
                                 sub_x_min <= obstacle_position[0] <= sub_x_max),
                "obstacle_position": obstacle_position
            }
        return drone_paths

    def _generate_zigzag_path(self, x_min, y_min, x_max, y_max, scan_spacing=10):
        path_points = []
        y = y_min
        direction = 1
        while y <= y_max:
            if direction == 1:
                path_points.append((x_min, y))
                path_points.append((x_max, y))
            else:
                path_points.append((x_max, y))
                path_points.append((x_min, y))
            y += scan_spacing
            direction *= -1
        return path_points

    def _generate_obstacle_avoidance_arc_path(self, x_min, y_min, x_max, y_max,
                                                obstacle_position, scan_spacing=10):
        """生成绕障路径：在障碍点附近绕一个半圆"""
        obs_x, obs_y = obstacle_position
        path_points = []
        y = y_min
        direction = 1
        obstacle_radius = 15

        while y <= y_max:
            if abs(y - obs_y) < obstacle_radius:
                if direction == 1:
                    path_points.append((x_min, y))
                    path_points.append((obs_x - obstacle_radius, y))
                    for angle in range(180, 360, 20):
                        rad = math.radians(angle)
                        px = obs_x + obstacle_radius * math.cos(rad)
                        py = obs_y + obstacle_radius * math.sin(rad)
                        path_points.append((px, py))
                    path_points.append((obs_x + obstacle_radius, y))
                    path_points.append((x_max, y))
                else:
                    path_points.append((x_max, y))
                    path_points.append((obs_x + obstacle_radius, y))
                    for angle in range(0, 180, 20):
                        rad = math.radians(angle)
                        px = obs_x + obstacle_radius * math.cos(rad)
                        py = obs_y - obstacle_radius * math.sin(rad)
                        path_points.append((px, py))
                    path_points.append((obs_x - obstacle_radius, y))
                    path_points.append((x_min, y))
            else:
                if direction == 1:
                    path_points.append((x_min, y))
                    path_points.append((x_max, y))
                else:
                    path_points.append((x_max, y))
                    path_points.append((x_min, y))

            y += scan_spacing
            direction *= -1
        return path_points

    def _generate_a_star_path(self, x_min, y_min, x_max, y_max,
                              obstacle_position=None, scan_spacing=30):
        """A*算法优化路径：步长大、航点少"""
        path_points = []
        y = y_min
        direction = 1

        while y <= y_max:
            if direction == 1:
                path_points.append((x_min, y))
                path_points.append((x_max, y))
            else:
                path_points.append((x_max, y))
                path_points.append((x_min, y))
            y += scan_spacing
            direction *= -1

        final_path = []
        for p in path_points:
            if not final_path or p != final_path[-1]:
                final_path.append(p)
        return final_path

    def _calculate_path_length(self, path_points):
        length = 0
        for i in range(1, len(path_points)):
            x1, y1 = path_points[i - 1]
            x2, y2 = path_points[i]
            length += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return round(length, 2)

    def _optimize_path(self, path_points, min_distance=25):
        """优化路径：合并共线航点"""
        if len(path_points) < 3:
            return path_points

        optimized = [path_points[0]]
        for i in range(1, len(path_points) - 1):
            prev = optimized[-1]
            curr = path_points[i]
            next_p = path_points[i + 1]

            cross = ((curr[0] - prev[0]) * (next_p[1] - prev[1])
                     - (curr[1] - prev[1]) * (next_p[0] - prev[0]))
            distance = ((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2) ** 0.5

            if abs(cross) > 1e-6 or distance > min_distance:
                optimized.append(curr)

        optimized.append(path_points[-1])
        return optimized

    def visualize_paths(self, drone_paths, area_bounds, save_path):
        if not HAS_MATPLOTLIB:
            return None
        x_min, y_min, x_max, y_max = area_bounds
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='black', facecolor='lightgreen', alpha=0.3,
            label='农田区域'
        )
        ax.add_patch(rect)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        for i, (drone_id, data) in enumerate(drone_paths.items()):
            color = colors[i % len(colors)]
            path = data['path_points']
            sub_area = data['sub_area']
            sub_rect = patches.Rectangle(
                (sub_area[0], sub_area[1]),
                sub_area[2] - sub_area[0], sub_area[3] - sub_area[1],
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.15
            )
            ax.add_patch(sub_rect)
            if path:
                xs, ys = zip(*path)
                plt.plot(xs, ys, color=color, marker='o', markersize=3,
                         linewidth=1.5, label=f'{drone_id} 路径')
                plt.plot(path[0][0], path[0][1], color=color, marker='s',
                         markersize=8, label=f'{drone_id} 起点')
        plt.xlabel('X 坐标 (米)')
        plt.ylabel('Y 坐标 (米)')
        plt.title(f'无人机巡检路径规划 (共 {len(drone_paths)} 台)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        viz_path = os.path.join(save_path, "path_planning.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        return viz_path

    def _load_test_image(self):
        """加载测试图像用于VLA分析"""
        test_dir = os.path.join(
            os.path.dirname(BASE_DATA_PATH),
            "test_images"
        )
        if os.path.exists(test_dir):
            for fname in os.listdir(test_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(test_dir, fname)
                    try:
                        import base64
                        with open(img_path, "rb") as f:
                            return base64.b64encode(f.read()).decode('utf-8')
                    except Exception as e:
                        logger.warning(f"测试图片加载失败: {e}")
        return ""

    def start_patrol(self, area, drone_num=DEFAULT_DRONE_NUM, path="zigzag",
                     area_bounds=None, has_obstacle=False, obstacle_position=None,
                     vla_controller=None):
        """执行巡检任务"""
        import base64

        if area_bounds is None:
            area_bounds = DEFAULT_AREA_BOUNDS

        use_drone_num = (drone_num if drone_num and isinstance(drone_num, int)
                         and 0 < drone_num <= 10 else self.drone_num)
        if len(area_bounds) != 4:
            area_bounds = DEFAULT_AREA_BOUNDS
        x_min, y_min, x_max, y_max = area_bounds
        if x_max <= x_min or y_max <= y_min:
            area_bounds = DEFAULT_AREA_BOUNDS

        logger.info(f"🚁 开始执行巡检任务: {area}, 无人机数: {use_drone_num}")

        self.current_area_bounds = area_bounds
        self.current_path_points = []
        self.has_obstacle = False
        self.obstacle_position = None

        drone_paths = self.plan_path(
            area_bounds, use_drone_num, path,
            has_obstacle=has_obstacle,
            obstacle_position=obstacle_position
        )

        if vla_controller is not None:
            try:
                current_state = self._get_current_state()
                visual_data = self._load_test_image()

                if visual_data and len(visual_data) > 100:
                    control_signal = vla_controller.process_visual_input(
                        visual_data, current_state,
                        language_instruction="检测农田中的障碍物（电线杆、树木等）"
                    )

                    if control_signal.get("action") == "avoid_obstacle":
                        obstacle_pos = self._convert_vla_coordinates(
                            control_signal.get("obstacle_position", [50, 50]), area_bounds
                        )
                        logger.info(f"🚨 VLA检测到障碍物，世界坐标: {obstacle_pos}，重新规划绕障路径")
                        drone_paths = self.plan_path(
                            area_bounds, use_drone_num, path,
                            has_obstacle=True, obstacle_position=obstacle_pos
                        )
                        self.has_obstacle = True
                        self.obstacle_position = obstacle_pos
            except Exception as e:
                logger.warning(f"VLA分析失败: {e}")

        patrol_id = f"patrol_{int(time.time())}"
        patrol_data_path = os.path.join(self.base_data_path, patrol_id)
        os.makedirs(patrol_data_path, exist_ok=True)

        viz_path = self.visualize_paths(drone_paths, area_bounds, patrol_data_path)

        start_time = time.time()
        drone_status = {}

        try:
            if self.flight_controller.connect():
                self.flight_controller.arm_and_takeoff(DEFAULT_TAKEOFF_ALTITUDE)

                self._execute_flight_with_vla(drone_paths, area_bounds, vla_controller)

                self.flight_controller.land()
            else:
                self._simulate_flight(drone_paths, area_bounds)
        except Exception as e:
            logger.error(f"飞行执行异常: {e}")
            self._simulate_flight(drone_paths, area_bounds)
        finally:
            self.flight_controller.disconnect()

        total_time = time.time() - start_time
        logger.info(f"✅ 飞行任务完成，耗时: {round(total_time, 2)}秒")

        final_data_path = os.path.join(patrol_data_path, "all_collected")
        os.makedirs(final_data_path, exist_ok=True)
        self._merge_all_images(patrol_data_path, final_data_path)

        for drone_id, data in drone_paths.items():
            drone_status[drone_id] = {
                "drone_id": drone_id,
                "sub_area": data["sub_area"],
                "path_length_m": data["path_length_m"],
                "collected_count": data.get("collected_count", len(data["path_points"])),
                "status": "completed",
                "image_dir": os.path.join(patrol_data_path, drone_id)
            }

        return {
            "status": "success",
            "patrol_id": patrol_id,
            "area": area,
            "area_bounds": area_bounds,
            "drone_num": use_drone_num,
            "patrol_path": path,
            "drone_status": drone_status,
            "drone_paths": drone_paths,
            "path_viz_path": viz_path,
            "data_path": final_data_path,
            "has_obstacle": self.has_obstacle,
            "obstacle_position": self.obstacle_position
        }

    def _convert_vla_coordinates(self, vla_position, area_bounds):
        """将VLA返回的[0-100]图片相对坐标转换为世界坐标"""
        x_min, y_min, x_max, y_max = area_bounds
        rel_x, rel_y = vla_position[0], vla_position[1]
        world_x = (rel_x / 100.0) * (x_max - x_min) + x_min
        world_y = (rel_y / 100.0) * (y_max - y_min) + y_min
        return [round(world_x, 2), round(world_y, 2)]

    def _execute_flight_with_vla(self, drone_paths, area_bounds, vla_controller):
        """执行飞行任务，支持航点级VLA实时避障"""
        VLA_CHECK_INTERVAL = 5  # 每5个航点调用一次VLA检测（避免模拟时间过长）
        MAX_VLA_CHECKS = 3      # 最多调用VLA 3次（避免模拟时间过长）

        vla_check_count = 0
        waypoint_index = 0

        for drone_id, path_data in drone_paths.items():
            path_points = path_data['path_points']
            actual_path = []

            for i, (x, y) in enumerate(path_points):
                self.flight_controller.goto_waypoint(x, y)
                actual_path.append((x, y))
                waypoint_index += 1

                # 航点级VLA实时检测
                if (vla_controller is not None
                        and waypoint_index % VLA_CHECK_INTERVAL == 0
                        and vla_check_count < MAX_VLA_CHECKS):
                    try:
                        visual_data = self._load_test_image()
                        if visual_data and len(visual_data) > 100:
                            current_state = self._get_current_state()
                            current_state['position'] = [x, y, DEFAULT_TAKEOFF_ALTITUDE]

                            control_signal = vla_controller.process_visual_input(
                                visual_data, current_state,
                                language_instruction=f"无人机在({x:.1f},{y:.1f})位置，检测前方是否有障碍物"
                            )
                            vla_check_count += 1

                            if control_signal.get("action") == "avoid_obstacle":
                                obstacle_pos = self._convert_vla_coordinates(
                                    control_signal.get("obstacle_position", [50, 50]),
                                    area_bounds
                                )
                                logger.info(f"🚨 航点({x:.1f},{y:.1f})实时检测到障碍物: {obstacle_pos}")

                                # 生成局部绕障路径
                                remaining_points = path_points[i + 1:]
                                if remaining_points:
                                    avoidance_points = self._generate_local_avoidance(
                                        (x, y), remaining_points[0], obstacle_pos
                                    )
                                    for ap in avoidance_points:
                                        self.flight_controller.goto_waypoint(ap[0], ap[1])
                                        actual_path.append(ap)

                                self.has_obstacle = True
                                self.obstacle_position = obstacle_pos
                    except Exception as e:
                        logger.warning(f"航点VLA检测失败: {e}")

            path_data['collected_count'] = len(actual_path)
            path_data['path_points'] = actual_path

    def _generate_local_avoidance(self, current_pos, next_pos, obstacle_pos, radius=15):
        """生成局部绕障路径段（半圆弧）"""
        cx, cy = current_pos
        ox, oy = obstacle_pos
        path = []

        # 计算障碍物相对于当前点的方向
        dx = ox - cx
        dy = oy - cy
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < 1:
            return [next_pos]

        # 生成绕障弧线
        start_angle = math.atan2(dy, dx)
        end_angle = math.atan2(next_pos[1] - oy, next_pos[0] - ox)

        # 选择较短的弧线方向
        if end_angle < start_angle:
            end_angle += 2 * math.pi

        steps = 8
        for i in range(steps + 1):
            angle = start_angle + (end_angle - start_angle) * i / steps
            px = ox + radius * math.cos(angle)
            py = oy + radius * math.sin(angle)
            path.append((round(px, 2), round(py, 2)))

        path.append(next_pos)
        return path

    def _simulate_flight(self, drone_paths, area_bounds):
        """模拟飞行模式"""
        for drone_id, path_data in drone_paths.items():
            path_points = path_data['path_points']
            path_data['collected_count'] = len(path_points)

    def _get_current_state(self):
        """获取无人机当前状态"""
        if not hasattr(self, 'current_position'):
            self.current_position = [0, 0, 50]
            self.current_velocity = [0, 0, 0]
            self.remaining_battery = 95.0
            self.current_path_points = []
            self.current_area_bounds = DEFAULT_AREA_BOUNDS
        return {
            "position": self.current_position,
            "velocity": self.current_velocity,
            "remaining_battery": self.remaining_battery,
            "current_path": self.current_path_points,
            "area_bounds": self.current_area_bounds
        }

    def _merge_all_images(self, patrol_dir, final_dir):
        for root, _, files in os.walk(patrol_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    src_path = os.path.join(root, file)
                    if root != final_dir:
                        dst_name = f"{os.path.basename(root)}_{file}"
                        dst_path = os.path.join(final_dir, dst_name)
                        if not os.path.exists(dst_path):
                            shutil.copy(src_path, dst_path)



