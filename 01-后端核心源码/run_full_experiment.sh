#!/bin/bash
# 一键启动仿真+实验（Jetson专用）

# 杀死所有残留进程
pkill -f roscore
pkill -f px4
pkill -f gazebo
pkill -f gzserver
pkill -f gzclient
pkill -f mavros
pkill -f python3
sleep 3

# 启动roscore
gnome-terminal -- bash -c "roscore; exec bash"
sleep 3

# 启动PX4+Gazebo（带摄像头模型）
gnome-terminal -- bash -c "cd ~/PX4-Autopilot && make px4_sitl_default gazebo_iris_camera; exec bash"
sleep 15

# 启动MAVROS
gnome-terminal -- bash -c "roslaunch mavros px4.launch fcu_url:=\"udp://:14540@127.0.0.1:14557\" timesync_mode:=MAVLINK; exec bash"
sleep 5

# 运行路径规划实验
echo "✅ 所有环境启动完成，开始运行实验..."
python3 path_planning_ros_experiment.py

echo "🎉 实验完成！结果已保存为 path_planning_ros_results.csv"