# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os

# ====================== 统一相对路径管理（适配新文件夹结构） ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = os.path.dirname(BASE_DIR)  # 素材与源码 目录
PROJECT_ROOT = os.path.dirname(SOURCE_ROOT)  # 项目根目录
# 数据集配置文件路径
YAML_PATH = os.path.join(SOURCE_ROOT, "05-数据集样本", "dataset_sample", "plantvillage.yaml")
# 训练结果保存路径
PROJECT_PATH = os.path.join(SOURCE_ROOT, "06-辅助数据", "runs", "train")
if __name__ == "__main__":
    print("=== YOLO模型训练脚本 ===")
    # 【修改点2】路径容错判断
    if not os.path.exists(YAML_PATH):
        print(f"\n❌ 数据集配置文件不存在：\n{YAML_PATH}")
        print(f"\n提示：\n"
        f"1. 本脚本仅用于训练演示，完整数据集请自行下载\n"
        f"2. 下载后请修改 YAML_PATH 指向你的完整配置文件\n"
        f"3. 完整数据集下载链接请查看大赛《作品信息摘要》文档")
        exit(0)

    # 1. 加载预训练模型
    print("正在加载预训练模型...")
    model = YOLO("yolov8n.pt")

    # 自动判断设备：有GPU用GPU，无GPU用CPU
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    # 2. 开始训练
    print(f"开始训练，配置文件：{YAML_PATH}，训练设备：{device}")
    results = model.train(
        data=YAML_PATH,
        epochs=10,  # 新手先跑10轮测试
        imgsz=640,
        batch=8,
        lr0=0.01,
        device=device,  # 自动适配设备
        project=PROJECT_PATH,
        name="plantvillage_model",
        save=True,
        patience=5,
    )

    # 3. 验证
    metrics = model.val()
    print(f"验证集mAP50：{metrics.box.map50:.2f}")

    # 4. 保存路径提示
    best_model_path = os.path.join(PROJECT_PATH, "plantvillage_model", "weights", "best.pt")
    print(f"\n✅ 训练完成！")
    print(f"最优模型路径：{best_model_path}")
    print(f"请将 best.pt 复制到项目的 model/ 文件夹下供主程序使用")