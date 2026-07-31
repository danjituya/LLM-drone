# -*- coding: utf-8 -*-
import os
import shutil
import random
from pathlib import Path

# ====================== 统一相对路径管理（适配新文件夹结构） ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = os.path.dirname(BASE_DIR)  # 素材与源码 目录
PROJECT_ROOT = os.path.dirname(SOURCE_ROOT)  # 项目根目录
# 源数据集目录
SOURCE_DIR = os.path.join(SOURCE_ROOT, "05-数据集样本", "dataset_sample")
# 目标YOLO格式目录
TARGET_DIR = os.path.join(SOURCE_ROOT, "05-数据集样本", "dataset_sample_yolo")

FILTER_CROP = None

if __name__ == "__main__":
    print("=== 数据集转换脚本（仅用于示例） ===")
    # 【修改点2】路径容错
    if not os.path.exists(SOURCE_DIR) or len(os.listdir(SOURCE_DIR)) == 0:
        print(f"\n❌ 示例数据集目录为空：\n{SOURCE_DIR}")
        print(f"\n提示：\n"
              f"1. 本脚本仅用于演示数据集转换流程\n"
              f"2. 完整PlantVillage数据集请自行下载，并修改 SOURCE_DIR 指向完整路径\n"
              f"3. 完整数据集下载链接请查看大赛《作品信息摘要》文档")
        exit(0)

    # 初始化路径
    images_train = os.path.join(TARGET_DIR, "images", "train")
    images_val = os.path.join(TARGET_DIR, "images", "val")
    labels_train = os.path.join(TARGET_DIR, "labels", "train")
    labels_val = os.path.join(TARGET_DIR, "labels", "val")
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(images_val, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)
    os.makedirs(labels_val, exist_ok=True)

    # 1. 获取类别和图片
    class_names = []
    image_paths = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for dir_name in dirs:
            class_names.append(dir_name)
            class_dir = os.path.join(root, dir_name)
            for file in os.listdir(class_dir):
                if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".JPG")):
                    image_paths.append((os.path.join(class_dir, file), dir_name))

    class_names = list(set(class_names))
    class_names.sort()
    class2id = {name: idx for idx, name in enumerate(class_names)}

    print(f"识别到的类别数：{len(class_names)}")
    print(f"总图片数：{len(image_paths)}")

    if len(image_paths) == 0:
        print("❌ 未找到任何图片！")
        exit(0)

    # 2. 划分数据集
    random.seed(42)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * 0.7)
    train_images = image_paths[:split_idx]
    val_images = image_paths[split_idx:]

    # 3. 处理图片
    def process_images(image_list, img_dst, lbl_dst):
        for img_path, class_name in image_list:
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(img_dst, img_name))
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(lbl_dst, lbl_name)
            with open(lbl_path, "w") as f:
                f.write(f"{class2id[class_name]} 0.5 0.5 1.0 1.0")

    process_images(train_images, images_train, labels_train)
    process_images(val_images, images_val, labels_val)

    # 4. 生成yaml（使用相对路径，适配任何电脑）
    yaml_content = f"""
    # 数据集配置文件（适配YOLOv8）
    path: ./
    train: images/train
    val: images/val
    nc: {len(class_names)}
    names: {class_names}
    """
    yaml_path = os.path.join(TARGET_DIR, "plantvillage.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n✅ 转换完成！")
    print(f"配置文件：{yaml_path}")
    print(f"提示：完整数据集转换请修改 SOURCE_DIR 为你的完整数据集路径")