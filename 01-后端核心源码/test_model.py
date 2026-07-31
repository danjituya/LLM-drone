# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os
import cv2

# ====================== 统一相对路径管理（适配新文件夹结构） ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_ROOT = os.path.dirname(BASE_DIR)  # 素材与源码 目录
PROJECT_ROOT = os.path.dirname(SOURCE_ROOT)  # 项目根目录
# 模型路径
MODEL_PATH = os.path.join(SOURCE_ROOT, "04-模型文件", "best.pt")
# 测试图片路径
TEST_IMG_DIR = os.path.join(SOURCE_ROOT, "05-数据集样本", "dataset_sample")
# 测试结果保存路径
RESULT_SAVE_PATH = os.path.join(SOURCE_ROOT, "06-辅助数据", "test_result.jpg")

if __name__ == "__main__":
    print("=== 模型测试脚本 ===")

    # 【修改点2】路径容错
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 模型文件不存在：\n{MODEL_PATH}")
        print(f"请先将训练好的 best.pt 放到 model/ 文件夹下")
        exit(0)

    # 找一张测试图
    test_img_path = None
    if os.path.exists(TEST_IMG_DIR):
        for f in os.listdir(TEST_IMG_DIR):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                # 先找子文件夹里的图
                sub_dir = os.path.join(TEST_IMG_DIR, f)
                if os.path.isdir(sub_dir):
                    for sub_f in os.listdir(sub_dir):
                        if sub_f.lower().endswith((".jpg", ".jpeg", ".png")):
                            test_img_path = os.path.join(sub_dir, sub_f)
                            break
                else:
                    test_img_path = os.path.join(TEST_IMG_DIR, f)
                if test_img_path:
                    break

    if not test_img_path:
        print(f"\n❌ 未找到测试图片，请在 {TEST_IMG_DIR} 下放一张示例图片")
        exit(0)

    print(f"测试图片：{test_img_path}")

    # 1. 加载模型
    model = YOLO(MODEL_PATH)

    # 2. 测试
    results = model(test_img_path)

    # 3. 解析结果
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            confidence = round(float(box.conf), 2)
            print(f"识别结果：{cls_name}，置信度：{confidence}")

    # 4. 可视化（加极致容错，无GUI也不崩溃）
    annotated_img = results[0].plot()
    try:
        # 尝试显示弹窗
        cv2.imshow("病虫害识别结果", annotated_img)
        cv2.waitKey(3000)  # 显示3秒自动关闭
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"⚠️  无法显示弹窗（无GUI环境），已保存结果图片")
    # 始终保存结果图片
    cv2.imwrite(RESULT_SAVE_PATH, annotated_img)
    print(f"✅ 结果图片已保存到：{RESULT_SAVE_PATH}")
    print("✅ 测试完成！")