"""
test_hsv_yolo.py
================
对比测试：HSV 颜色检测 vs YOLO-World 在不同距离下的检测效果

HSV 检测原理：
  网球颜色 = ITF "optic yellow"（荧光黄绿）
  在 HSV 空间中有非常独特的色相范围，与蓝色球场、绿色外场、白色线条
  都有明显区别，哪怕只有几个像素也能检测到。

运行：
  python test_hsv_yolo.py
"""

import time
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ultralytics import YOLOWorld

# =====================================================================
# 1. 连接仿真
# =====================================================================
print("=" * 60)
print("  HSV + YOLO-World 混合检测测试")
print("=" * 60)

client = RemoteAPIClient()
sim = client.require('sim')
print("✅ ZMQ 连接成功")

sensor_h = sim.getObject('/visionSensor')
youbot_h = sim.getObject('/youBot')
fl = sim.getObject('/rollingJoint_fl')
fr = sim.getObject('/rollingJoint_fr')
rl = sim.getObject('/rollingJoint_rl')
rr = sim.getObject('/rollingJoint_rr')
print("✅ 句柄获取成功")


def set_motors(vfl, vfr, vrl, vrr):
    sim.setJointTargetVelocity(fl, vfl)
    sim.setJointTargetVelocity(fr, vfr)
    sim.setJointTargetVelocity(rl, vrl)
    sim.setJointTargetVelocity(rr, vrr)


def stop_robot():
    set_motors(0, 0, 0, 0)


def get_rgb_image():
    """从 visionSensor 获取图像，返回 BGR numpy 数组"""
    img_buf, res = sim.getVisionSensorImg(sensor_h, 0)
    W, H = res[0], res[1]
    if isinstance(img_buf, (bytes, bytearray)):
        img_np = np.frombuffer(img_buf, dtype=np.uint8).reshape(H, W, 3)
    else:
        img_np = np.array(img_buf, dtype=np.uint8).reshape(H, W, 3)
    img_np = np.flipud(img_np)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


# =====================================================================
# 2. HSV 网球检测器
# =====================================================================

# --- HSV 阈值（针对仿真中的 optic yellow 网球）---
# 网球 Lua 颜色: {0.85, 0.92, 0.10} (RGB 归一化)
# 换算 RGB255: (217, 235, 26)
# 换算 HSV:    H≈34, S≈222, V≈235
#
# 但 CoppeliaSim 渲染会受光照影响，所以范围要宽一些
# 同时发光属性 feltEmit={0.06, 0.07, 0.01} 会让球偏亮

HSV_LOWER = np.array([20, 80, 80])     # H 下限, S 下限, V 下限
HSV_UPPER = np.array([55, 255, 255])   # H 上限, S 上限, V 上限
MIN_AREA  = 4                           # 最小像素面积（远距离球可能很小）


def detect_hsv(img_bgr):
    """
    HSV 颜色检测网球。

    返回:
        detections: list of dict, 每个元素:
            cx, cy    : 质心像素坐标
            area      : 像素面积
            bbox      : (x, y, w, h)
            angle     : 水平偏移角（归一化 [-1, 1]，0=画面中心）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # 形态学操作：去噪 + 填充
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H_img, W_img = img_bgr.shape[:2]
    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        x, y, w, h = cv2.boundingRect(cnt)

        # 长宽比过滤：网球应该接近圆形
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if aspect > 3.0:  # 太扁长的不是球
            continue

        # 水平偏移角归一化
        angle = (cx - W_img / 2) / (W_img / 2)  # [-1, 1]

        detections.append({
            "cx": cx, "cy": cy,
            "area": area,
            "bbox": (x, y, w, h),
            "angle": angle,
        })

    # 按面积降序（最大的最可能是最近的球）
    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections, mask


# =====================================================================
# 3. YOLO-World 加载
# =====================================================================
print("\n📦 正在加载 YOLO-World...")
model = YOLOWorld("yolov8s-worldv2")
model.set_classes(["tennis ball", "ball", "yellow ball", "small yellow sphere"])
print("✅ YOLO-World 加载完成")


def detect_yolo(img_bgr):
    """YOLO-World 检测，返回 results"""
    results = model.predict(
        source=img_bgr,
        conf=0.05,
        iou=0.3,
        verbose=False,
        device=0,
    )
    return results[0]


# =====================================================================
# 4. 混合检测 + 可视化
# =====================================================================

def detect_and_visualize(img_bgr, save_prefix, frame_label=""):
    """
    同时运行 HSV 和 YOLO-World，生成对比可视化。
    保存 3 张图：原图、HSV 结果、YOLO 结果、混合结果
    """
    H_img, W_img = img_bgr.shape[:2]

    # --- HSV 检测 ---
    t0 = time.time()
    hsv_dets, hsv_mask = detect_hsv(img_bgr)
    t_hsv = (time.time() - t0) * 1000

    # --- YOLO 检测 ---
    t0 = time.time()
    yolo_result = detect_yolo(img_bgr)
    t_yolo = (time.time() - t0) * 1000

    yolo_boxes = yolo_result.boxes

    # --- 打印结果 ---
    print(f"\n  📸 {frame_label}")
    print(f"     HSV:  {len(hsv_dets)} 个目标  ({t_hsv:.1f}ms)")
    for i, d in enumerate(hsv_dets):
        print(f"       [{i}] area={d['area']:.0f}px  "
              f"center=({d['cx']:.0f},{d['cy']:.0f})  "
              f"angle={d['angle']:+.2f}")

    print(f"     YOLO: {len(yolo_boxes)} 个目标  ({t_yolo:.1f}ms)")
    for i, box in enumerate(yolo_boxes):
        conf = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        print(f"       [{i}] {cls_name} conf={conf:.3f}  "
              f"bbox=({x1},{y1},{x2},{y2})")

    # --- 可视化：混合结果图 ---
    img_combined = img_bgr.copy()

    # 画 HSV 检测（蓝色框）
    for d in hsv_dets:
        x, y, w, h = d["bbox"]
        cv2.rectangle(img_combined, (x, y), (x+w, y+h), (255, 150, 0), 2)
        label = f"HSV a={d['area']:.0f}"
        cv2.putText(img_combined, label, (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)

    # 画 YOLO 检测（绿色框）
    for box in yolo_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cv2.rectangle(img_combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"YOLO {conf:.2f}"
        cv2.putText(img_combined, label, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 图例
    cv2.putText(img_combined, "Blue=HSV  Green=YOLO", (5, H_img - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imwrite(f"{save_prefix}_combined.png", img_combined)

    # HSV mask 可视化
    mask_color = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{save_prefix}_hsv_mask.png", mask_color)

    # 原图
    cv2.imwrite(f"{save_prefix}_raw.png", img_bgr)

    print(f"     💾 保存: {save_prefix}_combined.png / _hsv_mask.png / _raw.png")

    return hsv_dets, yolo_boxes


# =====================================================================
# 5. 运行测试
# =====================================================================

# --- 测试 1：当前视角 ---
print("\n" + "=" * 60)
print("  测试 1：当前视角")
print("=" * 60)
img = get_rgb_image()
detect_and_visualize(img, "test_current", "当前视角")

# --- 测试 2：转一圈搜索 ---
print("\n" + "=" * 60)
print("  测试 2：旋转搜索（12 步 × 30°）")
print("=" * 60)

hsv_total = 0
yolo_total = 0
hsv_only_count = 0   # HSV 检测到但 YOLO 没有的帧数

TURN_SPEED = 1.5
TURN_TIME = 0.35
NUM_STEPS = 12

for step in range(NUM_STEPS):
    set_motors(TURN_SPEED, -TURN_SPEED, TURN_SPEED, -TURN_SPEED)
    time.sleep(TURN_TIME)
    stop_robot()
    time.sleep(0.3)

    img = get_rgb_image()
    hsv_dets, yolo_boxes = detect_and_visualize(
        img, f"test_step_{step:02d}", f"步骤 {step+1}/{NUM_STEPS}")

    hsv_total += len(hsv_dets)
    yolo_total += len(yolo_boxes)
    if len(hsv_dets) > 0 and len(yolo_boxes) == 0:
        hsv_only_count += 1

stop_robot()

# =====================================================================
# 6. 汇总
# =====================================================================
print("\n" + "=" * 60)
print("  📊 检测汇总")
print("=" * 60)
print(f"  总帧数: {NUM_STEPS + 1}")
print(f"  HSV 总检测数:  {hsv_total}")
print(f"  YOLO 总检测数: {yolo_total}")
print(f"  HSV 独占帧数:  {hsv_only_count} (HSV 检测到但 YOLO 未检测到)")
print()

if hsv_total > yolo_total:
    print("  ✅ HSV 检测范围明显大于 YOLO-World")
    print("     → 混合方案有效：HSV 远距离发现，YOLO 近距离确认")
elif hsv_total == 0:
    print("  ⚠️ HSV 也未检测到任何目标")
    print("     → 需要调整 HSV 阈值，请查看 _hsv_mask.png 确认颜色范围")
    print("     → 可以手动打开 _raw.png，用取色器确认网球的实际 HSV 值")
else:
    print("  ℹ️ HSV 和 YOLO 检测数量接近")
    print("     → HSV 辅助效果有限，但作为备份仍然有价值")

print("\n  💡 如果 HSV 阈值不对，可以运行以下代码查看网球的实际 HSV 值：")
print('     img = cv2.imread("test_current_raw.png")')
print('     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)')
print('     # 鼠标点击网球区域，查看 hsv[y, x] 的值')
print("=" * 60)