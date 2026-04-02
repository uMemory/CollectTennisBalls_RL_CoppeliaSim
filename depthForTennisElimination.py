import time
import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ================= 1. 初始化与连接 =================
client = RemoteAPIClient()
sim = client.require('sim')

print("✅ 连接成功！深度传感器消除测试...")

# ── 句柄 ──
fl     = sim.getObject('/youBot/rollingJoint_fl')
fr     = sim.getObject('/youBot/rollingJoint_fr')
rl     = sim.getObject('/youBot/rollingJoint_rl')
rr     = sim.getObject('/youBot/rollingJoint_rr')
cam    = sim.getObject('/youBot/visionSensor')
youbot = sim.getObject('/youBot')

# ================= 2. 参数 =================
CAM_NEAR        = 0.01   # 与CoppeliaSim中visionSensor近裁面一致
CAM_FAR         = 15.0    # 与CoppeliaSim中visionSensor远裁面一致
DEPTH_THRESHOLD = 0.20   # 深度图判定消除阈值（米）
ANGLE_DEAD_ZONE = 0.08   # 转向死区（弧度，约5°）
ARRIVE_DIST     = 0.22   # 全局距离兜底阈值

# HSV 黄色阈值
YELLOW_LOW  = np.array([25, 120, 120], dtype=np.uint8)
YELLOW_HIGH = np.array([45, 255, 255], dtype=np.uint8)

# ================= 3. 电机控制 =================
def set_motors(vfl, vfr, vrl, vrr):
    sim.setJointTargetVelocity(fl, vfl)
    sim.setJointTargetVelocity(fr, vfr)
    sim.setJointTargetVelocity(rl, vrl)
    sim.setJointTargetVelocity(rr, vrr)

def stop():
    set_motors(0, 0, 0, 0)

def move_forward(speed=1.5):
    set_motors(speed, -speed, speed, -speed)

def turn_left(speed=1.0):
    set_motors(-speed, -speed, -speed, -speed)

def turn_right(speed=1.0):
    set_motors(speed, speed, speed, speed)

# ================= 4. 获取所有球句柄 =================
def get_all_balls():
    balls = []
    for i in range(1, 16):
        try:
            h = sim.getObject(f'/TennisBall_{i:02d}')
            balls.append(h)
        except:
            pass
    return balls

# ================= 5. 找最近球（全局坐标，仅用于导航） =================
def get_nearest_ball(balls):
    bot_pos = sim.getObjectPosition(youbot, sim.handle_world)
    nearest_h, nearest_dist, nearest_pos = None, float('inf'), None
    for h in balls:
        try:
            pos  = sim.getObjectPosition(h, sim.handle_world)
            dist = np.sqrt((bot_pos[0]-pos[0])**2 + (bot_pos[1]-pos[1])**2)
            if dist < nearest_dist:
                nearest_h, nearest_dist, nearest_pos = h, dist, pos
        except:
            pass
    return nearest_h, nearest_dist, nearest_pos

# ================= 6. 3D导航转向 =================
def navigate_toward(target_pos):
    """
    用全局坐标计算角度差驱动转向/前进
    误差大 → 转向，误差小 → 前进
    """
    bot_pos = sim.getObjectPosition(youbot, sim.handle_world)
    bot_ori = sim.getObjectOrientation(youbot, sim.handle_world)
    yaw     = bot_ori[2]

    dx = target_pos[0] - bot_pos[0]
    dy = target_pos[1] - bot_pos[1]
    target_angle = np.arctan2(dy, dx)

    # 归一化到 [-π, π]
    error = (target_angle - yaw + np.pi) % (2 * np.pi) - np.pi

    if abs(error) > ANGLE_DEAD_ZONE:
        speed = min(1.5, abs(error) * 1.2)
        turn_left(speed) if error > 0 else turn_right(speed)
    else:
        move_forward(speed=1.5)

# ================= 7. 深度图读取球的距离 =================
def get_ball_depth_distance():
    """
    RGB图中找黄色球的像素质心 (cx, cy)
    在深度图对应位置读取真实距离（米）
    返回 (found, depth_meters)
    """
    # RGB 图
    img, resolution = sim.getVisionSensorImg(cam)
    img_np = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    img_np = np.flipud(img_np)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask    = cv2.inRange(img_hsv, YELLOW_LOW, YELLOW_HIGH)
    kernel  = np.ones((3, 3), np.uint8)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 30:
        return False, None

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return False, None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # 深度图
    depth_buf, _ = sim.getVisionSensorDepth(cam)
    depth_np = np.frombuffer(depth_buf, dtype=np.float32).reshape(resolution[1], resolution[0])
    depth_np = np.flipud(depth_np)

    # 5×5 邻域中位数抗噪
    y1 = max(0, cy-2); y2 = min(resolution[1]-1, cy+2)
    x1 = max(0, cx-2); x2 = min(resolution[0]-1, cx+2)
    depth_raw    = float(np.median(depth_np[y1:y2, x1:x2]))

    # [0,1] → 实际距离（米）
    depth_meters = CAM_NEAR + depth_raw * (CAM_FAR - CAM_NEAR)

    return True, depth_meters

# ================= 8. 主测试循环 =================
print("🎾 测试：3D导航靠近 → 深度图触发消除")
print(f"   深度阈值: {DEPTH_THRESHOLD}m | 转向死区: {np.degrees(ANGLE_DEAD_ZONE):.1f}°")
print(f"   全局兜底: {ARRIVE_DIST}m（深度图失效时备用）")
print("=" * 55)

collected = 0

try:
    while True:
        balls = get_all_balls()
        print(f"balls数量: {len(balls)}")
        print(f"bot_ori: {sim.getObjectOrientation(youbot, sim.handle_world)}")
        if not balls:
            stop()
            print(f"\n🏆 全部收集完毕！共 {collected} 个")
            break

        # 找最近球
        nearest_h, nearest_dist, nearest_pos = get_nearest_ball(balls)

        # 读深度图
        found, depth_dist = get_ball_depth_distance()

        # 打印对比（同行刷新）
        depth_str = f"{depth_dist:.3f}m" if found else "未检测到"
        print(f"全局距离: {nearest_dist:.3f}m | 深度图: {depth_str}   ", end='\r')

        # ── 消除判定 ──
        # 优先：深度图距离小于阈值
        if found and depth_dist < DEPTH_THRESHOLD:
            stop()
            print(f"\n✅ [深度图触发] 消除！depth={depth_dist:.3f}m  global={nearest_dist:.3f}m")
            try:
                sim.removeObjects([nearest_h])
                collected += 1
            except:
                pass
            print(f"   已收集: {collected} 个 | 剩余: {len(balls)-1} 个")
            time.sleep(0.3)
            continue

        # 兜底：全局距离过近（深度图未检测到球时保险）
        if nearest_dist < ARRIVE_DIST:
            stop()
            print(f"\n⚠️  [全局兜底] 消除！global={nearest_dist:.3f}m  depth={depth_str}")
            try:
                sim.removeObjects([nearest_h])
                collected += 1
            except:
                pass
            print(f"   已收集: {collected} 个 | 剩余: {len(balls)-1} 个")
            time.sleep(0.3)
            continue

        # ── 3D导航靠近 ──
        navigate_toward(nearest_pos)
        time.sleep(0.05)

except KeyboardInterrupt:
    stop()
    print(f"\n🛑 测试中止 | 已收集: {collected} 个")
