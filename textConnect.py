"""
tennis_collector_v2.py
======================
阶段测试：全局坐标导航 + 深度图触发消除 + 球网绕行

目标：
  1. 直接读取每个网球的全局 XY 坐标（作弊导航，验证流程）
  2. YouBot 依次驶向最近的网球
     - 若目标球在对侧半场，自动规划绕网路径点（从网柱外侧绕过）
     - 转向使用低通滤波，避免车头左右抖动
  3. 全局距离 < ELIM_GLOBAL_DIST 时触发消除
  4. 全部消除后停止

球网约束（来自 lua 场景）：
  - Net_Collision_Wall: X=0, Y ∈ [-netW/2, +netW/2] = [-6.40, +6.40]m, 高 0.914m
  - 绕行通道：|Y| > NET_BYPASS_Y（网柱外侧）穿越 X=0

依赖：
  pip install coppeliasim_zmqremoteapi_client numpy opencv-python

运行前：
  - CoppeliaSim 已打开场景并启动仿真 (Play)
  - YouBot 已放置在场景中，visionSensor 已挂载并命名
"""

import time
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# =====================================================================
# 0. 参数配置
# =====================================================================

# --- 消除参数 ---
ELIM_GLOBAL_DIST = 0.35   # m，全局距离 < 此值触发消除（YouBot 半径约 0.3m）

# --- 导航参数 ---
BASE_SPEED    = 3.0   # rad/s 基础轮速
TURN_SPEED    = 2.5   # rad/s 最大转向轮速
RECOVER_SPEED = 1.5   # rad/s 卡顿后退速度

# --- 深度图参数（保留函数供后续视觉模块使用，当前消除不依赖深度）---
DEPTH_ROI_SIZE = 40     # px
SENSOR_NEAR    = 0.01   # m
SENSOR_FAR     = 15.0   # m

# --- 球网绕行参数（来自 lua 场景几何） ---
WAYPOINT_REACHED = 0.40   # m，到达路径点的距离容差

# --- 对象名称 ---
BALL_COUNT  = 12
SENSOR_NAME = "/visionSensor"
YOUBOT_NAME = "/youBot"

# =====================================================================
# 1. 初始化连接
# =====================================================================
print("=" * 60)
print("  网球收集器 v2 — 全局坐标导航 + 低通滤波转向 + 全局距离消除")
print("=" * 60)

client = RemoteAPIClient()
sim    = client.require('sim')
print("✅ ZMQ 连接成功")

# 轮子句柄
fl = sim.getObject('/rollingJoint_fl')
fr = sim.getObject('/rollingJoint_fr')
rl = sim.getObject('/rollingJoint_rl')
rr = sim.getObject('/rollingJoint_rr')
print("✅ 轮子句柄获取成功")

# YouBot 本体句柄（用于读取自身全局位置和朝向）
try:
    youbot_h = sim.getObject('/youBot')
except Exception:
    try:
        youbot_h = sim.getObject('/youBot_base')
    except Exception as e:
        print(f"⚠️  YouBot 根节点获取失败，请检查名称: {e}")
        youbot_h = None

# Vision Sensor 句柄
try:
    sensor_h = sim.getObject('/visionSensor')
    print("✅ visionSensor 句柄获取成功")
except Exception as e:
    sensor_h = None
    print(f"⚠️  visionSensor 获取失败: {e}")

# =====================================================================
# 2. 电机控制工具函数
# =====================================================================

def set_motors(vfl, vfr, vrl, vrr):
    sim.setJointTargetVelocity(fl, vfl)
    sim.setJointTargetVelocity(fr, vfr)
    sim.setJointTargetVelocity(rl, vrl)
    sim.setJointTargetVelocity(rr, vrr)

def stop_robot():
    set_motors(0, 0, 0, 0)

def move_forward(speed=BASE_SPEED):
    """麦克纳姆轮前进：[+, +, +, +]（依据 YoubotMovement.py 实测）"""
    set_motors(speed, speed, speed, speed)

def turn_left(speed=TURN_SPEED):
    """原地左转：[+, -, +, -]"""
    set_motors(speed, -speed, speed, -speed)

def turn_right(speed=TURN_SPEED):
    """原地右转：[-, +, -, +]"""
    set_motors(-speed, speed, -speed, speed)

def drive_to(smooth_angle, dist):
    """
    混合驱动控制，接收已平滑的角度误差。

    当误差 > 90°：原地纯转向，不前进（防止向反方向冲）
    当误差 ≤ 90°：前进 + 转向同时输出，cos 衰减前进速度

    smooth_angle : 低通滤波后的角度误差（弧度），正=左，负=右
    dist         : 与目标的直线距离（米）
    """
    FWD_KP   = 1.2
    TURN_KP  = 2.0
    FWD_MAX  = BASE_SPEED
    TURN_MAX = TURN_SPEED
    FWD_MIN  = 0.3
    TURN_THRESHOLD = math.radians(90)   # 误差超过此值只转不走

    v_turn = max(-TURN_MAX, min(TURN_MAX, TURN_KP * smooth_angle))

    if abs(smooth_angle) > TURN_THRESHOLD:
        # 误差过大：原地转向
        set_motors(v_turn, -v_turn, v_turn, -v_turn)
    else:
        # 误差在 90° 以内：混合前进 + 转向
        v_fwd = max(FWD_MIN, min(FWD_MAX, FWD_KP * dist * math.cos(smooth_angle)))
        set_motors(v_fwd + v_turn,
                   v_fwd - v_turn,
                   v_fwd + v_turn,
                   v_fwd - v_turn)


# =====================================================================
# 3. 机器人重置函数
# =====================================================================

# 默认出生位置与朝向（从 CoppeliaSim 控制台实测）
# 欧拉角 [Rx=-90°, Ry=0°, Rz=-90°] 对应四元数（仅用于 setObjectOrientation）
DEFAULT_POS = [6.400, -0.100, 0.096]
DEFAULT_ORI = [math.radians(-90.0), math.radians(0.1), math.radians(-90.0)]

def reset_robot(enabled=True, pos=None, ori=None):
    """
    将 YouBot 重置到指定位置和朝向，并确保静止。

    参数：
        enabled : bool  — True 时执行重置，False 时跳过
        pos     : list  — [x, y, z]，默认 DEFAULT_POS
        ori     : list  — [rx, ry, rz]（弧度欧拉角），默认 DEFAULT_ORI
    """
    if not enabled:
        print("⏭️  跳过重置，使用当前位置")
        return

    if youbot_h is None:
        print("⚠️  YouBot 句柄无效，跳过重置")
        return

    target_pos = pos if pos is not None else DEFAULT_POS
    target_ori = ori if ori is not None else DEFAULT_ORI

    stop_robot()
    time.sleep(0.1)

    sim.setObjectPosition(youbot_h, target_pos, sim.handle_world)
    sim.setObjectOrientation(youbot_h, target_ori, sim.handle_world)

    try:
        sim.resetDynamicObject(youbot_h)
    except Exception:
        pass

    time.sleep(0.3)

    # 验证时用四元数读取 yaw，与导航一致
    p   = sim.getObjectPosition(youbot_h, sim.handle_world)
    rx, ry, yaw = get_youbot_pose()
    print(f"✅ 重置完成：位置=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
          f"yaw={math.degrees(yaw):.1f}°")

# =====================================================================
# 4. 感知工具函数
# =====================================================================

def get_youbot_pose():
    """
    返回 YouBot 当前全局位置 (x, y) 和 朝向角 yaw (弧度)。

    使用四元数计算 yaw，避免欧拉角在 Rx=-90° 时的奇异点跳变问题。
    四元数公式：yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))

    YouBot 车头朝 -X 方向（初始 Rz=-90°），补偿 +π 使
    yaw=0 对应车头朝 +X，与 atan2 世界坐标系一致。
    """
    if youbot_h is None:
        return 0.0, 0.0, 0.0
    pos = sim.getObjectPosition(youbot_h, sim.handle_world)
    q   = sim.getObjectQuaternion(youbot_h, sim.handle_world)  # [qx, qy, qz, qw]
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]
    yaw = math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))
    yaw += math.pi
    while yaw >  math.pi: yaw -= 2 * math.pi
    while yaw < -math.pi: yaw += 2 * math.pi
    return pos[0], pos[1], yaw


def get_ball_position(ball_h):
    """返回网球全局 (x, y) 坐标"""
    pos = sim.getObjectPosition(ball_h, sim.handle_world)
    return pos[0], pos[1]


def get_depth_roi_mean():
    """
    从 visionSensor 读取深度图，返回中心 ROI 的平均归一化深度值。
    返回 None 表示传感器不可用。
    """
    if sensor_h is None:
        return None
    try:
        depth_buf, res = sim.getVisionSensorDepth(sensor_h, 0)
        W, H = res[0], res[1]

        # depth_buf 是原始字节串（bytes），每4字节为一个 float32
        if isinstance(depth_buf, (bytes, bytearray)):
            depth_np = np.frombuffer(depth_buf, dtype=np.float32).reshape(H, W)
        else:
            # 兼容旧版 API 直接返回 float 列表的情况
            depth_np = np.array(depth_buf, dtype=np.float32).reshape(H, W)

        # 取中心 ROI
        cx, cy   = W // 2, H // 2
        half     = DEPTH_ROI_SIZE // 2
        roi      = depth_np[cy - half: cy + half, cx - half: cx + half]
        mean_val = float(np.mean(roi))
        return mean_val
    except Exception as e:
        print(f"  ⚠️  深度图读取异常: {e}")
        return None


def depth_to_meters(norm_val):
    """将归一化深度值转换为实际距离（米）"""
    return SENSOR_NEAR + norm_val * (SENSOR_FAR - SENSOR_NEAR)


def get_rgb_image():
    """调试用：获取 RGB 图（可选用）"""
    if sensor_h is None:
        return None
    try:
        img_buf, res = sim.getVisionSensorImg(sensor_h, 0)
        W, H = res[0], res[1]

        if isinstance(img_buf, (bytes, bytearray)):
            img_np = np.frombuffer(img_buf, dtype=np.uint8).reshape(H, W, 3)
        else:
            img_np = np.array(img_buf, dtype=np.uint8).reshape(H, W, 3)

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception:
        return None

# =====================================================================
# 4. 球网绕行：路径点规划
# =====================================================================

def angle_diff(a, b):
    """计算从角度 b 到角度 a 的差值，归一化到 (-π, π]"""
    diff = a - b
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < -math.pi: diff += 2 * math.pi
    return diff


# 球网几何常量（来自 lua）
NET_WALL_Y    = 6.40    # m，碰撞墙 Y 方向半宽（netW/2 ≈ 6.40）
NET_BYPASS_Y  = 7.50    # m，绕行通道 Y 坐标，网柱外侧（外场边界 9.145m 内）
COURT_X_HALF  = 11.885  # m，球场半长（CL/2 = 23.77/2），用于限制路径点不出界

def needs_net_bypass(rx, bx):
    """
    只在 YouBot 与球跨越 X=0 两侧时才绕网。
    """
    return (rx > 0) != (bx > 0)


def plan_bypass_waypoints(rx, ry, bx, by):
    """
    基于 lua 球网几何的 L 形绕行路径。

    球网碰撞墙：X=0，|Y| ≤ 6.40m，高 0.914m
    绕行通道：  |Y| > 6.40m（网柱外侧），此处 X=0 无障碍可穿越

    路径设计（纯轴向移动，每段不斜穿球网）：

      WP1: (rx,      bypass_y)  ← 纵向（Y方向）移到侧道，X 不变，不穿网
      WP2: (0,       bypass_y)  ← 横向（X方向）移到网中线，Y 固定在侧道
      WP3: (bx_safe, bypass_y)  ← 横向进入目标半场，Y 固定在侧道
      WP4: (bx_safe, by_clamped)← 纵向移到球的 Y 附近，X 已在目标侧安全位置

    bypass_y 选择：取与当前 ry 同号的侧道（最短路径）
    bx_safe：目标侧离网至少 2m，防止接近时再次碰网
    """
    # 选最近侧道方向
    bypass_y = NET_BYPASS_Y if ry >= 0 else -NET_BYPASS_Y

    sign_b  = 1 if bx >= 0 else -1
    bx_safe = sign_b * max(abs(bx), 2.0)   # 目标侧离网至少 2m
    bx_safe = max(-COURT_X_HALF, min(COURT_X_HALF, bx_safe))  # 不出界

    # WP4 的 Y：朝球靠近但不进入网区（|Y| 保持安全）
    by_clamped = max(-8.0, min(8.0, by))

    waypoints = [
        (rx,      bypass_y),     # WP1: 纵向移到侧道
        (0.0,     bypass_y),     # WP2: 横向穿越 X=0
        (bx_safe, bypass_y),     # WP3: 横向进入目标半场
        (bx_safe, by_clamped),   # WP4: 纵向移到球附近
    ]

    # 如果已在侧道（|ry| 已超过 bypass_y），跳过 WP1
    if abs(ry) >= NET_BYPASS_Y:
        waypoints = waypoints[1:]

    return waypoints


def navigate_to_waypoint(tx, ty, label="WP"):
    """
    两阶段路径点导航，保证两点之间走直线：
      Phase 1 — 原地转向：纯转向指令，直到朝向误差 < ALIGN_TOL
      Phase 2 — 直线前进：纯前进指令，不叠加任何转向分量

    每次 dist 更新后检测是否已对准，若对准就前进，否则先转。
    卡顿检测：近距离未改善超过阈值时后退重试。
    """
    LOOP_DT       = 0.05
    ALIGN_TOL     = math.radians(8)   # 对准容差 8°，达到后才前进
    TURN_KP       = 1.5               # 转向比例增益（速度随误差大小变化）
    TURN_MIN      = 0.4               # 最小转速（防止太慢转不动）
    FWD_KP        = 1.0               # 前进比例增益
    FWD_MIN       = 0.5               # 最小前进速度
    STALL_LIMIT   = 100               # 卡顿帧数（约 5s）
    STALL_MARGIN  = 0.05              # m
    STALL_MIN_DIST = 1.5              # m，只在此距离内做卡顿检测
    MAX_STALL_RETRIES = 3             # 最多后退重试次数

    print(f"    ➡️  路径点 {label}: ({tx:.2f}, {ty:.2f})")

    stall_best    = float('inf')
    stall_count   = 0
    stall_retries = 0

    for step in range(3000):
        rx, ry, ryaw = get_youbot_pose()
        dx   = tx - rx
        dy   = ty - ry
        dist = math.hypot(dx, dy)

        # ── 到达判定 ─────────────────────────────────────────────
        if dist < WAYPOINT_REACHED:
            stop_robot()
            print(f"    ✓  到达 {label} (残差={dist:.2f}m)")
            return True

        target_angle = math.atan2(dy, dx)
        angle_err    = angle_diff(target_angle, ryaw)

        # ── 卡顿检测 ─────────────────────────────────────────────
        if dist < STALL_MIN_DIST:
            if dist < stall_best - STALL_MARGIN:
                stall_best  = dist
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= STALL_LIMIT:
                stall_retries += 1
                if stall_retries > MAX_STALL_RETRIES:
                    stop_robot()
                    print(f"    ❌ {label} 多次卡顿，放弃")
                    return False
                print(f"    ⚠️  {label} 卡顿（第{stall_retries}次），后退恢复...")
                set_motors(-RECOVER_SPEED, -RECOVER_SPEED,
                           -RECOVER_SPEED, -RECOVER_SPEED)
                time.sleep(1.2)
                stop_robot()
                time.sleep(0.2)
                stall_best  = float('inf')
                stall_count = 0
                continue
        else:
            stall_count = 0

        # ── 两阶段控制 ───────────────────────────────────────────
        if abs(angle_err) > ALIGN_TOL:
            # Phase 1：原地纯转向，不前进
            turn_spd = max(TURN_MIN, min(TURN_SPEED, TURN_KP * abs(angle_err)))
            if angle_err > 0:
                turn_left(turn_spd)
            else:
                turn_right(turn_spd)
            phase = f"转向 err={math.degrees(angle_err):+.1f}° spd={turn_spd:.2f}"
        else:
            # Phase 2：纯直线前进，不转向
            fwd_spd = max(FWD_MIN, min(BASE_SPEED, FWD_KP * dist))
            move_forward(fwd_spd)
            phase = f"前进 dist={dist:.2f}m spd={fwd_spd:.2f}"

        if step % 20 == 0:
            print(f"      [{label} step={step}] {phase}")

        time.sleep(LOOP_DT)

    stop_robot()
    print(f"    ⏰ 路径点 {label} 超时")
    return False



# =====================================================================
# 5. 导航策略：低通滤波转向 + 全局距离消除
# =====================================================================

def navigate_to_ball(ball_h, ball_name):
    """
    导航到指定网球并用全局距离触发消除。
    对目标角做低通滤波，使转向平滑、不抖动。
    若目标球在对侧半场，先执行绕网路径点导航。
    返回 True：成功消除  |  False：超时放弃
    """
    MAX_ITER = 2000
    LOOP_DT  = 0.05
    ALPHA    = 0.15   # 低通滤波系数（越小越平滑）

    print(f"\n  🎾 目标：{ball_name}")

    # ── 绕网检测 ────────────────────────────────────────────────
    rx, ry, _ = get_youbot_pose()
    bx, by    = get_ball_position(ball_h)

    if needs_net_bypass(rx, bx):
        print(f"  🔀 跨半场（YouBot X={rx:.1f}，球 X={bx:.1f}），启动绕网...")
        for i, (wx, wy) in enumerate(plan_bypass_waypoints(rx, ry, bx, by)):
            if not navigate_to_waypoint(wx, wy, label=f"WP{i+1}"):
                print(f"  ❌ 绕网 WP{i+1} 失败，放弃 {ball_name}")
                return False
        print(f"  ✅ 绕网完成，继续接近目标球...")
    else:
        print(f"  ✅ 同侧半场（YouBot X={rx:.1f}，球 X={bx:.1f}），直接导航")

    # ── 接近 + 全局距离消除循环 ──────────────────────────────────
    smooth_err   = None
    stall_best   = float('inf')   # 历史最近距离
    stall_count  = 0              # 距离未改善的连续帧数
    STALL_LIMIT  = 120            # 帧数上限（约 6s）后触发恢复
    STALL_MARGIN = 0.05           # m，距离需改善超过此值才重置计数
    STALL_MIN_DIST = 1.0          # m，只在距离小于此值时才做卡顿检测，避免转向期间误触发

    for step in range(MAX_ITER):
        rx, ry, ryaw = get_youbot_pose()
        bx, by       = get_ball_position(ball_h)

        dx   = bx - rx
        dy   = by - ry
        dist = math.hypot(dx, dy)

        # ── 全局距离消除判定 ─────────────────────────────────────
        if dist < ELIM_GLOBAL_DIST:
            stop_robot()
            print(f"  💥 距离触发！全局={dist:.3f}m < {ELIM_GLOBAL_DIST}m — 消除 {ball_name}")
            try:
                sim.removeObjects([ball_h])
                print(f"  ✅ {ball_name} 已消除")
            except Exception as e:
                print(f"  ⚠️  removeObjects 失败: {e}")
            time.sleep(0.3)
            return True

        # ── 卡顿检测（仅在近距离时生效）────────────────────────────
        if dist < STALL_MIN_DIST:
            if dist < stall_best - STALL_MARGIN:
                stall_best  = dist
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count = 0   # 远距离时重置，不累积

        if stall_count >= STALL_LIMIT:
            # 触发恢复：后退 1s 脱离障碍，然后重置平滑器重新导航
            print(f"  ⚠️  step={step} 卡顿检测！dist={dist:.2f}m 未改善 {STALL_LIMIT} 帧，后退恢复...")
            set_motors(-RECOVER_SPEED, -RECOVER_SPEED, -RECOVER_SPEED, -RECOVER_SPEED)
            time.sleep(1.0)
            stop_robot()
            time.sleep(0.2)
            smooth_err  = None   # 重置平滑器，重新计算方向
            stall_best  = float('inf')
            stall_count = 0
            continue

        # ── 低通滤波转向角 ───────────────────────────────────────
        raw_err = angle_diff(math.atan2(dy, dx), ryaw)
        if smooth_err is None:
            smooth_err = raw_err
        else:
            smooth_err = (1 - ALPHA) * smooth_err + ALPHA * raw_err

        # ── 日志（每 20 步）────────────────────────────────────────
        if step % 20 == 0:
            print(f"    step={step:4d} | 距离={dist:.2f}m | "
                  f"误差(原始)={math.degrees(raw_err):+.1f}° | "
                  f"误差(平滑)={math.degrees(smooth_err):+.1f}°")

        # ── 驱动 ─────────────────────────────────────────────────
        drive_to(smooth_err, dist)
        time.sleep(LOOP_DT)

    stop_robot()
    print(f"  ⏰ 超过最大步数，放弃 {ball_name}")
    return False

# =====================================================================
# 6. 主流程：收集所有网球
# =====================================================================

def collect_all_balls():
    # 获取所有网球句柄
    ball_handles = {}
    print("\n📋 正在获取网球句柄...")
    for i in range(1, BALL_COUNT + 1):
        name = f"TennisBall_{i:02d}"
        try:
            h = sim.getObject(f"/{name}")
            ball_handles[name] = h
            print(f"  ✓ {name}: handle={h}")
        except Exception as e:
            print(f"  ✗ {name}: 获取失败 ({e})")

    if not ball_handles:
        print("❌ 未找到任何网球，退出")
        return

    print(f"\n🎯 共找到 {len(ball_handles)} 个网球，开始收集...\n")
    collected = 0
    failed    = 0

    remaining = dict(ball_handles)

    while remaining:
        # 每次选择距离 YouBot 最近的球（全局坐标贪心策略）
        rx, ry, _ = get_youbot_pose()
        nearest_name = None
        nearest_dist = float('inf')
        nearest_h    = None

        for name, h in remaining.items():
            try:
                bx, by = get_ball_position(h)
                d = math.hypot(bx - rx, by - ry)
                if d < nearest_dist:
                    nearest_dist  = d
                    nearest_name  = name
                    nearest_h     = h
            except Exception:
                # 球可能已被删除（外部操作），清理掉
                continue

        if nearest_name is None:
            break

        success = navigate_to_ball(nearest_h, nearest_name)

        if success:
            collected += 1
        else:
            failed += 1

        del remaining[nearest_name]

        # 简短停顿后继续下一个
        stop_robot()
        time.sleep(0.5)

    # ── 结果汇报 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  🏁 收集完成！")
    print(f"     成功消除：{collected} 个")
    print(f"     放弃/失败：{failed} 个")
    print("=" * 60)


# =====================================================================
# 7. 入口
# =====================================================================
def motor_selftest():
    """
    启动时自检：依次执行前进/转向，打印位置和朝向变化量，
    帮助确认指令与实际运动的对应关系。
    每个动作执行 1 秒后停止并打印结果。
    """
    print("\n🔧 电机自检（每个动作 1 秒）...")
    tests = [
        ("前进",  lambda: move_forward(2.0)),
        ("前进",  lambda: move_forward(2.0)),
        ("前进",  lambda: move_forward(2.0)),
        ("左转",  lambda: turn_left(2.0)),
        ("左转",  lambda: turn_left(2.0)),
        ("左转",  lambda: turn_left(2.0)),
        ("右转",  lambda: turn_right(2.0)),
        ("右转",  lambda: turn_right(2.0)),
        ("右转",  lambda: turn_right(2.0)),
    ]
    for name, action in tests:
        x0, y0, yaw0 = get_youbot_pose()
        action()
        time.sleep(1.0)
        stop_robot()
        time.sleep(0.3)
        x1, y1, yaw1 = get_youbot_pose()
        dyaw = math.degrees(angle_diff(yaw1, yaw0))
        ddist = math.hypot(x1 - x0, y1 - y0)
        print(f"  [{name}] 位移={ddist:.3f}m  朝向变化={dyaw:+.1f}°")
    print("🔧 自检完成\n")


if __name__ == "__main__":
    print("\n⏳ 等待 1 秒让仿真稳定...")
    time.sleep(1.0)

    # ── 重置小车位置 ──────────────────────────────────────────────
    # 改为 enabled=False 可跳过重置，直接从当前位置开始
    reset_robot(enabled=True)
    # 在 reset_robot() 之后、motor_selftest() 之前插入
    m = sim.getObjectMatrix(youbot_h, sim.handle_world)
    print(f"变换矩阵前3列（世界坐标下的局部XYZ轴）:")
    print(f"  局部X在世界: ({m[0]:.3f}, {m[4]:.3f}, {m[8]:.3f})")
    print(f"  局部Y在世界: ({m[1]:.3f}, {m[5]:.3f}, {m[9]:.3f})")
    print(f"  局部Z在世界: ({m[2]:.3f}, {m[6]:.3f}, {m[10]:.3f})")
    print(f"  位置:        ({m[3]:.3f}, {m[7]:.3f}, {m[11]:.3f})")

    # 然后前进 1 秒看位移方向
    x0, y0, _ = sim.getObjectPosition(youbot_h, sim.handle_world)
    move_forward(2.0)
    time.sleep(1.0)
    stop_robot()
    time.sleep(0.3)
    x1, y1, _ = sim.getObjectPosition(youbot_h, sim.handle_world)
    print(f"  前进位移: dx={x1 - x0:.3f}, dy={y1 - y0:.3f}")
    motor_selftest()

    try:
        collect_all_balls()
    except KeyboardInterrupt:
        print("\n🛑 用户中断，停止机器人...")
    finally:
        stop_robot()
        print("✅ 电机已停止，程序退出")