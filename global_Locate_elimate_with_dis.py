import time
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# --- 消除参数 ---
ELIM_GLOBAL_DIST = 0.35   # m，全局距离 < 此值触发消除（YouBot 半径约 0.3m）

# --- 导航参数 ---
BASE_SPEED    = 7.5
TURN_SPEED    = 3.5
RECOVER_SPEED = 1.5

# --- 深度图参数（保留函数供后续视觉模块使用）---
DEPTH_ROI_SIZE = 40     # px
SENSOR_NEAR    = 0.01   # m
SENSOR_FAR     = 15.0   # m

# --- 球网绕行参数---
NET_WALL_Y    = 6.40    # m
NET_BYPASS_Y  = 7.40    # m
COURT_X_HALF  = 11.885  # m，球场半长（CL/2）
NET_NEAR_THRESHOLD = 0.5  # m
WAYPOINT_REACHED = 0.50   # m

# --- 对象名称 ---
BALL_COUNT  = 12
SENSOR_NAME = "/visionSensor"
YOUBOT_NAME = "/youBot"


print("=" * 60)
print("  网球收集器 v3 — 矩阵yaw + 统一导航 + 简化绕网")
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
        print(f"⚠YouBot 根节点获取失败，请检查名称: {e}")
        youbot_h = None

# Vision Sensor 句柄
try:
    sensor_h = sim.getObject('/visionSensor')
    print("visionSensor 句柄获取成功")
except Exception as e:
    sensor_h = None
    print(f"visionSensor 获取失败: {e}")


# 2. 电机控制工具函数
def set_motors(vfl, vfr, vrl, vrr):
    sim.setJointTargetVelocity(fl, vfl)
    sim.setJointTargetVelocity(fr, vfr)
    sim.setJointTargetVelocity(rl, vrl)
    sim.setJointTargetVelocity(rr, vrr)

def stop_robot():
    set_motors(0, 0, 0, 0)

def move_forward(speed=BASE_SPEED):
    """麦克纳姆轮前进：[+, +, +, +]"""
    set_motors(speed, speed, speed, speed)

def turn_left(speed=TURN_SPEED):
    """原地左转：[+, -, +, -]"""
    set_motors(speed, -speed, speed, -speed)

def turn_right(speed=TURN_SPEED):
    """原地右转：[-, +, -, +]"""
    set_motors(-speed, speed, -speed, speed)

def drive_to(smooth_angle, dist):
    # 混合驱动控制，接收已平滑的角度误差。
    FWD_KP   = 1.8
    TURN_KP  = 2.0
    FWD_MAX  = BASE_SPEED
    TURN_MAX = TURN_SPEED
    FWD_MIN  = 0.3
    TURN_THRESHOLD = math.radians(90)

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


# 3. 机器人重置函数
DEFAULT_POS = [6.400, -0.100, 0.096]
DEFAULT_ORI = [math.radians(-90.0), math.radians(0.1), math.radians(-90.0)]

def reset_robot(enabled=True, pos=None, ori=None):
    """将 YouBot 重置到指定位置和朝向，并确保静止。"""
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

    p = sim.getObjectPosition(youbot_h, sim.handle_world)
    rx, ry, yaw = get_youbot_pose()
    print(f"✅ 重置完成：位置=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) "
          f"yaw={math.degrees(yaw):.1f}°")


def get_youbot_pose():

    if youbot_h is None:
        return 0.0, 0.0, 0.0
    pos = sim.getObjectPosition(youbot_h, sim.handle_world)
    m   = sim.getObjectMatrix(youbot_h, sim.handle_world)
    # 车头 = 局部 -Z 轴
    head_wx = -m[2]   # 车头在世界 X 分量
    head_wy = -m[6]   # 车头在世界 Y 分量
    yaw = math.atan2(head_wy, head_wx)
    return pos[0], pos[1], yaw


def angle_diff(a, b):
    diff = a - b
    while diff >  math.pi: diff -= 2 * math.pi
    while diff < -math.pi: diff += 2 * math.pi
    return diff

def get_ball_position(ball_h):
    """返回网球全局 (x, y) 坐标"""
    pos = sim.getObjectPosition(ball_h, sim.handle_world)
    return pos[0], pos[1]


def get_depth_roi_mean():
    if sensor_h is None:
        return None
    try:
        depth_buf, res = sim.getVisionSensorDepth(sensor_h, 0)
        W, H = res[0], res[1]

        if isinstance(depth_buf, (bytes, bytearray)):
            depth_np = np.frombuffer(depth_buf, dtype=np.float32).reshape(H, W)
        else:
            depth_np = np.array(depth_buf, dtype=np.float32).reshape(H, W)

        cx, cy   = W // 2, H // 2
        half     = DEPTH_ROI_SIZE // 2
        roi      = depth_np[cy - half: cy + half, cx - half: cx + half]
        mean_val = float(np.mean(roi))
        return mean_val
    except Exception as e:
        print(f"  ⚠️  深度图读取异常: {e}")
        return None


def depth_to_meters(norm_val):
    return SENSOR_NEAR + norm_val * (SENSOR_FAR - SENSOR_NEAR)


def get_rgb_image():
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


def navigate_to_target(tx, ty, reach_dist, label="TARGET", max_iter=2000):
    LOOP_DT  = 0.05
    ALPHA    = 0.15   # 低通滤波系数

    # 卡顿检测参数
    STALL_LIMIT     = 120    # 帧数上限（约 6s）
    STALL_MARGIN    = 0.05   # m
    STALL_MIN_DIST  = 1.5    # m，只在此距离内做卡顿检测
    MAX_STALL_RETRIES = 3

    smooth_err    = None
    stall_best    = float('inf')
    stall_count   = 0
    stall_retries = 0

    for step in range(max_iter):
        rx, ry, ryaw = get_youbot_pose()
        dx   = tx - rx
        dy   = ty - ry
        dist = math.hypot(dx, dy)

        # ── 到达判定 ─────────────────────────────────────────────
        if dist < reach_dist:
            stop_robot()
            print(f"    ✓ 到达 {label} (残差={dist:.2f}m)")
            return True

        # ── 卡顿检测 ─────────────────────────────────────────────
        if dist < STALL_MIN_DIST:
            if dist < stall_best - STALL_MARGIN:
                stall_best  = dist
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count = 0

        if stall_count >= STALL_LIMIT:
            stall_retries += 1
            if stall_retries > MAX_STALL_RETRIES:
                stop_robot()
                print(f"    ❌ {label} 多次卡顿，放弃")
                return False
            print(f"    ⚠️ {label} 卡顿（第{stall_retries}次），后退恢复...")
            set_motors(-RECOVER_SPEED, -RECOVER_SPEED,
                       -RECOVER_SPEED, -RECOVER_SPEED)
            time.sleep(1.0)
            stop_robot()
            time.sleep(0.2)
            smooth_err  = None
            stall_best  = float('inf')
            stall_count = 0
            continue

        # ── 低通滤波转向角 ───────────────────────────────────────
        raw_err = angle_diff(math.atan2(dy, dx), ryaw)
        if smooth_err is None:
            smooth_err = raw_err
        else:
            smooth_err = (1 - ALPHA) * smooth_err + ALPHA * raw_err

        # ── 日志 ─────────────────────────────────────────────────
        if step % 30 == 0:
            print(f"    [{label} step={step:4d}] dist={dist:.2f}m "
                  f"err={math.degrees(raw_err):+.1f}° "
                  f"smooth={math.degrees(smooth_err):+.1f}°")

        # ── 驱动 ─────────────────────────────────────────────────
        drive_to(smooth_err, dist)
        time.sleep(LOOP_DT)

    stop_robot()
    print(f"    ⏰ {label} 超时")
    return False



# 6. 球网绕行
def needs_net_bypass(rx, bx):
    return (rx > 0) != (bx > 0)


def plan_bypass_waypoints(rx, ry, bx, by):
    # 选最近侧道
    bypass_y = NET_BYPASS_Y if ry >= 0 else -NET_BYPASS_Y

    # 当前侧安全 X（不太靠近网）
    sign_r  = 1 if rx >= 0 else -1
    rx_safe = sign_r * max(abs(rx), 1.0)
    rx_safe = max(-COURT_X_HALF, min(COURT_X_HALF, rx_safe))

    # 目标侧安全 X（离网至少 1.5m）
    sign_b  = 1 if bx >= 0 else -1
    bx_safe = sign_b * max(abs(bx), 1.5)
    bx_safe = max(-COURT_X_HALF, min(COURT_X_HALF, bx_safe))

    # 目标 Y 限制在外场范围内
    by_clamped = max(-8.5, min(8.5, by))
    waypoints = [
        (rx_safe, bypass_y),
        (bx_safe, bypass_y),
        (bx_safe, by_clamped),
    ]
    # 如果已在侧道附近（|ry| 已接近 bypass_y），跳过 WP1
    if abs(ry) >= NET_BYPASS_Y - 0.5:
        waypoints = waypoints[1:]

    return waypoints


# 7. 导航策略：绕网 + 接近消除
def navigate_to_ball(ball_h, ball_name):
    print(f"\n  目标：{ball_name}")

    rx, ry, _ = get_youbot_pose()
    bx, by    = get_ball_position(ball_h)

    # ── 绕网判断 ────────────────────────────────────────────────
    if needs_net_bypass(rx, bx):
        print(f" 跨半场（YouBot X={rx:.1f}，球 X={bx:.1f}），启动绕网...")
        waypoints = plan_bypass_waypoints(rx, ry, bx, by)
        for i, (wx, wy) in enumerate(waypoints):
            print(f"WP{i+1}: ({wx:.2f}, {wy:.2f})")
            if not navigate_to_target(wx, wy, WAYPOINT_REACHED, label=f"WP{i+1}"):
                print(f" 绕网 WP{i+1} 失败，放弃 {ball_name}")
                return False
        print(f" 绕网完成，继续接近目标球...")
    elif abs(bx) < NET_NEAR_THRESHOLD:
        # 球紧贴球网（|bx| < 0.5m），从当前侧绕到球的 Y 位置再靠近
        print(f" 球贴近球网（bx={bx:.2f}），从侧面接近...")
        # 先移到与球同 Y 但 X 保持安全距离的位置
        safe_x = (1.0 if rx >= 0 else -1.0) * 1.0  # 保持在当前侧 1m 处
        if not navigate_to_target(safe_x, by, WAYPOINT_REACHED, label="侧面接近"):
            print(f" 侧面接近失败，放弃 {ball_name}")
            return False
    else:
        print(f" 同侧半场（YouBot X={rx:.1f}，球 X={bx:.1f}），直接导航")

    # ── 接近 + 消除 ──────────────────────────────────────────────
    # 动态追踪球的实时位置（球可能因物理引擎微移）
    MAX_ITER = 2000
    LOOP_DT  = 0.05
    ALPHA    = 0.15

    STALL_LIMIT    = 120
    STALL_MARGIN   = 0.05
    STALL_MIN_DIST = 1.0

    smooth_err  = None
    stall_best  = float('inf')
    stall_count = 0

    for step in range(MAX_ITER):
        rx, ry, ryaw = get_youbot_pose()
        try:
            bx, by = get_ball_position(ball_h)
        except Exception:
            print(f"  球句柄失效，视为已消除")
            stop_robot()
            return True

        dx   = bx - rx
        dy   = by - ry
        dist = math.hypot(dx, dy)

        # ── 消除判定 ─────────────────────────────────────────────
        if dist < ELIM_GLOBAL_DIST:
            stop_robot()
            print(f" 距离触发！全局={dist:.3f}m < {ELIM_GLOBAL_DIST}m — 消除 {ball_name}")
            try:
                sim.removeObjects([ball_h])
                print(f"  {ball_name} 已消除")
            except Exception as e:
                print(f"   removeObjects 失败: {e}")
            time.sleep(0.3)
            return True

        # ── 卡顿检测 ─────────────────────────────────────────────
        if dist < STALL_MIN_DIST:
            if dist < stall_best - STALL_MARGIN:
                stall_best  = dist
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count = 0

        if stall_count >= STALL_LIMIT:
            print(f"  step={step} 卡顿！dist={dist:.2f}m 未改善，后退恢复...")
            set_motors(-RECOVER_SPEED, -RECOVER_SPEED,
                       -RECOVER_SPEED, -RECOVER_SPEED)
            time.sleep(1.0)
            stop_robot()
            time.sleep(0.2)
            smooth_err  = None
            stall_best  = float('inf')
            stall_count = 0
            continue

        # ── 低通滤波转向 ─────────────────────────────────────────
        raw_err = angle_diff(math.atan2(dy, dx), ryaw)
        if smooth_err is None:
            smooth_err = raw_err
        else:
            smooth_err = (1 - ALPHA) * smooth_err + ALPHA * raw_err

        # ── 日志 ─────────────────────────────────────────────────
        if step % 30 == 0:
            print(f"    step={step:4d} | dist={dist:.2f}m | "
                  f"err={math.degrees(raw_err):+.1f}° | "
                  f"smooth={math.degrees(smooth_err):+.1f}°")

        # ── 驱动 ─────────────────────────────────────────────────
        drive_to(smooth_err, dist)
        time.sleep(LOOP_DT)

    stop_robot()
    print(f"  超过最大步数，放弃 {ball_name}")
    return False


def collect_all_balls():
    ball_handles = {}
    print("\n正在获取网球句柄...")
    for i in range(1, BALL_COUNT + 1):
        name = f"TennisBall_{i:02d}"
        try:
            h = sim.getObject(f"/{name}")
            ball_handles[name] = h
            bx, by = get_ball_position(h)
            print(f"  ✓ {name}: handle={h}  pos=({bx:.1f}, {by:.1f})")
        except Exception as e:
            print(f"  ✗ {name}: 获取失败 ({e})")

    if not ball_handles:
        print("未找到任何网球，退出")
        return

    print(f"\n共找到 {len(ball_handles)} 个网球，开始收集...\n")
    collected = 0
    failed    = 0

    remaining = dict(ball_handles)

    while remaining:
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
                continue

        if nearest_name is None:
            break

        success = navigate_to_ball(nearest_h, nearest_name)

        if success:
            collected += 1
        else:
            failed += 1

        del remaining[nearest_name]

        stop_robot()
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f" 收集完成！")
    print(f" 成功消除：{collected} 个")
    print(f" 放弃/失败：{failed} 个")
    print("=" * 60)


def motor_selftest():
    print("\n🔧 电机自检（每个动作 1 秒）...")
    tests = [
        ("前进",  lambda: move_forward(2.0)),
        ("左转",  lambda: turn_left(2.0)),
        ("右转",  lambda: turn_right(2.0)),
    ]
    for name, action in tests:
        x0, y0, yaw0 = get_youbot_pose()
        action()
        time.sleep(1.0)
        stop_robot()
        time.sleep(0.3)
        x1, y1, yaw1 = get_youbot_pose()
        dyaw  = math.degrees(angle_diff(yaw1, yaw0))
        ddist = math.hypot(x1 - x0, y1 - y0)
        dx    = x1 - x0
        dy    = y1 - y0
        print(f"  [{name}] 位移={ddist:.3f}m (dx={dx:+.3f}, dy={dy:+.3f})  "
              f"朝向变化={dyaw:+.1f}°  当前yaw={math.degrees(yaw1):.1f}°")
    print("🔧 自检完成\n")


if __name__ == "__main__":
    print("\n⏳等待 1 秒让仿真稳定...")
    time.sleep(1.0)

    reset_robot(enabled=True)
    motor_selftest()

    try:
        collect_all_balls()
    except KeyboardInterrupt:
        print("\n用户中断，停止机器人...")
    finally:
        stop_robot()
        print("电机已停止，程序退出")
