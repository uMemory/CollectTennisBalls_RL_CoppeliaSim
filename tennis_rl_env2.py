"""
tennis_rl_env.py
================
Gymnasium 环境：基于视觉感知的半场捡网球 RL 环境

架构设计：
  ┌─────────────────────────────────────────────┐
  │  顶层调度（半场切换 + 绕网，规则代码）          │
  ├─────────────────────────────────────────────┤
  │  中层巡视（半场扫描确认无球，规则代码）          │
  ├─────────────────────────────────────────────┤
  │  底层 RL Agent（本环境）                       │
  │  单 episode = 在当前半场内找到并消除一个球       │
  └─────────────────────────────────────────────┘

状态空间：10 维单帧 × 3 帧堆叠 = 30 维
动作空间：7 个离散动作
奖励函数：稠密引导 + 稀疏目标 + 边界/球网梯度惩罚

"""

import math
import time
import collections
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# =====================================================================
#  常量配置
# =====================================================================

# ── 场地几何（与 场景一致）──
COURT_LENGTH     = 23.77
DOUBLES_WIDTH    = 10.97
OUTER_LENGTH     = 36.57
OUTER_WIDTH      = 18.29
FENCE_LENGTH     = 38.57
FENCE_WIDTH      = 20.29
NET_WALL_Y_HALF  = 6.40
NET_HEIGHT       = 0.914

# ── 半场范围 ──
HALF_COURT_X_MIN = 0.5
HALF_COURT_X_MAX = OUTER_LENGTH / 2   # ≈18.285
HALF_COURT_Y_MAX = OUTER_WIDTH / 2    # ≈9.145

# ── 传感器参数 ──
SENSOR_RES       = 1024
SENSOR_FOV       = math.radians(75)
SENSOR_NEAR      = 0.01
SENSOR_FAR       = 18.0

# ── 图像裁剪 ──
#    裁掉上方 40%，保留下方 60%
#    网球在地面，远处球在图像垂直约 40% 处，不会被裁掉
CROP_TOP_RATIO   = 0.40
CROP_TOP_PIXELS  = int(SENSOR_RES * CROP_TOP_RATIO)  # 410
CROPPED_HEIGHT   = SENSOR_RES - CROP_TOP_PIXELS       # 614

# ── RL 参数 ──
MAX_STEPS        = 500
ACTION_REPEAT    = 4
FRAME_STACK      = 3
SINGLE_OBS_DIM   = 10
ELIM_DIST        = 0.42     # 消除阈值距离
BALL_RADIUS      = 0.1
BALL_COUNT       = 12

# ── 动作映射 ──
# v2：加了 2 个后退动作（id 7、8）
ACTION_MAP = {
    0: {'forward': 6.0, 'turn':  0.0},    # FORWARD
    1: {'forward': 4.0, 'turn':  1.5},    # FORWARD_LEFT
    2: {'forward': 4.0, 'turn': -1.5},    # FORWARD_RIGHT
    3: {'forward': 0.0, 'turn':  2.5},    # TURN_LEFT
    4: {'forward': 0.0, 'turn': -2.5},    # TURN_RIGHT
    5: {'forward': 0.0, 'turn':  5.0},    # TURN_LEFT_LARGE
    6: {'forward': 0.0, 'turn': -5.0},    # TURN_RIGHT_LARGE
    7: {'forward': -3.0, 'turn':  0.0},   # BACKWARD (v2 新增)
    8: {'forward': -3.0, 'turn':  1.5},   # BACKWARD_TURN (v2 新增,后退+小角度转)
}
NUM_ACTIONS = len(ACTION_MAP)

# ── HSV 阈值（网球荧光黄绿色）──
HSV_LOWER = np.array([25, 80, 80])
HSV_UPPER = np.array([45, 255, 255])

# ── 归一化常量 ──
NORM_X      = HALF_COURT_X_MAX
NORM_Y      = HALF_COURT_Y_MAX
NORM_YAW    = math.pi
NORM_NET    = HALF_COURT_X_MAX
NORM_BOUND  = 3.0
NORM_AREA   = 0.05
NORM_COUNT  = 6


# =====================================================================
#  Gymnasium 环境
# =====================================================================

class TennisCollectorEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, active_half=1):
        super().__init__()
        self.render_mode = render_mode
        self.active_half = active_half

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(SINGLE_OBS_DIM * FRAME_STACK,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._connect_sim()

        self.frame_buffer = collections.deque(maxlen=FRAME_STACK)
        self.step_count = 0
        self.prev_obs_single = None
        self.prev_robot_x = 0.0
        self.prev_robot_y = 0.0
        self.stuck_boundary_count = 0
        self.stuck_net_count = 0
        self.current_ball_handles = {}
        # v2: 视野新鲜度计数——连续多少步没看到活跃半场的球
        # 越大惩罚越大,鼓励 agent 主动探索避免摇头停滞
        self.no_ball_steps = 0

    # =================================================================
    #  CoppeliaSim 连接（stepping 模式）
    # =================================================================

    def _connect_sim(self):
        """
        建立 ZMQ 连接，获取句柄。

        使用方式：
          方式 A（推荐）：先在 CoppeliaSim 中点 Play 启动仿真，再运行本脚本
          方式 B：仿真处于停止状态，由代码自动启动

        启动后切换到 stepping 模式，保证 agent 同步控制。
        """
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        # 检查仿真状态
        sim_state = self.sim.getSimulationState()

        if sim_state == self.sim.simulation_stopped:
            # 仿真未启动：先设置 stepping，再启动
            self.sim.setStepping(True)
            self.sim.startSimulation()
            print("[rl_env2] 仿真已启动（stepping 模式）")
        else:
            # 仿真已在运行（用户已点 Play）：直接切换到 stepping
            self.sim.setStepping(True)
            print("[rl_env2] 仿真已在运行，已切换到 stepping 模式")

        # 等几步，让仿真稳定
        for _ in range(10):
            self.sim.step()
        time.sleep(0.3)

        self.fl = self.sim.getObject('/rollingJoint_fl')
        self.fr = self.sim.getObject('/rollingJoint_fr')
        self.rl = self.sim.getObject('/rollingJoint_rl')
        self.rr = self.sim.getObject('/rollingJoint_rr')
        self.youbot_h = self.sim.getObject('/youBot')
        self.sensor_h = self.sim.getObject('/visionSensor')

        # 记录 YouBot 初始姿态（用于后续重置时保持正确朝向）
        self._default_ori = self.sim.getObjectOrientation(
            self.youbot_h, self.sim.handle_world
        )
        self._default_pos = self.sim.getObjectPosition(
            self.youbot_h, self.sim.handle_world
        )

        # 获取 BallSpawner child script 的宿主对象句柄
        spawner_obj = self.sim.getObject('/Bin_Entry')
        self._spawner_script = self.sim.getScript(
            self.sim.scripttype_customizationscript, spawner_obj
        )
        print("[rl_env2] BallSpawner 脚本句柄已就绪")
        print(f"[rl_env2] 连接成功 | 初始姿态 ori={[f'{v:.3f}' for v in self._default_ori]}")
        print(f"[rl_env2] 初始位置 pos={[f'{v:.3f}' for v in self._default_pos]}")

    # =================================================================
    #  电机控制
    # =================================================================

    def _set_motors(self, vfl, vfr, vrl, vrr):
        self.sim.setJointTargetVelocity(self.fl, vfl)
        self.sim.setJointTargetVelocity(self.fr, vfr)
        self.sim.setJointTargetVelocity(self.rl, vrl)
        self.sim.setJointTargetVelocity(self.rr, vrr)

    def _stop(self):
        self._set_motors(0, 0, 0, 0)

    def _execute_action(self, action_id):
        params = ACTION_MAP[action_id]
        v_fwd = params['forward']
        v_turn = params['turn']
        self._set_motors(
            v_fwd + v_turn, v_fwd - v_turn,
            v_fwd + v_turn, v_fwd - v_turn
        )
        for _ in range(ACTION_REPEAT):
            self.sim.step()

    # =================================================================
    #  YouBot 状态读取
    # =================================================================

    def _get_youbot_pose(self):
        pos = self.sim.getObjectPosition(self.youbot_h, self.sim.handle_world)
        m = self.sim.getObjectMatrix(self.youbot_h, self.sim.handle_world)
        yaw = math.atan2(-m[6], -m[2])
        return pos[0], pos[1], yaw

    # =================================================================
    #  视觉感知
    # =================================================================

    def _get_rgb_image(self):
        """
        获取 RGB 图像并裁剪上方 40%。
        返回 BGR numpy 数组, shape=(614, 1024, 3)
        """
        img_buf, res = self.sim.getVisionSensorImg(self.sensor_h, 0)
        w, h = res[0], res[1]
        if isinstance(img_buf, (bytes, bytearray)):
            img_np = np.frombuffer(img_buf, dtype=np.uint8).reshape(h, w, 3)
        else:
            img_np = np.array(img_buf, dtype=np.uint8).reshape(h, w, 3)
        img_np = np.flipud(img_np)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr[CROP_TOP_PIXELS:, :]

    def _detect_balls_in_image(self, img_bgr):
        """
        HSV 阈值分割检测网球。
        输入是裁剪后图像 (614×1024)，只用水平偏角（cx），不受裁剪影响。
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        balls = []
        min_area = 30

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            angle_norm = (cx - SENSOR_RES / 2) / (SENSOR_RES / 2)
            balls.append({'cx': cx, 'cy': cy, 'area': area, 'angle_norm': angle_norm})

        balls.sort(key=lambda b: b['area'], reverse=True)
        return balls

    def _estimate_distance_from_area(self, area_pixels):
        if area_pixels <= 1:
            return SENSOR_FAR
        pixel_radius = math.sqrt(area_pixels / math.pi)
        est_dist = (BALL_RADIUS * SENSOR_RES) / (2 * pixel_radius * math.tan(SENSOR_FOV / 2))
        return min(max(est_dist, SENSOR_NEAR), SENSOR_FAR)

    # =================================================================
    #  球可达性判断
    # =================================================================

    def _estimate_ball_world_pos(self, ball_angle_norm, ball_area, robot_x, robot_y, robot_yaw):
        """
        根据像素偏角和面积估计球在世界系下的 (bx, by)。

        角度公式使用精确反三角（atan），替代原来的小角度近似
        ball_angle_rad ≈ angle_norm * FOV/2（边缘误差 ~20%）。

        返回 (est_bx, est_by, est_dist)
        """
        tan_half_fov = math.tan(SENSOR_FOV / 2)
        ball_angle_rad = math.atan(ball_angle_norm * tan_half_fov)
        ball_world_angle = robot_yaw + ball_angle_rad
        est_dist = self._estimate_distance_from_area(ball_area)
        est_bx = robot_x + est_dist * math.cos(ball_world_angle)
        est_by = robot_y + est_dist * math.sin(ball_world_angle)
        return est_bx, est_by, est_dist

    def _ball_in_active_half(self, est_bx, strict=False):
        """
        判断估计球位置是否在活跃半场。

        strict=True  : 严格半场内（est_bx > 0 for X>0 half）
                       用于观察过滤，彻底排除对面球的干扰。
        strict=False : 容忍 0.5m 的距离估计误差（est_bx > -0.5）
                       用于 ball_reachable 可达标志，避免误杀网边的球。
        """
        margin = 0.0 if strict else 0.5
        if self.active_half > 0:
            return est_bx > -margin
        else:
            return est_bx < margin

    # =================================================================
    #  几何工具
    # =================================================================

    def _dist_to_nearest_boundary(self, rx, ry):
        dx_pos = FENCE_LENGTH / 2 - rx
        dx_neg = rx + FENCE_LENGTH / 2
        dy_pos = FENCE_WIDTH / 2 - ry
        dy_neg = ry + FENCE_WIDTH / 2
        return max(0.0, min(dx_pos, dx_neg, dy_pos, dy_neg))

    def _net_distance(self, rx):
        return abs(rx)

    def _crossed_net(self, rx):
        if self.active_half > 0:
            return rx < -0.3
        else:
            return rx > 0.3

    # =================================================================
    #  观测构建
    # =================================================================

    def _build_single_obs(self):
        rx, ry, ryaw = self._get_youbot_pose()

        img_bgr = self._get_rgb_image()
        balls_all = self._detect_balls_in_image(img_bgr)

        # ── 按活跃半场严格过滤（排除对面半场的球，避免观察污染）──
        # 同时记录每个球的估计世界坐标，后续复用
        balls_filtered = []
        for b in balls_all:
            est_bx, est_by, est_dist = self._estimate_ball_world_pos(
                b['angle_norm'], b['area'], rx, ry, ryaw
            )
            b['est_bx'] = est_bx
            b['est_by'] = est_by
            b['est_dist'] = est_dist
            if self._ball_in_active_half(est_bx, strict=True):
                balls_filtered.append(b)

        # 用于 debug 渲染：标记哪些被过滤掉了
        self._last_balls_all = balls_all
        self._last_balls_filtered = balls_filtered

        if self.render_mode == "human":
            self._render_debug(img_bgr, balls_all)

        total_pixels = SENSOR_RES * CROPPED_HEIGHT

        # ── 语义区分（用于奖励）──
        # any_ball_visible : 视野里看到任何球（包括对面半场）
        # ball_in_half     : 过滤后有活跃半场内的球
        any_ball_visible = len(balls_all) > 0
        ball_in_half = len(balls_filtered) > 0

        if ball_in_half:
            # 在活跃半场内的球中取面积最大（最近）的一个
            nearest = balls_filtered[0]
            ball_detected = 1.0
            ball_angle = float(np.clip(nearest['angle_norm'], -1.0, 1.0))
            ball_size = float(np.clip(nearest['area'] / (total_pixels * NORM_AREA), 0.0, 1.0))
            ball_count = float(np.clip(len(balls_filtered) / NORM_COUNT, 0.0, 1.0))
            # 用宽松判据给 reachable 标志（容忍估计误差）
            ball_reachable = 1.0 if self._ball_in_active_half(nearest['est_bx'], strict=False) else 0.0
        else:
            ball_detected = 0.0
            ball_angle = 0.0
            ball_size = 0.0
            ball_count = 0.0
            ball_reachable = 0.0

        norm_rx = np.clip(rx / NORM_X, -1.0, 1.0)
        norm_ry = np.clip(ry / NORM_Y, -1.0, 1.0)
        norm_yaw = np.clip(ryaw / NORM_YAW, -1.0, 1.0)
        norm_net = np.clip(self._net_distance(rx) / NORM_NET, 0.0, 1.0)
        norm_bound = np.clip(
            self._dist_to_nearest_boundary(rx, ry) / NORM_BOUND, 0.0, 1.0
        )

        obs = np.array([
            ball_detected, ball_angle, ball_size, ball_count, ball_reachable,
            norm_rx, norm_ry, norm_yaw, norm_net, norm_bound,
        ], dtype=np.float32)

        self._last_raw = {
            'robot_x': rx, 'robot_y': ry, 'robot_yaw': ryaw,
            'ball_detected': ball_detected > 0.5,
            'ball_angle': ball_angle,
            'ball_size': ball_size,
            'ball_is_reachable': ball_reachable > 0.5,
            'any_ball_visible': any_ball_visible,       # 含对面半场
            'ball_in_half': ball_in_half,               # 仅活跃半场
            'net_distance': self._net_distance(rx),
            'dist_to_boundary': self._dist_to_nearest_boundary(rx, ry),
        }
        return obs

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frame_buffer), axis=0)

    # =================================================================
    #  奖励计算
    # =================================================================

    def _compute_reward(self, ball_eliminated):
        curr = self._last_raw

        if ball_eliminated:
            return 100.0

        reward = -0.1

        displacement = math.hypot(
            curr['robot_x'] - self.prev_robot_x,
            curr['robot_y'] - self.prev_robot_y
        )
        is_moving = displacement > 0.02

        if not is_moving:
            reward -= 1.0

        # ── 视觉引导奖励 ──
        # 三种情况严格区分：
        #   (a) 看到活跃半场内的球（ball_in_half=True）→ 给正向引导，重置 no_ball_steps
        #   (b) 视野里只有对面半场的球（any_ball_visible && !ball_in_half）→ 强惩罚
        #   (c) 完全看不到任何球 → 线性递增的惩罚（v2 视野新鲜度）
        #       看不到越久,单步惩罚越大,鼓励主动探索而非摇头
        ball_in_half = curr['ball_in_half']
        any_visible = curr['any_ball_visible']
        prev_in_half = (self.prev_obs_single is not None and
                        self.prev_obs_single.get('ball_in_half', False))

        if ball_in_half:
            # 看到活跃半场球 → 重置视野新鲜度计数
            self.no_ball_steps = 0
            if is_moving:
                reward += 0.5
                reward += 1.0 * (1.0 - abs(curr['ball_angle']))
                if prev_in_half:
                    delta_size = curr['ball_size'] - self.prev_obs_single['ball_size']
                    reward += 5.0 * delta_size
        elif any_visible:
            # 视野里只有对面半场的球 —— 比"完全没看到"更糟
            # 也累加 no_ball_steps（从活跃半场角度也算没看到球）
            self.no_ball_steps += 1
            reward -= 0.5
        else:
            # 完全看不到球 → 惩罚线性增加
            # 0 步: -0.3, 30 步: -0.6, 60 步: -0.9, 封顶 -1.5
            self.no_ball_steps += 1
            freshness_penalty = min(0.3 + 0.01 * self.no_ball_steps, 1.5)
            reward -= freshness_penalty

        boundary_margin = 1.0
        dist_b = curr['dist_to_boundary']
        if dist_b < boundary_margin:
            reward -= 3.0 * (boundary_margin - dist_b)

        net_margin = 0.8
        dist_n = curr['net_distance']
        if dist_n < net_margin:
            reward -= 4.0 * (net_margin - dist_n)

        if len(self.frame_buffer) >= FRAME_STACK:
            oldest = self.frame_buffer[0]
            newest = self.frame_buffer[-1]
            yaw_change = abs(newest[7] - oldest[7])
            pos_change = math.hypot(newest[5] - oldest[5], newest[6] - oldest[6])
            if yaw_change > 0.3 and pos_change < 0.02:
                reward -= 1.0

        return reward

    # =================================================================
    #  网球句柄管理
    # =================================================================

    def _refresh_ball_handles(self):
        self.current_ball_handles = {}
        for i in range(1, BALL_COUNT + 1):
            name = f"TennisBall_{i:02d}"
            try:
                h = self.sim.getObject(f"/{name}")
                self.current_ball_handles[name] = h
            except Exception:
                continue

    def _check_elimination(self, rx, ry):
        for name, h in list(self.current_ball_handles.items()):
            try:
                bpos = self.sim.getObjectPosition(h, self.sim.handle_world)
                dist = math.hypot(bpos[0] - rx, bpos[1] - ry)
                if dist < ELIM_DIST:
                    self.sim.removeObjects([h])
                    del self.current_ball_handles[name]
                    # 让仿真器把删除事件消化掉,再返回
                    self.sim.step()
                    print(f"[rl_env2] 消除 {name} (dist={dist:.3f}m)")
                    return True
            except Exception:
                if name in self.current_ball_handles:
                    del self.current_ball_handles[name]
                continue
        return False

    def _count_balls_in_active_half(self):
        count = 0
        for name, h in list(self.current_ball_handles.items()):
            try:
                bpos = self.sim.getObjectPosition(h, self.sim.handle_world)
                if self.active_half > 0 and bpos[0] > 0:
                    count += 1
                elif self.active_half < 0 and bpos[0] < 0:
                    count += 1
            except Exception:
                continue
        return count

    # =================================================================
    #  YouBot 重置
    # =================================================================

    def _reset_youbot(self):
        """
        将 YouBot 随机放置在当前活跃半场内。

        朝向处理：
          不直接构造欧拉角（容易导致翻车），而是：
          1. 先恢复到初始正常姿态（从连接时记录的 _default_ori）
          2. 用四元数绕世界 Z 轴旋转随机角度
          这样保证 YouBot 始终水平放置，只改变航向角。
        """
        self._stop()
        for _ in range(3):
            self.sim.step()

        # 随机位置
        if self.active_half > 0:
            rx = np.random.uniform(2.0, HALF_COURT_X_MAX - 1.0)
        else:
            rx = np.random.uniform(-HALF_COURT_X_MAX + 1.0, -2.0)
        ry = np.random.uniform(-HALF_COURT_Y_MAX + 1.0, HALF_COURT_Y_MAX - 1.0)
        rz = self._default_pos[2]  # 使用初始高度
        # 设置位置
        self.sim.setObjectPosition(
            self.youbot_h, [rx, ry, rz], self.sim.handle_world
        )

        # 先恢复到初始正常姿态（保证水平）
        self.sim.setObjectOrientation(
            self.youbot_h, list(self._default_ori), self.sim.handle_world
        )

        # 再用四元数绕 Z 轴旋转随机角度（只改航向，不倾斜）
        random_yaw = np.random.uniform(-math.pi, math.pi)
        # 获取当前四元数
        quat = self.sim.getObjectQuaternion(self.youbot_h, self.sim.handle_world)
        # 构造绕 Z 轴旋转的四元数: [0, 0, sin(θ/2), cos(θ/2)]
        half = random_yaw / 2.0
        qz = [0.0, 0.0, math.sin(half), math.cos(half)]
        # 四元数乘法: q_new = qz * quat (先原始姿态，再绕 Z 旋转)
        new_quat = self._quat_multiply(qz, quat)
        self.sim.setObjectQuaternion(self.youbot_h, new_quat, self.sim.handle_world)

        # 重置动力学状态（清除残余速度/角速度）
        try:
            self.sim.resetDynamicObject(self.youbot_h)
        except Exception:
            pass

        # 等待物理引擎稳定
        for _ in range(15):
            self.sim.step()
        time.sleep(0.1)

    @staticmethod
    def _quat_multiply(q1, q2):
        """四元数乘法 q1 * q2，格式 [x, y, z, w]"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ]

    # =================================================================
    #  调试渲染
    # =================================================================

    def _render_debug(self, img_bgr, balls):
        debug_img = img_bgr.copy()
        h, w = debug_img.shape[:2]
        # 字号 + 加粗 + 黑色描边，配合 WINDOW_KEEPRATIO 实现窗口缩放时
        # 文字按比例自适应。注意：字号需控制在图像 1024px 宽度内不出界。
        FONT_AREA = 0.8
        FONT_INFO = 0.9
        TH_AREA   = 2
        TH_INFO   = 2
        for b in balls:
            cx, cy = int(b['cx']), int(b['cy'])
            r = max(5, int(math.sqrt(b['area'] / math.pi)))
            cv2.circle(debug_img, (cx, cy), r, (0, 255, 0), 2)
            label = f"a={b['area']:.0f}"
            # 测量文字尺寸，确保不画到图像外
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          FONT_AREA, TH_AREA)
            tx = min(cx + r + 5, w - tw - 5)
            ty = max(th + 5, min(cy, h - 5))
            cv2.putText(debug_img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_AREA, (0, 255, 0), TH_AREA)

        raw = self._last_raw if hasattr(self, '_last_raw') else {}
        # 拆成两行（原本单行约 50 字符 × 20px ≈ 1000px 接近图像边界），
        # 拆开后每行 ~25 字符 ≈ 500px，留足边距
        info_line1 = (
            f"step={self.step_count}  "
            f"det={raw.get('ball_detected', '?')}  "
            f"reach={raw.get('ball_is_reachable', '?')}"
        )
        info_line2 = (
            f"net={raw.get('net_distance', 0):.1f}m  "
            f"bound={raw.get('dist_to_boundary', 0):.1f}m"
        )
        for i, line in enumerate((info_line1, info_line2)):
            y_pos = 35 + i * 32
            cv2.putText(debug_img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_INFO, (255, 255, 255), TH_INFO)
        # 首次显示时声明 WINDOW_NORMAL → 窗口可自由拖拽缩放
        # WINDOW_KEEPRATIO 保持图像比例不被拉伸；文字随图像同比例缩放（自适应）
        if not getattr(self, '_debug_window_inited', False):
            cv2.namedWindow("TennisRL Debug",
                            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("TennisRL Debug", 800, 480)
            self._debug_window_inited = True
        cv2.imshow("TennisRL Debug", debug_img)
        cv2.waitKey(1)

    # =================================================================
    #  Gymnasium: reset
    # =================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._refresh_ball_handles()

        # ── 球数检查：分两层处理 ──
        if self._count_balls_in_active_half() == 0:
            total = self._count_total_balls()
            if total == 0:
                # 全场无球 → 重新生成
                print("[rl_env2] 全场无球，重新生成网球...")
                self._respawn_balls(ball_count=BALL_COUNT)
            else:
                # 当前半场无球但对面有球 → 切换半场
                self.active_half = -self.active_half
                print(f"[rl_env2] 当前半场无球，切换到 {'X>0' if self.active_half > 0 else 'X<0'} 半场继续训练")

        self._reset_youbot()

        self.step_count = 0
        self.prev_obs_single = None
        self.prev_robot_x, self.prev_robot_y, _ = self._get_youbot_pose()
        self.stuck_boundary_count = 0
        self.stuck_net_count = 0
        self.no_ball_steps = 0    # v2: 重置视野新鲜度

        first_obs = self._build_single_obs()
        self.prev_obs_single = dict(self._last_raw)
        for _ in range(FRAME_STACK):
            self.frame_buffer.append(first_obs.copy())

        stacked = self._get_stacked_obs()
        info = {
            'balls_in_half': self._count_balls_in_active_half(),
            'active_half': self.active_half,
        }
        return stacked, info

    # =================================================================
    #  Gymnasium: soft_reset  （部署专用）
    # =================================================================

    def soft_reset(self):
        """
        部署模式专用：只重置 episode 状态，不挪动 YouBot 位置，
        也不自动切换半场 / 重生成球。

        典型用法（部署流程）：
          第一次进入循环前调用一次 env.reset() 初始化 buffer；
          之后每轮 RL 捡球结束后调用 env.soft_reset()，保留车的位置
          供下一轮继续捡（或由外层规则代码接管导航）。

        不做的事（与 reset 的区别）：
          - 不调用 _reset_youbot()，YouBot 留在原位
          - 不自动切 active_half（这个由外层 deploy 的绕网流程管）
          - 不 respawn 球
        """
        self._stop()
        self._refresh_ball_handles()

        self.step_count = 0
        self.prev_obs_single = None
        rx, ry, _ = self._get_youbot_pose()
        self.prev_robot_x = rx
        self.prev_robot_y = ry
        self.stuck_boundary_count = 0
        self.stuck_net_count = 0
        self.no_ball_steps = 0    # v2: 重置视野新鲜度

        # 清空 frame buffer，用当前观察重新填满 FRAME_STACK 帧
        first_obs = self._build_single_obs()
        self.prev_obs_single = dict(self._last_raw)
        self.frame_buffer.clear()
        for _ in range(FRAME_STACK):
            self.frame_buffer.append(first_obs.copy())

        stacked = self._get_stacked_obs()
        info = {
            'balls_in_half': self._count_balls_in_active_half(),
            'active_half': self.active_half,
        }
        return stacked, info

    # =================================================================
    #  Gymnasium: step
    # =================================================================

    def step(self, action):
        self.step_count += 1
        self._execute_action(action)

        rx, ry, _ = self._get_youbot_pose()
        ball_eliminated = self._check_elimination(rx, ry)

        obs_single = self._build_single_obs()
        self.frame_buffer.append(obs_single)
        stacked = self._get_stacked_obs()

        reward = self._compute_reward(ball_eliminated)

        terminated = False
        truncated = False
        info = {}

        if ball_eliminated:
            terminated = True
            info['success'] = True
            info['reason'] = 'ball_eliminated'
        elif self._crossed_net(rx):
            reward -= 10.0
            terminated = True
            info['success'] = False
            info['reason'] = 'crossed_net'
        elif self.step_count >= MAX_STEPS:
            truncated = True
            info['success'] = False
            info['reason'] = 'timeout'

        # ── Stuck 判据：加"位移小"与条件，避免追球途经边界/网被误判 ──
        # 当前步位移（相对上一步）
        step_displacement = math.hypot(rx - self.prev_robot_x, ry - self.prev_robot_y)
        is_actually_stuck = step_displacement < 0.05

        if self._last_raw['dist_to_boundary'] < 0.3 and is_actually_stuck:
            self.stuck_boundary_count += 1
        else:
            self.stuck_boundary_count = 0
        if not terminated and not truncated and self.stuck_boundary_count > 30:
            reward -= 10.0
            terminated = True
            info['success'] = False
            info['reason'] = 'stuck_at_boundary'

        if self._last_raw['net_distance'] < 0.3 and is_actually_stuck:
            self.stuck_net_count += 1
        else:
            self.stuck_net_count = 0
        if not terminated and not truncated and self.stuck_net_count > 30:
            reward -= 10.0
            terminated = True
            info['success'] = False
            info['reason'] = 'stuck_at_net'

        self.prev_obs_single = dict(self._last_raw)
        self.prev_robot_x = rx
        self.prev_robot_y = ry

        if terminated or truncated:
            self._stop()

        info['step'] = self.step_count
        info['reward'] = reward
        info['balls_remaining'] = self._count_balls_in_active_half()

        return stacked, reward, terminated, truncated, info

    # =================================================================
    #  刷新网球，继续训练
    # =================================================================
    def _respawn_balls(self, ball_count=12, seed=0):
        """
        通过 callScriptFunction 调用挂载在 Bin_Base 上的spawnBalls() Lua 函数，
        重新生成网球。seed=0 表示基于时间随机（由 Lua 端处理）。
        """
        print(f"[rl_env2] 全场无球，调用 Lua spawnBalls({ball_count}, seed={seed})...")
        try:
            ret = self.sim.callScriptFunction(
                'spawnBalls',  # 函数名
                self._spawner_script,  # 宿主对象句柄
                [ball_count, seed],  # inInts
                [],  # inFloats
                [],  # inStrings
                ''  # inBuffer
            )
            # ret = (outInts, outFloats, outStrings, outBuffer)
            actual_count = ret[0][0] if ret and ret[0] else ball_count
            print(f"[rl_env2] 网球重生成完毕，共 {actual_count} 个")
        except Exception as e:
            print(f"[rl_env2] spawnBalls 调用失败: {e}")

        # 等待物理引擎稳定后刷新句柄
        for _ in range(5):
            self.sim.step()
        self._refresh_ball_handles()

    # =================================================================
    #  统计网球
    # =================================================================
    def _count_total_balls(self):
        """统计全场（两个半场）剩余球总数"""
        count = 0
        for name, h in list(self.current_ball_handles.items()):
            try:
                self.sim.getObjectPosition(h, self.sim.handle_world)
                count += 1
            except Exception:
                if name in self.current_ball_handles:
                    del self.current_ball_handles[name]
        return count

    # =================================================================
    #  Gymnasium: close
    # =================================================================

    def close(self):
        self._stop()
        if self.render_mode == "human":
            cv2.destroyAllWindows()


# =====================================================================
#  快速测试
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("[rl_env2] TennisCollectorEnv 环境测试（随机动作）\n")

    print("使用步骤：")
    print("1. 打开 CoppeliaSim，加载网球场景")
    print("2. 执行 tennis_scene_latest.lua 生成场景")
    print("3. 执行 Tennis_Generate.lua 生成网球")
    print("4. 点 Play ▶ 启动仿真")
    print("5. 运行本脚本: python tennis_rl_env.py\n")

    print("训练步骤：")
    print("1. 完成上述 1-4 步")
    print("2. 运行: python train_ppo.py\n")

    print("部署步骤：")
    print("1. 完成上述 1-4 步")
    print("2. 运行: python deploy_collector.py --model ./models/best/best_model.zip\n")
    input("按 Enter 开始测试（确保仿真已启动）...")

    env = TennisCollectorEnv(render_mode="human", active_half=1)

    for episode in range(3):
        obs, info = env.reset()
        print(f"\n[rl_env2] Episode {episode + 1} | obs shape={obs.shape} | info={info}")

        total_reward = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            if env.step_count % 50 == 0:
                print(f"[rl_env2] step={env.step_count:4d} | reward={reward:+.2f} | "
                      f"total={total_reward:+.1f} | info={info.get('reason', 'running')}")

        print(f"[rl_env2]✅ Episode 结束 | 总奖励={total_reward:+.1f} | "
              f"reason={info.get('reason', '?')} | steps={env.step_count}")

    env.close()
    print("\n[rl_env2] 环境测试完成")
