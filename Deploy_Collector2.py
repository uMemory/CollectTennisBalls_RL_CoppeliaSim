"""
deploy_collector.py
===================
部署脚本：RL Agent + 半场巡视 + 绕网切换 完整捡球流程

架构：
  顶层循环:
    while 场上有球:
      1. RL Agent 在当前半场捡球（循环调用 env.step）
      2. 连续看不到球 → 触发半场巡视
      3. 巡视确认当前半场清空 → 绕网切换到另一半场
      4. 重复

巡视路线：
  - 7 个路径点覆盖半场的近网区、中场、底线
  - 从距 YouBot 最近的点切入，不浪费路程
  - 正向走完后原路返回，正反两方向视野覆盖所有死角

用法：
  python deploy_collector.py --model ./models/best/best_model.zip
"""

import argparse
import math
import time
import datetime
import collections
import numpy as np
from stable_baselines3 import PPO
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tennis_rl_env2 import (
    TennisCollectorEnv,
    BALL_COUNT,
    HALF_COURT_X_MAX,
    HALF_COURT_Y_MAX,
    OUTER_WIDTH,
    NET_WALL_Y_HALF,
    FENCE_LENGTH,
    FENCE_WIDTH,
)


# =====================================================================
#  导航参数
# =====================================================================

# ── 绕网参数 ──
# 场地几何约束（见 tennis_scene_latest.lua）:
#   网柱位于 Y = ±6.40m，必须绕到 |Y| > 6.40 才能横穿 X=0
#   椅子 1 位于 y ≈ -8.35 (占用 Y ∈ [-8.55, -7.72])
#   椅子 2 位于 y ≈ +8.35 (占用 Y ∈ [+7.72, +8.55])
#   安全走廊: |Y| ∈ (6.40, 7.72), 取中值 7.05 给两侧各 ~0.6m 余量
NET_BYPASS_Y       = 7.05
COURT_X_HALF       = 11.885
WAYPOINT_REACHED   = 0.50
NAV_MAX_ITER       = 2000
NAV_LOOP_DT        = 0.05
NAV_ALPHA          = 0.15
BASE_SPEED         = 10.0
TURN_SPEED         = 5.0

# ── RL 捡球阶段的"原地摇头"检测参数 ──
# 当视野里没有当前半场的球（例如球藏在车背后）时，策略倾向原地摇头，
# 因为转头的累积惩罚（-0.4/步）比"往外冲撞边界"更小。这是一个局部最优。
# 这里做一个部署层的兜底：若最近 STUCK_WINDOW 步内 YouBot 位置范围
# 小于 STUCK_THRESHOLD，判定为卡住，终止 RL 循环，主动触发一次巡视
# 来打破僵局。
STUCK_WINDOW        = 80     # 滑动窗口长度（RL step 数），80 × 4 仿真步 ≈ 16 秒仿真
STUCK_THRESHOLD     = 0.20   # 窗口内 X 或 Y 的最大位移范围（米），小于此值判为卡住
POST_PATROL_GRACE   = 40     # 巡视中断发现球后，RL 启动的前 N 步不做卡住检测，
                             # 避免"巡视发现球 → RL 重启位置不理想 → 又触发卡住 →
                             # 又触发巡视"的死循环导致 ZMQ 过载崩溃


# =====================================================================
#  规则导航器
# =====================================================================

class RuleNavigator:
    """规则导航器：绕网 + 巡视"""

    def __init__(self, sim, youbot_h, wheels, env=None):
        self.sim = sim
        self.youbot_h = youbot_h
        self.fl, self.fr, self.rl, self.rr = wheels
        # env 可选：若传入且 render_mode=='human'，在导航过程中
        # 周期性刷新 OpenCV 调试窗口，避免窗口长时间无刷新变"无响应"
        self.env = env

    def _set_motors(self, vfl, vfr, vrl, vrr):
        self.sim.setJointTargetVelocity(self.fl, vfl)
        self.sim.setJointTargetVelocity(self.fr, vfr)
        self.sim.setJointTargetVelocity(self.rl, vrl)
        self.sim.setJointTargetVelocity(self.rr, vrr)

    def _stop(self):
        self._set_motors(0, 0, 0, 0)

    def _get_pose(self):
        pos = self.sim.getObjectPosition(self.youbot_h, self.sim.handle_world)
        m = self.sim.getObjectMatrix(self.youbot_h, self.sim.handle_world)
        yaw = math.atan2(-m[6], -m[2])
        return pos[0], pos[1], yaw

    def _refresh_debug_window(self):
        """
        导航阶段定期刷新 OpenCV 调试窗口。
        只取图 + 检测 + 渲染，不修改 env 的 frame_buffer / prev_obs_single，
        避免影响下一轮 RL 捡球的状态连续性。
        """
        if self.env is None or self.env.render_mode != "human":
            return
        try:
            img_bgr = self.env._get_rgb_image()
            balls = self.env._detect_balls_in_image(img_bgr)
            self.env._render_debug(img_bgr, balls)
        except Exception:
            # 刷新窗口失败不阻断导航
            pass

    @staticmethod
    def _angle_diff(a, b):
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def _escape_from_obstacle(self, safe_margin=1.5, max_steps=200, label="Escape"):
        """
        若 YouBot 当前贴近边界 / 球网，先后退 + 转向场地中心脱困，再返回。

        触发场景：上一轮 RL 因 stuck_at_boundary / stuck_at_net / crossed_net
        终止时 YouBot 物理上仍贴墙；若直接进入 navigate_to(...) 规则导航，
        电机会一直顶着墙转，navigate_to 必然超时形成假死循环。

        策略：
          - 计算到 4 边界 + 球网的最近距离 min_dist
          - 若 min_dist >= safe_margin 直接返回 True
          - 否则朝场地中心 (0, 0) 方向：
              * 车头朝向相对正确（偏差 < 60°）→ 前进 + 修正
              * 车头朝向障碍 → 后退 + 边退边转
          - 上限 max_steps 防止永远卡住，超时返回 False（外层仍可继续，
            只是后续导航大概率失败）

        返回 True/False 仅供日志参考，不影响后续流程。
        """
        FENCE_LX_HALF = FENCE_LENGTH / 2     # 19.285
        FENCE_LY_HALF = FENCE_WIDTH / 2      # 10.145

        for step in range(max_steps):
            rx, ry, ryaw = self._get_pose()

            dist_xpos = FENCE_LX_HALF - rx
            dist_xneg = rx + FENCE_LX_HALF
            dist_ypos = FENCE_LY_HALF - ry
            dist_yneg = ry + FENCE_LY_HALF
            dist_net  = abs(rx)
            min_dist = min(dist_xpos, dist_xneg, dist_ypos, dist_yneg, dist_net)

            if min_dist >= safe_margin:
                self._stop()
                for _ in range(3):
                    self.sim.step()
                if step > 0:
                    print(f"  ✅ {label} 脱困完成 ({step} 步, min_dist={min_dist:.2f}m)")
                return True

            # 目标朝向：场地中心
            target_yaw = math.atan2(-ry, -rx)
            angle_err = self._angle_diff(target_yaw, ryaw)

            if abs(angle_err) < math.radians(60):
                # 车头大致朝向中心 → 前进 + 转向修正
                v_turn = max(-TURN_SPEED, min(TURN_SPEED, 2.0 * angle_err))
                v_fwd = 2.0
            else:
                # 车头朝向障碍 → 后退（速度小）+ 边退边转
                v_turn = max(-TURN_SPEED * 0.5, min(TURN_SPEED * 0.5, 1.0 * angle_err))
                v_fwd = -1.5

            self._set_motors(v_fwd + v_turn, v_fwd - v_turn,
                             v_fwd + v_turn, v_fwd - v_turn)
            self.sim.step()
            if step % 5 == 0:
                self._refresh_debug_window()

        self._stop()
        for _ in range(3):
            self.sim.step()
        print(f"  ⚠️ {label} 脱困超时 (max={max_steps} 步, min_dist={min_dist:.2f}m)")
        return False

    def _drive_to(self, smooth_angle, dist):
        # 控制增益：
        #   TURN_KP 越大转向越激进；过大会震荡。3.0 是在稳定/响应之间的折中
        #   FWD_KP 越大接近目标时减速越早
        FWD_KP, TURN_KP = 1.5, 2.0
        FWD_MAX, TURN_MAX = BASE_SPEED, TURN_SPEED
        v_turn = max(-TURN_MAX, min(TURN_MAX, TURN_KP * smooth_angle))
        if abs(smooth_angle) > math.radians(90):
            # 角度偏差 > 90° → 纯转向（原地转,不前进）
            self._set_motors(v_turn, -v_turn, v_turn, -v_turn)
        else:
            v_fwd = max(0.3, min(FWD_MAX, FWD_KP * dist * math.cos(smooth_angle)))
            self._set_motors(v_fwd + v_turn, v_fwd - v_turn,
                             v_fwd + v_turn, v_fwd - v_turn)

    def navigate_to(self, tx, ty, reach_dist=WAYPOINT_REACHED, label="WP",
                    abort_check=None, abort_check_every=15):
        """导航到目标点，返回 True=到达、False=超时、"aborted"=被 abort_check 中断。

        注意: 仿真处于 stepping 模式（env 初始化时设置），
        必须在循环里显式调用 sim.step() 推进仿真，
        单靠 time.sleep 仿真不会动，server 会因超时 abort。

        可选参数:
          abort_check: 无参数回调函数，每 abort_check_every 个 sim.step() 调用一次。
                       返回 True 时立即停车并返回 "aborted"。用于巡视时实时检测球。
          abort_check_every: 多少次 sim.step() 检查一次（默认 15，避免 ZMQ 调用过密
                             导致 CoppeliaSim server 过载崩溃）
        """
        # 每 N 次 sim.step() 刷新一次调试窗口，取图有开销不能每帧都刷
        DEBUG_REFRESH_EVERY = 5

        # ── 途中卡住检测参数 ──
        # 若 STUCK_CHECK_EVERY 步内位移 < STUCK_DISPLACEMENT，判为撞墙/撞网
        # 触发一次脱困（_escape_from_obstacle）后继续向目标推进
        STUCK_CHECK_EVERY = 60       # 60 sim.step ≈ 3 秒仿真
        STUCK_DISPLACEMENT = 0.20    # 60 步内位移 < 20cm 即判卡住
        MAX_INFLIGHT_ESCAPES = 3     # 单次 navigate_to 内最多脱困 3 次

        smooth_err = None
        last_check_pos = None
        inflight_escapes = 0

        for step in range(NAV_MAX_ITER):
            rx, ry, ryaw = self._get_pose()
            dx, dy = tx - rx, ty - ry
            dist = math.hypot(dx, dy)
            if dist < reach_dist:
                self._stop()
                # 送一小段稳定帧，让电机真正停下
                for _ in range(3):
                    self.sim.step()
                self._refresh_debug_window()
                return True
            raw_err = self._angle_diff(math.atan2(dy, dx), ryaw)
            if smooth_err is None:
                smooth_err = raw_err
            else:
                smooth_err = (1 - NAV_ALPHA) * smooth_err + NAV_ALPHA * raw_err
            self._drive_to(smooth_err, dist)
            self.sim.step()   # ← 显式推进仿真（替代 time.sleep）
            if step % DEBUG_REFRESH_EVERY == 0:
                self._refresh_debug_window()
            # 可选的中断检查（例如巡视时检测到球）
            if abort_check is not None and step % abort_check_every == 0:
                if abort_check():
                    self._stop()
                    for _ in range(3):
                        self.sim.step()
                    return "aborted"

            # ── 途中卡住检测：每 STUCK_CHECK_EVERY 步检查累计位移 ──
            if step > 0 and step % STUCK_CHECK_EVERY == 0:
                if last_check_pos is not None:
                    inflight_disp = math.hypot(rx - last_check_pos[0],
                                               ry - last_check_pos[1])
                    if inflight_disp < STUCK_DISPLACEMENT:
                        if inflight_escapes >= MAX_INFLIGHT_ESCAPES:
                            self._stop()
                            for _ in range(3):
                                self.sim.step()
                            print(f"  ⚠️ {label} 途中反复卡住 ({MAX_INFLIGHT_ESCAPES} 次脱困仍卡)，放弃")
                            return False
                        print(f"  🚧 {label} 途中卡住 "
                              f"({STUCK_CHECK_EVERY} 步位移仅 {inflight_disp:.2f}m)，触发脱困")
                        self._escape_from_obstacle(safe_margin=1.5,
                                                   label=f"{label}-Mid#{inflight_escapes+1}")
                        inflight_escapes += 1
                        # 脱困后位置已变，重置平滑误差与位置基准
                        smooth_err = None
                        last_check_pos = self._get_pose()[:2]
                        continue
                last_check_pos = (rx, ry)
        self._stop()
        for _ in range(3):
            self.sim.step()
        self._refresh_debug_window()
        print(f"  ⚠️ {label} 导航超时")
        return False

    # -----------------------------------------------------------------
    #  绕网
    # -----------------------------------------------------------------

    def bypass_net(self, target_half):
        """绕网到目标半场（+1 或 -1）。

        两段式路径（以 X>0 → X<0 为例）:
          WP1: 先纵向退到 Y=±7.05m 的安全走廊（避开椅子 Y∈[7.72,8.55]，
               也在网柱 |Y|=6.40 之外可以横穿 X=0）
          WP2: 横向穿越 X=0，到达目标半场内 X=±3
          完成后当前位置在 (target_x, ±7.05)，已经在对面半场，
          RL agent 接管后会边走边找球，比规则代码直线拉回 Y=0 更有效
        """
        # 绕网前先脱困：同 patrol_half，避免贴墙状态下规则导航撞墙超时
        self._escape_from_obstacle(safe_margin=1.5, label="Pre-Bypass")

        rx, ry, _ = self._get_pose()

        bypass_y = NET_BYPASS_Y if ry >= 0 else -NET_BYPASS_Y
        sign_r = 1 if rx >= 0 else -1
        # 起步先离网 ≥ 1.5m，避免贴网时 WP2 横穿瞬间碰撞网墙
        rx_safe = sign_r * max(abs(rx), 1.5)
        target_x = target_half * 3.0

        waypoints = [
            (rx_safe, bypass_y),
            (target_x, bypass_y),
        ]

        print(f"🔀绕网到 {'X>0' if target_half > 0 else 'X<0'} 半场")
        for i, (wx, wy) in enumerate(waypoints):
            print(f"➡️ WP{i + 1}: ({wx:.1f}, {wy:.1f})")
            if not self.navigate_to(wx, wy, label=f"Bypass_WP{i + 1}"):
                print(f"❌ 绕网失败")
                return False
        print(f"✅ 绕网完成")
        return True

    # -----------------------------------------------------------------
    #  半场巡视（端点就近切入 + 往返覆盖）
    # -----------------------------------------------------------------

    def _build_patrol_points(self, active_half):
        """
        构建半场 S 形巡视路径点（6 个点，覆盖三个纵深层）。

        每层从一侧横扫到另一侧，层间纵向前进，形成 S 形。
        删除了中间点（中场中央、底线中央），因为 YouBot 从 Y- 侧走到
        Y+ 侧的直线本身就经过中央，加中间点只是多停顿、没有视野增益。

        俯视图（X>0 半场为例，Y+ 在左，Y- 在右）:

                        Y+                      Y-
                       │                       │
            5 ←────────┼────────── 4   x_far  (底线)
                       │           ↑
                       │           │
            2 ─────────┼─────────→ 3   x_mid  (中场)
            ↑          │
            │          │
            1 ←────────┼────────── 0     x_near (近网)
                       │
          ─────────────┼────────── X=0 (球网)

        路线: 0(近网,Y-) → 1(近网,Y+) → 2(中场,Y+)
              → 3(中场,Y-) → 4(底线,Y-) → 5(底线,Y+)

        就近切入规则：只比较到点 0 和点 5 的距离，
        从更近的端点开始，保证完整覆盖所有路径点。
        """
        sign = active_half
        x_near = sign * 2.5
        x_mid = sign * (HALF_COURT_X_MAX / 2)
        x_far = sign * (HALF_COURT_X_MAX - 1.5)
        # 巡视 Y 边界：椅子占用 |Y| ∈ [7.72, 8.55]，
        # 取 |Y| = 7.0 距椅子 0.72m，与绕网走廊 Y=7.05 也一致
        y_pos = 7.0
        y_neg = -7.0

        points = [
            (x_near, y_neg),     # 0: 近网 Y- 侧（S 起点）
            (x_near, y_pos),     # 1: 近网 Y+ 侧
            (x_mid,  y_pos),     # 2: 中场 Y+ 侧
            (x_mid,  y_neg),     # 3: 中场 Y- 侧
            (x_far,  y_neg),     # 4: 底线 Y- 侧
            (x_far,  y_pos),     # 5: 底线 Y+ 侧（S 终点）
        ]
        return points

    def patrol_half(self, active_half):
        """
        半场巡视：S 形端点就近切入，单程走完 6 个点。

        步骤：
          1. 计算 YouBot 到 S 形首端（点0）和尾端（点5）的距离
          2. 从更近的端点开始，沿 S 形走完全部 6 个点（单程，不再往返）
          3. 在移动过程中每隔若干仿真步检查视野里是否有当前半场的球，
             一旦发现立即停止巡视并返回 True，交给外层的 RL 去捡

        示例（从点0开始）:
          0 → 1 → 2 → 3 → 4 → 5

        示例（从点5开始，离5更近时）:
          5 → 4 → 3 → 2 → 1 → 0

        返回:
          True  - 巡视过程中发现了当前半场的球，已中断
          False - 走完全部点没发现球
        """
        # 巡视前先脱困：避免上一轮 stuck_at_boundary / stuck_at_net 残留贴墙状态
        # 导致 navigate_to 一直撞墙超时
        self._escape_from_obstacle(safe_margin=1.5, label="Pre-Patrol")

        rx, ry, _ = self._get_pose()
        points = self._build_patrol_points(active_half)
        n = len(points)

        # ── 只比较首端和尾端，决定从哪头开始 ──
        dist_to_start = math.hypot(points[0][0] - rx, points[0][1] - ry)
        dist_to_end = math.hypot(points[-1][0] - rx, points[-1][1] - ry)

        if dist_to_start <= dist_to_end:
            # 离点0更近 → 走 0→5
            route = list(range(n))
        else:
            # 离点5更近 → 走 5→0
            route = list(range(n - 1, -1, -1))

        start_label = route[0]
        end_label = route[-1]
        print(f"🔍 开始半场巡视（{'X>0' if active_half > 0 else 'X<0'}）")
        print(f"S 形单程: 点#{start_label} → 点#{end_label}")
        print(f"总导航次数: {len(route)}")

        # ── 构造"发现球"的中断检查回调 ──
        # 只有 env 可用时才启用；调 _build_single_obs 会刷新 frame_buffer，
        # 巡视结束后外层会调 soft_reset 清空 buffer，所以污染不是问题
        #
        # 双重保险避免对面球穿帮（距离估计在"车靠网+对面球贴网"时可能误判）:
        #   A. 严格阈值：要求球的估计世界坐标明显在己方半场内 (|est_bx| > 1.0)
        #      而非仅仅跨过 X=0，这样网边模糊区的球不会触发中断
        #   B. 连续确认：要求连续 PATROL_ABORT_CONFIRM 次检测都报告有球才中断，
        #      过滤相机噪声/单帧误检
        PATROL_EST_BX_MARGIN   = 1.0   # 方案 A：球 est_bx 到己方半场至少 1.0m
        PATROL_ABORT_CONFIRM   = 3     # 方案 B：连续 3 次检测到才中断（~2.25s 仿真）

        confirm_counter = [0]  # 用 list 包一层以便闭包修改

        def _ball_detected():
            if self.env is None:
                return False
            try:
                self.env._build_single_obs()
                # 默认 _build_single_obs 已经按 strict=True (est_bx > 0) 过滤
                # 这里再重新扫一遍 balls_all,用更严格的阈值二次过滤
                balls_all = getattr(self.env, '_last_balls_all', [])
                active = self.env.active_half
                has_confident_ball = False
                for b in balls_all:
                    est_bx = b.get('est_bx', 0.0)
                    if active > 0 and est_bx > PATROL_EST_BX_MARGIN:
                        has_confident_ball = True
                        break
                    if active < 0 and est_bx < -PATROL_EST_BX_MARGIN:
                        has_confident_ball = True
                        break
                if has_confident_ball:
                    confirm_counter[0] += 1
                else:
                    confirm_counter[0] = 0
                return confirm_counter[0] >= PATROL_ABORT_CONFIRM
            except Exception:
                return False

        abort_cb = _ball_detected if self.env is not None else None

        for step_i, pt_idx in enumerate(route):
            wx, wy = points[pt_idx]
            print(f"🚩→ 点#{pt_idx}: ({wx:.1f}, {wy:.1f})")
            result = self.navigate_to(
                wx, wy,
                reach_dist=1.0,
                label=f"Patrol_{pt_idx}",
                abort_check=abort_cb,
                # 每 15 个 sim.step 检查一次（~0.75s 仿真），避免 ZMQ 调用过密
                # 导致 CoppeliaSim server 过载崩溃
                abort_check_every=15,
            )
            if result == "aborted":
                print(f"🎾 巡视途中发现当前半场的球，中断巡视交给 RL")
                return True

        print(f"✅ 巡视完成（未发现球）")
        return False


# =====================================================================
#  主部署流程
# =====================================================================

def deploy(model_path, max_rounds=30):
    """
    完整捡球流程：
      RL Agent（半场内）+ 巡视确认 + 绕网切换
    """
    print("=" * 60)
    print("部署模式：RL Agent + 半场切换")
    print("=" * 60)

    # 加载 RL 模型
    print(f"\n📂 加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建环境
    active_half = 1
    env = TennisCollectorEnv(render_mode="human", active_half=active_half)

    # 兼容性检查：动作空间维度必须匹配（避免 V1 env 加载 V2 模型导致 KeyError）
    expected_n = env.action_space.n
    loaded_n = model.action_space.n if hasattr(model.action_space, 'n') else None
    if loaded_n != expected_n:
        env.close()
        raise ValueError(
            f"模型动作空间 ({loaded_n}) 与当前 env 动作空间 ({expected_n}) 不匹配。"
            f"V2 模型 (9 动作) 需配 tennis_rl_env2；V1 模型 (7 动作) 需配 tennis_rl_env。"
            f"请检查本文件顶部 `from tennis_rl_envX import ...` 与 --model 路径是否对应同一版本。"
        )
    print(f"✅ 动作空间一致 ({expected_n} 动作)")

    # 创建规则导航器
    # 传入 env 引用，让导航过程中能周期性刷新 OpenCV 调试窗口
    navigator = RuleNavigator(
        sim=env.sim,
        youbot_h=env.youbot_h,
        wheels=(env.fl, env.fr, env.rl, env.rr),
        env=env,
    )

    total_collected = 0
    no_ball_streak = 0
    NO_BALL_THRESHOLD = 3

    # 巡视中断（发现球）后，下一轮 RL 启动要跳过前 POST_PATROL_GRACE 步的卡住检测
    # 避免"巡视发现球 → RL 重启 → 瞬间卡住 → 又巡视"死循环 → ZMQ 过载崩溃
    patrol_just_found_ball = False

    # 部署启动：做一次真正的 env.reset() 初始化 frame buffer 和内部状态
    # 之后所有轮次都用 soft_reset()，保留 YouBot 位置由规则代码接管导航
    print("\n🔧 初始化环境（首次 reset 会随机选定部署起始位置）...")
    env.reset()

    for round_idx in range(max_rounds):
        print(f"\n{'=' * 40}")
        print(f"Round {round_idx + 1} | 半场: {'X>0' if active_half > 0 else 'X<0'} "
              f"| 已收集: {total_collected}")
        print(f"{'=' * 40}")

        # 刷新球数
        env._refresh_ball_handles()
        balls_in_half = env._count_balls_in_active_half()
        print(f"📋 当前半场剩余: {balls_in_half} 个球")

        if balls_in_half == 0:
            # 当前半场无球 → 巡视确认
            print(f"❓️当前半场看起来已清空，执行巡视确认...")
            found = navigator.patrol_half(active_half)
            if found:
                # 巡视中途看到了球 → 下一轮主循环会自然进入 RL 捡球
                # 这里不再走"巡视后检查 → 绕网"的分支
                patrol_just_found_ball = True
                continue

            # 巡视后再次检查
            env._refresh_ball_handles()
            balls_in_half = env._count_balls_in_active_half()

            if balls_in_half == 0:
                # 确认清空 → 检查另一半场
                other_half = -active_half
                env.active_half = other_half
                env._refresh_ball_handles()
                balls_other = env._count_balls_in_active_half()

                if balls_other == 0:
                    print(f"\n🎉 两个半场都已清空！")
                    print(f"📝 总计收集: {total_collected} 个球")
                    break

                # 绕网切换
                navigator.bypass_net(other_half)
                active_half = other_half
                env.active_half = active_half
                no_ball_streak = 0
                continue
            else:
                print(f"⚠️ 巡视后发现还有 {balls_in_half} 个球！继续捡...")
                no_ball_streak = 0

        # ── RL Agent 捡球 ──
        # 部署阶段用 soft_reset：保留 YouBot 当前位置，只清 episode 状态
        # 这样上一轮结束的位置、刚绕网过来的位置、刚巡视到的位置都能被继承
        obs, info = env.soft_reset()
        done = False
        ep_reward = 0

        # Action smoothing: 连续 N 次预测到新动作才真正切换,避免高频抖动
        last_action = None
        switch_count = 0
        REQUIRED_SWITCH_CONFIRM = 1  # 需要连续 2 步确认才切换

        # 原地摇头检测：滑动窗口记录最近 STUCK_WINDOW 步的 (x, y)
        # 一旦窗口满且位置范围都 < STUCK_THRESHOLD，跳出 RL 循环触发巡视
        pos_window = collections.deque(maxlen=STUCK_WINDOW)
        stuck_triggered = False

        # 宽限期：如果本轮是巡视发现球后紧接着的 RL，前 POST_PATROL_GRACE 步
        # 跳过卡住检测，给 RL 充足时间定位到球
        grace_steps_remaining = POST_PATROL_GRACE if patrol_just_found_ball else 0
        patrol_just_found_ball = False  # 用过即清

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            if last_action is None or action == last_action:
                # 第一步 or 预测和上一步一致 → 直接执行
                final_action = action
                switch_count = 0
            else:
                # 预测到了不同的动作 → 累计确认次数
                switch_count += 1
                if switch_count >= REQUIRED_SWITCH_CONFIRM:
                    final_action = action
                    switch_count = 0
                else:
                    final_action = last_action  # 还没确认够,维持上一个动作

            obs, reward, terminated, truncated, info = env.step(final_action)
            last_action = final_action
            ep_reward += reward
            done = terminated or truncated

            # 记录位置并检查是否"原地摇头"（宽限期内跳过）
            rx_now, ry_now, _ = env._get_youbot_pose()
            if grace_steps_remaining > 0:
                grace_steps_remaining -= 1
                continue
            pos_window.append((rx_now, ry_now))
            if len(pos_window) >= STUCK_WINDOW:
                xs = [p[0] for p in pos_window]
                ys = [p[1] for p in pos_window]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                if x_range < STUCK_THRESHOLD and y_range < STUCK_THRESHOLD:
                    print(f"🌀 检测到原地摇头 "
                          f"(最近 {STUCK_WINDOW} 步 Δx={x_range:.2f}m Δy={y_range:.2f}m)"
                          f"，跳出 RL 循环触发巡视")
                    stuck_triggered = True
                    break

        """
        # 未加平滑版，暂时保留
        # ── RL Agent 捡球 ──
        obs, info = env.soft_reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        """


        if stuck_triggered:
            # 卡住不计入 RL 失败，直接巡视打破僵局
            print(f"🔍 卡住触发：执行巡视寻找球...")
            found = navigator.patrol_half(active_half)
            if found:
                patrol_just_found_ball = True
            no_ball_streak = 0
        elif info.get('success', False):
            total_collected += 1
            no_ball_streak = 0
            print(f"✅ 捡到球！(奖励={ep_reward:+.1f}, 步数={info['step']})")
        else:
            no_ball_streak += 1
            print(f"❌ 未捡到 (原因={info.get('reason', '?')}, "
                  f"奖励={ep_reward:+.1f}, 连续失败={no_ball_streak})")

        # 连续失败 → 巡视
        if no_ball_streak >= NO_BALL_THRESHOLD:
            print(f"  🔍 连续 {no_ball_streak} 次失败，触发巡视...")
            found = navigator.patrol_half(active_half)
            if found:
                patrol_just_found_ball = True
            no_ball_streak = 0

    env.close()

    print(f"\n{'=' * 60}")
    print(f"🏁 部署结束 | 总计收集: {total_collected} 个球")
    print(f"{'=' * 60}")


# =====================================================================
#  入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="部署 RL 捡球 Agent")
    parser.add_argument("--model", type=str, default="./models_v2/best_model/best_model.zip", help="模型文件路径")
    parser.add_argument("--rounds", type=int, default=30, help="最大轮数")
    args = parser.parse_args()

    print(f"开始：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    start_time = time.time()
    deploy(args.model, max_rounds=args.rounds)
    span_time = time.time() - start_time
    print(f"用时{span_time}")
    print(f"\n结束：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
