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
import numpy as np
from stable_baselines3 import PPO
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from tennis_rl_env import (
    TennisCollectorEnv,
    BALL_COUNT,
    HALF_COURT_X_MAX,
    HALF_COURT_Y_MAX,
    OUTER_WIDTH,
    NET_WALL_Y_HALF,
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


# =====================================================================
#  规则导航器
# =====================================================================

class RuleNavigator:
    """规则导航器：绕网 + 巡视"""

    def __init__(self, sim, youbot_h, wheels):
        self.sim = sim
        self.youbot_h = youbot_h
        self.fl, self.fr, self.rl, self.rr = wheels

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

    @staticmethod
    def _angle_diff(a, b):
        diff = a - b
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def _drive_to(self, smooth_angle, dist):
        FWD_KP, TURN_KP = 1.8, 2.0
        FWD_MAX, TURN_MAX = BASE_SPEED, TURN_SPEED
        v_turn = max(-TURN_MAX, min(TURN_MAX, TURN_KP * smooth_angle))
        if abs(smooth_angle) > math.radians(90):
            self._set_motors(v_turn, -v_turn, v_turn, -v_turn)
        else:
            v_fwd = max(0.3, min(FWD_MAX, FWD_KP * dist * math.cos(smooth_angle)))
            self._set_motors(v_fwd + v_turn, v_fwd - v_turn,
                             v_fwd + v_turn, v_fwd - v_turn)

    def navigate_to(self, tx, ty, reach_dist=WAYPOINT_REACHED, label="WP"):
        """导航到目标点，返回 True=到达。

        注意: 仿真处于 stepping 模式（env 初始化时设置），
        必须在循环里显式调用 sim.step() 推进仿真，
        单靠 time.sleep 仿真不会动，server 会因超时 abort。
        """
        smooth_err = None
        for step in range(NAV_MAX_ITER):
            rx, ry, ryaw = self._get_pose()
            dx, dy = tx - rx, ty - ry
            dist = math.hypot(dx, dy)
            if dist < reach_dist:
                self._stop()
                # 送一小段稳定帧，让电机真正停下
                for _ in range(3):
                    self.sim.step()
                return True
            raw_err = self._angle_diff(math.atan2(dy, dx), ryaw)
            if smooth_err is None:
                smooth_err = raw_err
            else:
                smooth_err = (1 - NAV_ALPHA) * smooth_err + NAV_ALPHA * raw_err
            self._drive_to(smooth_err, dist)
            self.sim.step()   # ← 显式推进仿真（替代 time.sleep）
        self._stop()
        for _ in range(3):
            self.sim.step()
        print(f"[Deploy] ⚠️ {label} 导航超时")
        return False

    # -----------------------------------------------------------------
    #  绕网
    # -----------------------------------------------------------------

    def bypass_net(self, target_half):
        """绕网到目标半场（+1 或 -1）。

        三段式路径（以 X>0 → X<0 为例）:
          WP1: 先纵向退到 Y=±7.05m 的安全走廊（避开椅子 Y∈[7.72,8.55]，
               也在网柱 |Y|=6.40 之外可以横穿 X=0）
          WP2: 横向穿越 X=0，到达目标半场内 X=±3
          WP3: 纵向回到场中 Y=0，交给 RL agent 接管
        """
        rx, ry, _ = self._get_pose()

        bypass_y = NET_BYPASS_Y if ry >= 0 else -NET_BYPASS_Y
        sign_r = 1 if rx >= 0 else -1
        # 起步先离网 ≥ 1.5m，避免贴网时 WP2 横穿瞬间碰撞网墙
        rx_safe = sign_r * max(abs(rx), 1.5)
        target_x = target_half * 3.0

        waypoints = [
            (rx_safe, bypass_y),
            (target_x, bypass_y),
            (target_x, 0.0),
        ]

        print(f"[Deploy] 🔀 绕网到 {'X>0' if target_half > 0 else 'X<0'} 半场")
        for i, (wx, wy) in enumerate(waypoints):
            print(f"[Deploy] ➡️ WP{i + 1}: ({wx:.1f}, {wy:.1f})")
            if not self.navigate_to(wx, wy, label=f"Bypass_WP{i + 1}"):
                print(f"[Deploy] ❌ 绕网失败")
                return False
        print(f"[Deploy] ✅ 绕网完成")
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
            5 ←─────┼────────────── 4   x_far  (底线)
                    │               ↑
                    │               │
            2 ──────┼─────────────→ 3   x_mid  (中场)
            ↑       │
            │       │
            1 ←─────┼────────────── 0     x_near (近网)
                    │
          ──────────┼────────────── X=0 (球网)

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
        半场巡视：S 形端点就近切入 + 正向走完 + 原路返回。

        步骤：
          1. 计算 YouBot 到 S 形首端（点0）和尾端（点5）的距离
          2. 从更近的端点开始，沿 S 形走完全部 6 个点
          3. 到达另一端后原路返回
          4. 往返两趟保证每个区域被正反两个方向的视野覆盖

        示例（从点0开始）:
          正向: 0 → 1 → 2 → 3 → 4 → 5
          返回: 5 → 4 → 3 → 2 → 1 → 0

        示例（从点5开始，离5更近时）:
          正向: 5 → 4 → 3 → 2 → 1 → 0
          返回: 0 → 1 → 2 → 3 → 4 → 5
        """
        rx, ry, _ = self._get_pose()
        points = self._build_patrol_points(active_half)
        n = len(points)

        # ── 只比较首端和尾端，决定从哪头开始 ──
        dist_to_start = math.hypot(points[0][0] - rx, points[0][1] - ry)
        dist_to_end = math.hypot(points[-1][0] - rx, points[-1][1] - ry)

        if dist_to_start <= dist_to_end:
            # 离点0更近 → 正向 0→5，返回 5→0
            forward_order = list(range(n))
        else:
            # 离点5更近 → 反向 5→0，返回 0→5
            forward_order = list(range(n - 1, -1, -1))

        backward_order = list(reversed(forward_order))
        full_route = forward_order + backward_order

        start_label = forward_order[0]
        end_label = forward_order[-1]
        print(f"[Deploy] 🔍 开始半场巡视（{'X>0' if active_half > 0 else 'X<0'}）")
        print(f"[Deploy] S 形路径: 点#{start_label} → 点#{end_label} → 点#{start_label}")
        print(f"[Deploy] 总导航次数: {len(full_route)}")

        for step_i, pt_idx in enumerate(full_route):
            wx, wy = points[pt_idx]
            is_forward = step_i < len(forward_order)
            phase = "正向" if is_forward else "返回"
            arrow = "→" if is_forward else "←"
            print(f"{arrow} [{phase}] 点#{pt_idx}: ({wx:.1f}, {wy:.1f})")
            self.navigate_to(wx, wy, reach_dist=1.0, label=f"Patrol_{phase}_{pt_idx}")

        print(f"[Deploy] ✅ 巡视完成（S 形往返覆盖）")


# =====================================================================
#  主部署流程
# =====================================================================

def deploy(model_path, max_rounds=30):
    """
    完整捡球流程：
      RL Agent（半场内）+ 巡视确认 + 绕网切换
    """
    print("=" * 60)
    print("[Deploy] 部署模式：RL Agent + 半场切换")
    print("=" * 60)

    # 加载 RL 模型
    print(f"\n[Deploy] 📂 加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建环境
    active_half = 1
    env = TennisCollectorEnv(render_mode="human", active_half=active_half)

    # 创建规则导航器
    navigator = RuleNavigator(
        sim=env.sim,
        youbot_h=env.youbot_h,
        wheels=(env.fl, env.fr, env.rl, env.rr),
    )

    total_collected = 0
    no_ball_streak = 0
    NO_BALL_THRESHOLD = 3

    # 部署启动：做一次真正的 env.reset() 初始化 frame buffer 和内部状态
    # 之后所有轮次都用 soft_reset()，保留 YouBot 位置由规则代码接管导航
    print("\n🔧 初始化环境（首次 reset 会随机选定部署起始位置）...")
    env.reset()

    for round_idx in range(max_rounds):
        print(f"\n{'=' * 40}")
        print(f"[Deploy] Round {round_idx + 1} | 半场: {'X>0' if active_half > 0 else 'X<0'} "
              f"| 已收集: {total_collected}")
        print(f"{'=' * 40}")

        # 刷新球数
        env._refresh_ball_handles()
        balls_in_half = env._count_balls_in_active_half()
        print(f"[Deploy] 当前半场剩余: {balls_in_half} 个球")

        if balls_in_half == 0:
            # 当前半场无球 → 巡视确认
            print(f"[Deploy] 📋 当前半场看起来已清空，执行巡视确认...")
            navigator.patrol_half(active_half)

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
                    print(f"\n[Deploy] 🎉 两个半场都已清空！")
                    print(f"[Deploy] 总计收集: {total_collected} 个球")
                    break

                # 绕网切换
                navigator.bypass_net(other_half)
                active_half = other_half
                env.active_half = active_half
                no_ball_streak = 0
                continue
            else:
                print(f"  ⚠️ 巡视后发现还有 {balls_in_half} 个球！继续捡...")
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
        REQUIRED_SWITCH_CONFIRM = 2  # 需要连续 2 步确认才切换

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


        if info.get('success', False):
            total_collected += 1
            no_ball_streak = 0
            print(f"[Deploy] ✅ 捡到球！(奖励={ep_reward:+.1f}, 步数={info['step']})")
        else:
            no_ball_streak += 1
            print(f"[Deploy] ❌ 未捡到 (原因={info.get('reason', '?')}, "
                  f"奖励={ep_reward:+.1f}, 连续失败={no_ball_streak})")

        # 连续失败 → 巡视
        if no_ball_streak >= NO_BALL_THRESHOLD:
            print(f"[Deploy] 🔍 连续 {no_ball_streak} 次失败，触发巡视...")
            navigator.patrol_half(active_half)
            no_ball_streak = 0

    env.close()

    print(f"\n{'=' * 60}")
    print(f"[Deploy] 🏁 部署结束 | 总计收集: {total_collected} 个球")
    print(f"{'=' * 60}")


# =====================================================================
#  入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="部署 RL 捡球 Agent")
    parser.add_argument("--model", type=str, default="./models/best_model/best_model.zip", help="模型文件路径")
    parser.add_argument("--rounds", type=int, default=30, help="最大轮数")
    args = parser.parse_args()

    deploy(args.model, max_rounds=args.rounds)
