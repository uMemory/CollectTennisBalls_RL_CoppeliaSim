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

NET_BYPASS_Y       = 7.40
COURT_X_HALF       = 11.885
WAYPOINT_REACHED   = 0.50
NAV_MAX_ITER       = 2000
NAV_LOOP_DT        = 0.05
NAV_ALPHA          = 0.15
BASE_SPEED         = 7.5
TURN_SPEED         = 3.5


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
        """导航到目标点，返回 True=到达"""
        smooth_err = None
        for step in range(NAV_MAX_ITER):
            rx, ry, ryaw = self._get_pose()
            dx, dy = tx - rx, ty - ry
            dist = math.hypot(dx, dy)
            if dist < reach_dist:
                self._stop()
                return True
            raw_err = self._angle_diff(math.atan2(dy, dx), ryaw)
            if smooth_err is None:
                smooth_err = raw_err
            else:
                smooth_err = (1 - NAV_ALPHA) * smooth_err + NAV_ALPHA * raw_err
            self._drive_to(smooth_err, dist)
            time.sleep(NAV_LOOP_DT)
        self._stop()
        print(f"  ⚠️ {label} 导航超时")
        return False

    # -----------------------------------------------------------------
    #  绕网
    # -----------------------------------------------------------------

    def bypass_net(self, target_half):
        """绕网到目标半场（+1 或 -1）"""
        rx, ry, _ = self._get_pose()

        bypass_y = NET_BYPASS_Y if ry >= 0 else -NET_BYPASS_Y
        sign_r = 1 if rx >= 0 else -1
        rx_safe = sign_r * max(abs(rx), 1.0)
        target_x = target_half * 3.0

        waypoints = [
            (rx_safe, bypass_y),
            (target_x, bypass_y),
            (target_x, 0.0),
        ]

        print(f"  🔀 绕网到 {'X>0' if target_half > 0 else 'X<0'} 半场")
        for i, (wx, wy) in enumerate(waypoints):
            print(f"    ➡️ WP{i + 1}: ({wx:.1f}, {wy:.1f})")
            if not self.navigate_to(wx, wy, label=f"Bypass_WP{i + 1}"):
                print(f"  ❌ 绕网失败")
                return False
        print(f"  ✅ 绕网完成")
        return True

    # -----------------------------------------------------------------
    #  半场巡视（端点就近切入 + 往返覆盖）
    # -----------------------------------------------------------------

    def _build_patrol_points(self, active_half):
        """
        构建半场 S 形巡视路径点（8 个点，覆盖三个纵深层）。

        每层从一侧横扫到另一侧，层间纵向前进，形成真正的 S 形。

        俯视图（X>0 半场为例，Y+ 在左，Y- 在右）:

                        Y+                      Y-
                    │                       │
            7 ←─────┼── 6 ←────── 5   x_far  (底线)
                    │             ↑
                    │             │
            2 ──────┼→ 3 ─────→ 4 ┘   x_mid  (中场)
            ↑       │
            │       │
            1 ←─────┼──────────── 0     x_near (近网)
                    │
          ──────────┼──────────── X=0 (球网)

        路线: 0(近网,Y-) → 1(近网,Y+) → 2(中场,Y+) → 3(中场,中)
              → 4(中场,Y-) → 5(底线,Y-) → 6(底线,中) → 7(底线,Y+)

        就近切入规则：只比较到点 0 和点 7 的距离，
        从更近的端点开始，保证完整覆盖所有路径点。
        """
        sign = active_half
        x_near = sign * 2.5
        x_mid = sign * (HALF_COURT_X_MAX / 2)
        x_far = sign * (HALF_COURT_X_MAX - 1.5)
        y_pos = HALF_COURT_Y_MAX - 1.5     # Y+ 侧
        y_neg = -HALF_COURT_Y_MAX + 1.5    # Y- 侧

        points = [
            (x_near, y_neg),     # 0: 近网 Y- 侧（S 起点）
            (x_near, y_pos),     # 1: 近网 Y+ 侧
            (x_mid,  y_pos),     # 2: 中场 Y+ 侧
            (x_mid,  0.0),       # 3: 中场 中央
            (x_mid,  y_neg),     # 4: 中场 Y- 侧
            (x_far,  y_neg),     # 5: 底线 Y- 侧
            (x_far,  0.0),       # 6: 底线 中央
            (x_far,  y_pos),     # 7: 底线 Y+ 侧（S 终点）
        ]
        return points

    def patrol_half(self, active_half):
        """
        半场巡视：S 形端点就近切入 + 正向走完 + 原路返回。

        步骤：
          1. 计算 YouBot 到 S 形首端（点0）和尾端（点7）的距离
          2. 从更近的端点开始，沿 S 形走完全部 8 个点
          3. 到达另一端后原路返回
          4. 往返两趟保证每个区域被正反两个方向的视野覆盖

        示例（从点0开始）:
          正向: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
          返回: 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0

        示例（从点7开始，离7更近时）:
          正向: 7 → 6 → 5 → 4 → 3 → 2 → 1 → 0
          返回: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
        """
        rx, ry, _ = self._get_pose()
        points = self._build_patrol_points(active_half)
        n = len(points)

        # ── 只比较首端和尾端，决定从哪头开始 ──
        dist_to_start = math.hypot(points[0][0] - rx, points[0][1] - ry)
        dist_to_end = math.hypot(points[-1][0] - rx, points[-1][1] - ry)

        if dist_to_start <= dist_to_end:
            # 离点0更近 → 正向 0→7，返回 7→0
            forward_order = list(range(n))
        else:
            # 离点7更近 → 反向 7→0，返回 0→7
            forward_order = list(range(n - 1, -1, -1))

        backward_order = list(reversed(forward_order))
        full_route = forward_order + backward_order

        start_label = forward_order[0]
        end_label = forward_order[-1]
        print(f"  🔍 开始半场巡视（{'X>0' if active_half > 0 else 'X<0'}）")
        print(f"     S 形路径: 点#{start_label} → 点#{end_label} → 点#{start_label}")
        print(f"     总导航次数: {len(full_route)}")

        for step_i, pt_idx in enumerate(full_route):
            wx, wy = points[pt_idx]
            is_forward = step_i < len(forward_order)
            phase = "正向" if is_forward else "返回"
            arrow = "→" if is_forward else "←"
            print(f"    {arrow} [{phase}] 点#{pt_idx}: ({wx:.1f}, {wy:.1f})")
            self.navigate_to(wx, wy, reach_dist=1.0, label=f"Patrol_{phase}_{pt_idx}")

        print(f"  ✅ 巡视完成（S 形往返覆盖）")


# =====================================================================
#  主部署流程
# =====================================================================

def deploy(model_path, max_rounds=30):
    """
    完整捡球流程：
      RL Agent（半场内）+ 巡视确认 + 绕网切换
    """
    print("=" * 60)
    print("  部署模式：RL Agent + 半场切换")
    print("=" * 60)

    # 加载 RL 模型
    print(f"\n📂 加载模型: {model_path}")
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

    for round_idx in range(max_rounds):
        print(f"\n{'=' * 40}")
        print(f"  Round {round_idx + 1} | 半场: {'X>0' if active_half > 0 else 'X<0'} "
              f"| 已收集: {total_collected}")
        print(f"{'=' * 40}")

        # 刷新球数
        env._refresh_ball_handles()
        balls_in_half = env._count_balls_in_active_half()
        print(f"  当前半场剩余: {balls_in_half} 个球")

        if balls_in_half == 0:
            # 当前半场无球 → 巡视确认
            print(f"  📋 当前半场看起来已清空，执行巡视确认...")
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
                    print(f"\n🎉 两个半场都已清空！")
                    print(f"   总计收集: {total_collected} 个球")
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
        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        if info.get('success', False):
            total_collected += 1
            no_ball_streak = 0
            print(f"  ✅ 捡到球！(奖励={ep_reward:+.1f}, 步数={info['step']})")
        else:
            no_ball_streak += 1
            print(f"  ❌ 未捡到 (原因={info.get('reason', '?')}, "
                  f"奖励={ep_reward:+.1f}, 连续失败={no_ball_streak})")

        # 连续失败 → 巡视
        if no_ball_streak >= NO_BALL_THRESHOLD:
            print(f"  🔍 连续 {no_ball_streak} 次失败，触发巡视...")
            navigator.patrol_half(active_half)
            no_ball_streak = 0

    env.close()

    print(f"\n{'=' * 60}")
    print(f"  🏁 部署结束 | 总计收集: {total_collected} 个球")
    print(f"{'=' * 60}")


# =====================================================================
#  入口
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="部署 RL 捡球 Agent")
    parser.add_argument("--model", type=str, default="./models/best/best_model.zip", help="模型文件路径")
    parser.add_argument("--rounds", type=int, default=30, help="最大轮数")
    args = parser.parse_args()

    deploy(args.model, max_rounds=args.rounds)
