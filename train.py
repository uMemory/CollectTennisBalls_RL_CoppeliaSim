"""
train.py (v5)
核心修复: ACTION_REPEAT — 每个动作保持多个仿真步
─────────────────────────────────────────────────────
根因: PPO 每步输出随机动作, 仿真步长 50ms → 20Hz 动作切换
      连续两步动作可能完全相反 → 高频抖动
修复: 每个动作保持 ACTION_REPEAT=5 个仿真步 (250ms)
      等效动作频率 4Hz, 机器人有充足时间响应
─────────────────────────────────────────────────────
"""

import time
import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# ══════════════════════════════════════════════════════════════
#  YouBot Controller
# ══════════════════════════════════════════════════════════════
class YouBotController:
    # ── 关键参数 ──────────────────────────────────────────────
    SPEED_SCALE   = 2.0     # 轮速系数 (YouBot 麦轮最大约 10 rad/s)
    ACTION_REPEAT = 5       # ★ 核心修复: 每个动作保持 5 个仿真步
    SMOOTH        = 0.4     # 动作平滑系数 (0=无平滑, 1=完全保持旧值)

    def __init__(self):
        client   = RemoteAPIClient()
        self.sim = client.require('sim')
        sim      = self.sim

        # ── 同步模式 ─────────────────────────────────────────
        sim.setStepping(True)
        print("✅ Synchronous mode enabled")

        # ── 获取句柄 ─────────────────────────────────────────
        self.body = sim.getObject('/youBot')
        self.fl   = sim.getObject('/rollingJoint_fl')
        self.fr   = sim.getObject('/rollingJoint_fr')
        self.rl   = sim.getObject('/rollingJoint_rl')
        self.rr   = sim.getObject('/rollingJoint_rr')

        # 4.10.0 API: 不传第二参数 = 世界坐标系
        self.init_pos = list(sim.getObjectPosition(self.body))
        self.init_ori = list(sim.getObjectOrientation(self.body))

        self._prev_action   = [0.0, 0.0, 0.0]
        self._reset_counter = 0

        print("✅ YouBotController initialized")
        print(f"   Start pos : {[f'{v:.3f}' for v in self.init_pos]}")
        print(f"   ACTION_REPEAT = {self.ACTION_REPEAT}")
        print(f"   Effective action freq ≈ {1.0 / (0.05 * self.ACTION_REPEAT):.1f} Hz")

    # ── 运动控制 ──────────────────────────────────────────────
    def move(self, vx: float, vy: float, omega: float):
        """设置轮速, 内含平滑"""
        # 平滑 (对动作而非轮速做平滑, 保证四轮协调)
        s  = self.SMOOTH
        vx    = s * self._prev_action[0] + (1 - s) * vx
        vy    = s * self._prev_action[1] + (1 - s) * vy
        omega = s * self._prev_action[2] + (1 - s) * omega
        self._prev_action = [vx, vy, omega]

        # YouBot 麦轮运动学
        spd = self.SPEED_SCALE
        self.sim.setJointTargetVelocity(self.fl, (-vx - vy - omega) * spd)
        self.sim.setJointTargetVelocity(self.rl, (-vx + vy - omega) * spd)
        self.sim.setJointTargetVelocity(self.rr, (-vx - vy + omega) * spd)
        self.sim.setJointTargetVelocity(self.fr, (-vx + vy + omega) * spd)

    def step_action(self):
        """★ 核心: 保持当前轮速不变, 推进 ACTION_REPEAT 个仿真步"""
        for _ in range(self.ACTION_REPEAT):
            self.sim.step()

    def stop(self):
        self._prev_action = [0.0, 0.0, 0.0]
        for j in [self.fl, self.fr, self.rl, self.rr]:
            self.sim.setJointTargetVelocity(j, 0)

    def get_pose(self):
        pos = self.sim.getObjectPosition(self.body)
        ori = self.sim.getObjectOrientation(self.body)
        return pos[0], pos[1], ori[2]

    def teleport_to_start(self):
        self.stop()
        self.sim.setObjectPosition(self.body, self.init_pos)
        self.sim.setObjectOrientation(self.body, self.init_ori)
        self._prev_action = [0.0, 0.0, 0.0]

    def reset_episode(self):
        self.teleport_to_start()
        self._reset_counter += 1
        self.sim.setInt32Signal('reset_counter', self._reset_counter)
        # 给仿真几步时间稳定
        for _ in range(10):
            self.sim.step()

    # ── 信号读取 ──────────────────────────────────────────────
    def get_carried(self) -> int:
        try:
            v = self.sim.getInt32Signal('carried_balls')
            return v if v is not None else 0
        except:
            return 0

    def get_scored(self) -> int:
        try:
            v = self.sim.getInt32Signal('scored_balls')
            return v if v is not None else 0
        except:
            return 0

    def get_remaining(self) -> int:
        try:
            v = self.sim.getInt32Signal('remaining_balls')
            return v if v is not None else 12
        except:
            return 12

    def get_nearest_ball(self, n=12):
        rx, ry, _ = self.get_pose()
        min_dist  = float('inf')
        ndx = ndy = 0.0
        for i in range(1, n + 1):
            try:
                h   = self.sim.getObject(f'/TennisBall_{i:02d}')
                pos = self.sim.getObjectPosition(h)
                if pos[2] > 0.01:
                    dx   = pos[0] - rx
                    dy   = pos[1] - ry
                    dist = math.hypot(dx, dy)
                    if dist < min_dist:
                        min_dist = dist
                        ndx, ndy = dx, dy
            except:
                pass
        if min_dist == float('inf'):
            return 0.0, 0.0, 0.0
        return ndx, ndy, min_dist


# ══════════════════════════════════════════════════════════════
#  Gym Environment
# ══════════════════════════════════════════════════════════════
class TennisEnv(gym.Env):
    NUM_BALLS = 12
    MAX_CARRY = 5

    # ★ 根据 v7 场景真实尺寸更新 ★
    # 场地 23.77×10.97m, 围栏 38.6×20.3m
    FENCE_L   = 38.6
    FENCE_W   = 20.3
    BIN_POS   = np.array([14.4, 8.2])   # 回收仓位置

    # 每个 env.step() = ACTION_REPEAT 个仿真步 = 250ms
    # 3000 steps × 250ms = 750s = 12.5 分钟仿真时间, 足够
    MAX_STEPS = 3000

    def __init__(self):
        super().__init__()
        self.robot = YouBotController()

        self.action_space = spaces.Box(
            low  = np.full(3, -1.0, dtype=np.float32),
            high = np.full(3,  1.0, dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._step         = 0
        self._prev_scored  = 0
        self._prev_carried = 0
        self._prev_ball_dist = None
        self._prev_bin_dist  = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.reset_episode()
        self._step           = 0
        self._prev_scored    = 0
        self._prev_carried   = 0
        self._prev_ball_dist = None
        self._prev_bin_dist  = None
        return self._obs(), {}

    def step(self, action):
        vx    = float(action[0])
        vy    = float(action[1])
        omega = float(action[2])

        # ★ 设置轮速 (含平滑)
        self.robot.move(vx, vy, omega)

        # ★ 保持动作推进 ACTION_REPEAT 步 (核心修复)
        self.robot.step_action()

        self._step += 1

        # ── 读取状态 ─────────────────────────────────────────
        rx, ry, _           = self.robot.get_pose()
        carried             = self.robot.get_carried()
        scored              = self.robot.get_scored()
        remaining           = self.robot.get_remaining()
        bdx, bdy, ball_dist = self.robot.get_nearest_ball()
        bin_dx              = self.BIN_POS[0] - rx
        bin_dy              = self.BIN_POS[1] - ry
        bin_dist            = math.hypot(bin_dx, bin_dy)

        # ── 奖励 ─────────────────────────────────────────────
        reward     = -0.01          # 轻微时间惩罚
        terminated = False
        truncated  = self._step >= self.MAX_STEPS

        # 出界惩罚 + 重置位置
        half_l = self.FENCE_L / 2 - 0.5
        half_w = self.FENCE_W / 2 - 0.5
        if abs(rx) > half_l or abs(ry) > half_w:
            reward -= 5.0
            self.robot.teleport_to_start()
            for _ in range(3):
                self.robot.step_action()
            rx, ry, _ = self.robot.get_pose()

        # 接近球奖励
        if carried < self.MAX_CARRY and ball_dist > 0:
            if self._prev_ball_dist is not None:
                delta = self._prev_ball_dist - ball_dist
                reward += 2.0 * delta
            self._prev_ball_dist = ball_dist
        else:
            self._prev_ball_dist = None

        # 拾取奖励
        if carried > self._prev_carried:
            reward += 20.0 * (carried - self._prev_carried)
        self._prev_carried = carried

        # 接近仓奖励
        if carried > 0:
            if self._prev_bin_dist is not None:
                delta = self._prev_bin_dist - bin_dist
                reward += 1.5 * delta
            self._prev_bin_dist = bin_dist
        else:
            self._prev_bin_dist = None

        # 投放奖励
        if scored > self._prev_scored:
            reward           += 30.0 * (scored - self._prev_scored)
            self._prev_scored = scored

        # 全部完成
        if scored >= self.NUM_BALLS:
            reward    += 150.0
            terminated = True

        info = {
            'carried'  : carried,
            'scored'   : scored,
            'remaining': remaining,
            'step'     : self._step,
        }
        return self._obs(), reward, terminated, truncated, info

    def _obs(self):
        rx, ry, theta       = self.robot.get_pose()
        carried             = self.robot.get_carried()
        remaining           = self.robot.get_remaining()
        bdx, bdy, ball_dist = self.robot.get_nearest_ball()
        bin_dx              = self.BIN_POS[0] - rx
        bin_dy              = self.BIN_POS[1] - ry
        bin_dist            = math.hypot(bin_dx, bin_dy)

        return np.array([
            rx        / (self.FENCE_L / 2),
            ry        / (self.FENCE_W / 2),
            theta     / math.pi,
            bdx       / self.FENCE_L,
            bdy       / self.FENCE_W,
            ball_dist / self.FENCE_L,
            bin_dx    / self.FENCE_L,
            bin_dy    / self.FENCE_W,
            bin_dist  / self.FENCE_L,
            carried   / self.MAX_CARRY,
            remaining / self.NUM_BALLS,
        ], dtype=np.float32)

    def close(self):
        self.robot.stop()


# ══════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import torch
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs',   exist_ok=True)

    env = Monitor(TennisEnv())

    model = PPO(
        'MlpPolicy', env,
        learning_rate   = 3e-4,
        n_steps         = 1024,     # 增大 buffer (每次收集更多经验)
        batch_size      = 64,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        policy_kwargs   = dict(net_arch=[256, 256, 128]),
        tensorboard_log = './logs/',
        verbose         = 1,
        device          = 'cpu',
    )

    print("\n" + "=" * 60)
    print("  YouBot Tennis Ball Collector — RL Training v5")
    print("  ★ 核心修复: ACTION_REPEAT=5 (动作保持 250ms)")
    print("  Make sure CoppeliaSim simulation is RUNNING")
    print("=" * 60 + "\n")

    model.learn(
        total_timesteps     = 500_000,
        callback            = CheckpointCallback(
            save_freq   = 20_000,
            save_path   = './models/',
            name_prefix = 'youbot_ppo_v5',
        ),
        tb_log_name         = 'YouBot_v5',
        reset_num_timesteps = True,
    )

    model.save('models/youbot_final_v5')
    print('✅ Training complete! Model saved -> models/youbot_final_v5')
    env.close()