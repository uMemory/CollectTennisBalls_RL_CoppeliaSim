"""
train_v2.py
===========
PPO 训练脚本 v2：对 train.py 的全面优化版本。

相比 train.py 的改动:
  1. 导入 tennis_rl_env_v2.TennisCollectorEnv (动作空间 7→9, 新增后退动作)
  2. 奖励加"视野新鲜度"线性递增惩罚 (env 端改动)
  3. learning_rate 从 3e-4 线性衰减到 1e-4 (PPO 原生支持)
  4. ent_coef 从 0.01 线性衰减到 0.003 (通过 EntCoefScheduleCallback 实现,
     因为 SB3 原生不支持 ent_coef 为 schedule)
  5. n_steps 1024 → 2048 (更稳定的 advantage 估计)

重要提示:
  - v2 的动作空间是 9 维，**不能**加载 train.py (7 维) 训的模型 resume
  - v2 的奖励函数不同，新训的模型也不能用旧 Deploy 的期望值直接评估
  - 建议从头训练 300k~500k step

用法 (独立于 train.py, 不影响现有流程):
  1. 打开 CoppeliaSim，加载网球场景
  2. 执行 tennis_scene_latest.lua（生成场地）
  3. 执行 Tennis_Generate.lua（生成网球）
  4. 点 Play ▶ 启动仿真
  5. 运行本脚本：python train_v2.py

产出目录:
  ./logs_v2/          TensorBoard 日志 (独立于 train.py 的 ./logs/)
  ./models_v2/        定期保存的模型检查点
  ./models_v2/final/  最终模型
"""

import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# v2: 导入新版本环境
from tennis_rl_env2 import TennisCollectorEnv


# =====================================================================
#  训练参数
# =====================================================================

# ── 环境参数 ──
ACTIVE_HALF    = 1         # 训练在 X>0 半场

# ── PPO 超参数 ──
TOTAL_TIMESTEPS = 200_000
# v2: 学习率从 3e-4 线性衰减到 1e-4,避免后期抖动
LEARNING_RATE_INIT = 3e-4
LEARNING_RATE_END  = 1e-4
N_STEPS         = 2048     # v2: 1024 → 2048, 更稳定的 advantage 估计
BATCH_SIZE      = 64
N_EPOCHS        = 10
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
# v2: ent_coef 线性衰减 (由 EntCoefScheduleCallback 实现, SB3 原生不支持 schedule)
ENT_COEF_INIT   = 0.01
ENT_COEF_END    = 0.003
VF_COEF         = 0.5
MAX_GRAD_NORM   = 0.5


def _linear_schedule(start, end):
    """PPO schedule 签名: progress_remaining ∈ [1, 0], 训练结束时为 0
    所以 progress_done = 1 - progress_remaining
    """
    def schedule(progress_remaining):
        progress_done = 1.0 - progress_remaining
        return start + (end - start) * progress_done
    return schedule

# ── 网络架构 ──
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[128, 128],
        vf=[128, 128],
    )
)

# ── 保存与日志 (v2 独立路径) ──
LOG_DIR              = "./logs_v2"
MODEL_DIR            = "./models_v2"
CHECKPOINT_DIR       = "./models_v2/checkpoints"
FINAL_MODEL_DIR      = "./models_v2/final"
BEST_MODEL_DIR       = "./models_v2/best_model"
SAVE_FREQ            = 10_000
BEST_MODEL_SAVE_FREQ = 5000
BEST_MODEL_WINDOW    = 50     # 计算滚动平均奖励的窗口大小（episode 数）
LOG_STATE_PATH       = "./models_v2/train_log_state.json"  # TrainingLogCallback 的持久化状态

# =====================================================================
#  训练日志回调
# =====================================================================

class TrainingLogCallback(BaseCallback):
    """打印 episode 奖励统计，支持跨 resume 持久化状态"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.success_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                self.episode_rewards.append(ep_reward)

                # 判断是否成功（通过最终 reason）
                is_success = info.get("success", False) or info.get("reason") == "ball_eliminated"
                if is_success:
                    self.success_count += 1

                # 每 10 个 episode 打印一次统计
                if self.episode_count % 10 == 0:
                    recent = self.episode_rewards[-10:]
                    mean_r = np.mean(recent)
                    max_r = np.max(recent)
                    min_r = np.min(recent)
                    success_rate = self.success_count / self.episode_count * 100
                    print(
                        f"  📊 Ep {self.episode_count:5d} | "
                        f"最近10局 mean={mean_r:+.1f} "
                        f"max={max_r:+.1f} min={min_r:+.1f} | "
                        f"成功率 {success_rate:.1f}% | "
                        f"当前 R={ep_reward:+.1f} L={ep_length}"
                    )
        return True

    def save_state(self, path):
        """保存累积统计状态到磁盘，供 resume 时继承"""
        import json
        state = {
            'episode_count': self.episode_count,
            'episode_rewards': [float(r) for r in self.episode_rewards],
            'success_count': self.success_count,
        }
        with open(path, 'w') as f:
            json.dump(state, f)

    def load_state(self, path):
        """从磁盘加载上次训练的累积统计状态"""
        import json
        if not os.path.exists(path):
            print(f"  ℹ️ 未找到累积统计文件 {path}，从零开始计数")
            return False
        with open(path, 'r') as f:
            state = json.load(f)
        self.episode_count = state.get('episode_count', 0)
        self.episode_rewards = state.get('episode_rewards', [])
        self.success_count = state.get('success_count', 0)
        sr = self.success_count / self.episode_count * 100 if self.episode_count else 0
        print(f"  ✅ 已继承上次统计: Ep={self.episode_count} "
              f"成功={self.success_count} 成功率={sr:.1f}%")
        return True


# =====================================================================
#  熵系数衰减回调（v2）
# =====================================================================

class EntCoefScheduleCallback(BaseCallback):
    """
    线性衰减 model.ent_coef。

    SB3 的 PPO 不原生支持 ent_coef 是 schedule,因为 loss 公式里
    self.ent_coef * entropy_loss 要求 ent_coef 是标量。
    但是 PPO.train() 每次被调用时都会重新读取 self.ent_coef,
    所以只要在 rollout 结束、train 开始前动态改一下它,就能实现衰减。

    _on_rollout_start 在每个 rollout 周期(n_steps 步收集)前触发,
    此时根据 num_timesteps / total_timesteps 算当前进度,更新 ent_coef。

    跨 resume 友好: 根据 self.num_timesteps 而不是本次 learn() 的相对步数,
    所以 resume 时继续按全局进度衰减。
    """

    def __init__(self, start, end, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps
        self._last_printed = None

    def _compute_current(self):
        progress = min(1.0, max(0.0, self.num_timesteps / self.total_timesteps))
        return self.start + (self.end - self.start) * progress

    def _on_rollout_start(self) -> None:
        new_coef = self._compute_current()
        self.model.ent_coef = new_coef
        # 每 5% 训练进度打印一次,避免刷屏
        progress_pct = int(self.num_timesteps / self.total_timesteps * 20) * 5
        if progress_pct != self._last_printed:
            self._last_printed = progress_pct
            if self.verbose:
                print(f"  🔥 ent_coef = {new_coef:.5f} "
                      f"(进度 {progress_pct}%, step {self.num_timesteps:,})")

    def _on_step(self) -> bool:
        return True


# =====================================================================
#  Best Model 回调（覆盖式保存当前最优模型）
# =====================================================================

class BestModelCallback(BaseCallback):
    """
    基于最近 N 局滚动平均奖励，按固定 step 频率检查。
    若当前指标优于历史最佳，则覆盖保存最佳模型。

    跨 resume 的历史最佳值会从磁盘恢复，保证"最佳"是全局最佳，
    不是本次 learn() 内的最佳。
    """

    def __init__(self, save_path, check_freq=5000, window_size=50, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.window_size = window_size
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        os.makedirs(save_path, exist_ok=True)
        # 磁盘里最佳指标记录文件
        self._meta_path = os.path.join(save_path, "best_model_meta.json")
        self._load_best_from_disk()

    def _load_best_from_disk(self):
        """若磁盘上存在 meta，则恢复历史最佳指标"""
        import json
        if os.path.exists(self._meta_path):
            try:
                with open(self._meta_path, 'r') as f:
                    meta = json.load(f)
                self.best_mean_reward = meta.get('best_mean_reward', -np.inf)
                if self.verbose:
                    print(f"  ✅ 已加载历史最佳指标: mean_reward={self.best_mean_reward:+.2f}")
            except Exception as e:
                print(f"  ⚠️ 历史最佳 meta 读取失败: {e}")

    def _save_best_to_disk(self, mean_reward):
        import json
        meta = {
            'best_mean_reward': float(mean_reward),
            'window_size': self.window_size,
            'num_timesteps': int(self.num_timesteps),
        }
        with open(self._meta_path, 'w') as f:
            json.dump(meta, f)

    def _on_step(self) -> bool:
        # 累积 episode 奖励（只保留最近 window_size 个，防止无限增长）
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                if len(self.episode_rewards) > self.window_size:
                    self.episode_rewards.pop(0)

        # 按频率检查
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) >= self.window_size:
                mean_r = float(np.mean(self.episode_rewards))
                if mean_r > self.best_mean_reward:
                    self.best_mean_reward = mean_r
                    best_path = os.path.join(self.save_path, "best_model")
                    self.model.save(best_path)
                    self._save_best_to_disk(mean_r)
                    if self.verbose:
                        print(f"  🏆 新最佳模型已保存 | "
                              f"最近{self.window_size}局 mean_reward={mean_r:+.2f} "
                              f"(step={self.num_timesteps})")
        return True


# =====================================================================
#  主训练流程
# =====================================================================

def train():
    print("=" * 60)
    print("  PPO 训练 v2：半场捡网球 RL Agent (理论最优版)")
    print("=" * 60)
    print(f"  半场: {'X>0' if ACTIVE_HALF > 0 else 'X<0'}")
    print(f"  总步数: {TOTAL_TIMESTEPS:,}")
    print(f"  动作空间: {9} 维 (v2 新增后退 2 动作)")
    print(f"  学习率: {LEARNING_RATE_INIT} → {LEARNING_RATE_END} (线性衰减)")
    print(f"  熵系数: {ENT_COEF_INIT} → {ENT_COEF_END} (线性衰减, 通过 callback)")
    print(f"  n_steps: {N_STEPS} (v2 提升到 2048)")
    print(f"  网络: pi={POLICY_KWARGS['net_arch']['pi']} "
          f"vf={POLICY_KWARGS['net_arch']['vf']}")
    print("=" * 60)
    print()
    print("√ 前置检查清单：")
    print("√ CoppeliaSim 已打开并加载场景")
    print("√ 网球已生成（场上可见黄绿色小球）")
    print("√ 仿真已点 Play ▶️ 处于运行状态")
    print()
    input(" 确认以上条件后按 Enter 开始训练...")

    # 创建目录
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # ── 创建单个训练环境 ──
    # 不使用 DummyVecEnv 包装，因为会引入不必要的额外初始化
    # 也不使用 eval_env，避免双环境冲突 ZMQ 连接
    print("\n⏳ 正在连接 CoppeliaSim 并创建环境...")
    env = TennisCollectorEnv(
        render_mode=None,    # 训练时关闭可视化，加速
        active_half=ACTIVE_HALF,
    )
    env = Monitor(env)  # 包装一层 Monitor 以便统计 episode 信息
    print("✅ 环境就绪")

    # ── 创建 PPO 模型 ──
    print("\n⏳ 正在创建 PPO 模型...")
    model = PPO(
        policy="MlpPolicy", # 针对这个网络结构（非 CNN），用 CPU 训练通常会更快、更高效。
        env=env,
        # v2: 学习率和熵系数都用线性衰减 schedule
        learning_rate=_linear_schedule(LEARNING_RATE_INIT, LEARNING_RATE_END),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF_INIT,   # 初始值,EntCoefScheduleCallback 会动态更新
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=LOG_DIR,
        verbose=1,
        seed=42,
        device='cpu'
    )
    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"✅ PPO 模型创建完成 | 参数量: {n_params:,}")

    # ── 回调 ──
    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_tennis",
        verbose=1,
    )
    best_model_cb = BestModelCallback(
        save_path=BEST_MODEL_DIR,
        check_freq=BEST_MODEL_SAVE_FREQ,
        window_size=BEST_MODEL_WINDOW,
        verbose=1,
    )
    log_cb = TrainingLogCallback()
    ent_sched_cb = EntCoefScheduleCallback(
        start=ENT_COEF_INIT,
        end=ENT_COEF_END,
        total_timesteps=TOTAL_TIMESTEPS,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_cb, best_model_cb, log_cb, ent_sched_cb])

    # ── 开始训练 ──
    print("\n🚀 开始训练...")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")
    print(f"按 Ctrl+C 可随时中断并保存当前模型\n")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=False,   # progress_bar 可能与某些终端冲突
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断训练")
    except Exception as e:
        print(f"\n\n❌ 训练异常: {e}")
        import traceback
        traceback.print_exc()

    # ── 保存最终模型 ──
    final_path = os.path.join(FINAL_MODEL_DIR, "ppo_tennis_final")
    model.save(final_path)
    print(f"\n💾 最终模型已保存: {final_path}.zip")

    # ── 保存日志回调的累积状态（供 resume 继承）──
    log_cb.save_state(LOG_STATE_PATH)
    print(f"💾 训练统计已保存: {LOG_STATE_PATH}")

    # ── 训练统计 ──
    if log_cb.episode_count > 0:
        success_rate = log_cb.success_count / log_cb.episode_count * 100
        mean_reward = np.mean(log_cb.episode_rewards) if log_cb.episode_rewards else 0
        print(f"\n📈 训练统计:")
        print(f"总 Episode 数: {log_cb.episode_count}")
        print(f"成功消除: {log_cb.success_count} 次")
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均奖励: {mean_reward:+.1f}")

    env.close()
    print("\n✅ 训练流程结束")


# =====================================================================
#  单独评估（训练完后运行）
# =====================================================================

def evaluate(model_path, n_episodes=10):
    """评估已训练模型"""
    print(f"\n📂 加载模型: {model_path}")
    env = TennisCollectorEnv(render_mode="human", active_half=ACTIVE_HALF)
    model = PPO.load(model_path)

    total_success = 0
    total_reward = 0

    for i in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        if info.get('success', False):
            total_success += 1

        total_reward += ep_reward
        print(f"Ep {i+1}: R={ep_reward:+.1f} | "
              f"reason={info.get('reason', '?')} | "
              f"成功={info.get('success', False)}")

    print(f"\n🏁 评估完成")
    print(f"成功率: {total_success}/{n_episodes} = {total_success/n_episodes*100:.1f}%")
    print(f"平均奖励: {total_reward/n_episodes:+.1f}")

    env.close()


# =====================================================================
#  继续训练（从检查点恢复）
# =====================================================================

def resume_training(checkpoint_path, additional_timesteps=500_000):
    """从已保存的 v2 检查点继续训练。

    注意: v2 动作空间为 9 维, **不能**加载 train.py (v1, 7 维) 的模型。
    如果尝试加载 v1 模型会在兼容性检查时明确报错,提示从头训练。
    """
    print(f"📂 加载模型: {checkpoint_path}")
    env = TennisCollectorEnv(render_mode=None, active_half=ACTIVE_HALF)
    env = Monitor(env)

    model = PPO.load(checkpoint_path, env=env, device='cpu')

    # v2 兼容性检查: 动作空间维度必须一致
    expected_n = env.action_space.n
    loaded_n = model.action_space.n if hasattr(model.action_space, 'n') else None
    if loaded_n != expected_n:
        env.close()
        raise ValueError(
            f"模型动作空间 ({loaded_n}) 与当前 env 动作空间 ({expected_n}) 不匹配。"
            f"v2 动作空间为 9 维,v1 是 7 维,不能互相 resume。请改用 train() 从头训。"
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_tennis",
        verbose=1,
    )
    best_model_cb = BestModelCallback(
        save_path=BEST_MODEL_DIR,
        check_freq=BEST_MODEL_SAVE_FREQ,
        window_size=BEST_MODEL_WINDOW,
        verbose=1,
    )
    log_cb = TrainingLogCallback()
    # 继承上次训练的累积统计（episode_count / success_count / episode_rewards）
    log_cb.load_state(LOG_STATE_PATH)
    # ent_coef 衰减 callback 用全局 TOTAL_TIMESTEPS 作分母,
    # resume 时 num_timesteps 是累计值,能自然接上之前的衰减进度
    ent_sched_cb = EntCoefScheduleCallback(
        start=ENT_COEF_INIT,
        end=ENT_COEF_END,
        total_timesteps=TOTAL_TIMESTEPS,
        verbose=1,
    )

    print(f"🚀 继续训练 {additional_timesteps:,} 步...")
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=CallbackList([checkpoint_cb, best_model_cb, log_cb, ent_sched_cb]),
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")

    final_path = os.path.join(FINAL_MODEL_DIR, "ppo_tennis_resumed")
    model.save(final_path)
    print(f"\n💾 最终模型已保存: {final_path}.zip")

    # 保存日志回调的累积状态（供下次 resume 继续继承）
    log_cb.save_state(LOG_STATE_PATH)
    print(f"💾 训练统计已保存: {LOG_STATE_PATH}")

    # 打印训练统计（和普通训练一样）
    if log_cb.episode_count > 0:
        success_rate = log_cb.success_count / log_cb.episode_count * 100
        mean_reward = np.mean(log_cb.episode_rewards) if log_cb.episode_rewards else 0
        print(f"\n📈 训练统计:")
        print(f"总 Episode 数: {log_cb.episode_count}")
        print(f"成功消除: {log_cb.success_count} 次")
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均奖励: {mean_reward:+.1f}")

    env.close()
    print("\n✅ 继续训练流程结束")

# =====================================================================
#  入口
# =====================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "eval":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "./models/final/ppo_tennis_final.zip"
            evaluate(model_path, n_episodes=10)
        elif cmd == "resume":
            ckpt = sys.argv[2] if len(sys.argv) > 2 else "./models/final/ppo_tennis_final.zip"
            resume_training(ckpt)
        else:
            print(f"未知命令: {cmd}")
            print("用法:")
            print("python train_ppo.py              # 开始训练")
            print("python train_ppo.py eval <模型>   # 评估模型")
            print("python train_ppo.py resume <模型> # 继续训练")
    else:
        train()
