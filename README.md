# 🎾 TennisBallsCollector — 基于纯视觉强化学习的 YouBot 网球收集机器人

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="CoppeliaSim" src="https://img.shields.io/badge/CoppeliaSim-4.10.0-2E7D32">
  <img alt="Stable-Baselines3" src="https://img.shields.io/badge/Stable--Baselines3-PPO-EF6C00">
  <img alt="Gymnasium" src="https://img.shields.io/badge/Gymnasium-1.2.3-1565C0">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-success">
</p>

> **简介**：在 CoppeliaSim 真实比例（1:1）网球场内，用一台仅装备**单目摄像头(Vision Sensor)**的 KUKA YouBot 移动机器人，通过 PPO 强化学习自主完成"巡场 → 发现 → 接近 → 收集"全流程，最终成功率稳定在 **88%~92%**。

## 📖 项目简介

本项目实现了一个 **完整工程化** 的纯视觉驱动强化学习捡网球系统，核心特点：

- **纯视觉感知**：YouBot 仅依赖一个前向 RGB visionSensor（1024×1024），**不依赖深度信息**，仅依靠 HSV 颜色分割 + 轮廓像素面积估计距离与方位。
- **半场专注式 RL**：将"全场捡球"这一巨型任务拆解为"半场内单球收集"的 Gym 环境（一个 episode = 在当前半场内消除一个球），使 PPO 能在合理时间内收敛。
- **规则 + RL 混合架构**：底层 RL Agent 负责视觉闭环捡球，中层规则代码负责 S 形半场巡视，顶层规则代码负责绕网切换半场，各层职责分明。
- **稠密奖励工程**：精心设计的多分支奖励函数（视觉引导 + 视野新鲜度 + 边界/球网梯度 + 摇头检测 + 多终止判据），有效抑制原地摇头、贴网、卡边界等局部最优行为。
- **生产级训练设施**：跨 resume 持久化的统计回调、滚动窗口最佳模型保存、学习率与熵系数线性衰减、动态网球重生成（避免位置过拟合）、Checkpoint 规范化。
- **从全局坐标作弊版到纯视觉版**：项目历经 V0~V4 多代演进，每一代的核心难点（万向锁、贴网震荡、视觉噪声、resume 统计归零、绕网撞椅子）都有详细的工程解法记录。

整体目标是**完整验证纯视觉 RL 在复杂仿真环境中的可行性**，并积累一套可复用的 Gym 环境/奖励工程/部署流程模板。

## ✨ 主要特性

| 模块 | 关键能力                                                                 |
|------|----------------------------------------------------------------------|
| 🎮 仿真环境 | CoppeliaSim 4.10.0 + ZMQ stepping 同步模式，通过 `sim.step()` 严格驱动物理与 agent |
| 👁️ 感知 | OpenCV HSV → 形态学 → 轮廓 → 面积排序 → 像素面积距离反算 + 半场严格过滤                     |
| 🧠 RL 算法 | Stable-Baselines3 PPO（MlpPolicy，pi/vf=[128,128]），CPU 训练即可            |
| 🎯 状态空间 | 10 维语义特征 × 3 帧堆叠 = 30 维（避免直接喂图像，CPU 训练成本极低）                          |
| 🕹️ 动作空间 | V2 共 9 个离散动作，含 2 个后退动作解决"背后球只能摇头"的局部最优                               |
| 🏆 奖励函数 | 稀疏 +100 / 稠密引导 / 视野新鲜度 / 边界球网梯度 / 摇头 / 终止性 -10                       |
| 🚀 训练设施 | 跨 resume 状态持久化、自动最佳模型保存、学习率/熵系数线性衰减                                  |
| 🤖 部署系统 | RL + 规则混合调度，绕网两段式路径 + S 形 6 点巡视 + 卡顿兜底                               |

## 🏛️ 系统架构

### 整体通信框架

```
CoppeliaSim 仿真环境（4.10.0）
        ↕ ZMQ Remote API
Python RL 训练 / 部署 (PPO 算法)
```

### 三层分层架构（部署时）

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 320" width="720" height="320">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#444"/>
    </marker>
  </defs>
  <rect x="20" y="20" width="680" height="80" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="40" y="48" font-family="Consolas,monospace" font-size="16" font-weight="bold" fill="#0d47a1">顶层调度（规则代码）</text>
  <text x="40" y="78" font-family="Consolas,monospace" font-size="13" fill="#0d47a1">半场切换 + 绕网（避开网柱 |Y|=6.40 与椅子 |Y|=7.72，走廊 |Y|=7.05）</text>

  <rect x="20" y="120" width="680" height="80" rx="8" fill="#fff3e0" stroke="#e65100" stroke-width="2"/>
  <text x="40" y="148" font-family="Consolas,monospace" font-size="16" font-weight="bold" fill="#bf360c">中层巡视（规则代码）</text>
  <text x="40" y="178" font-family="Consolas,monospace" font-size="13" fill="#bf360c">S 形 6 路径点扫描，端点就近切入 + 严格 est_bx 阈值 + 连续 3 次确认</text>

  <rect x="20" y="220" width="680" height="80" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="40" y="248" font-family="Consolas,monospace" font-size="16" font-weight="bold" fill="#1b5e20">底层 RL Agent（Gymnasium 环境 TennisCollectorEnv）</text>
  <text x="40" y="278" font-family="Consolas,monospace" font-size="13" fill="#1b5e20">PPO 策略 · 30 维语义观测 · 9 动作 · 单 episode = 在当前半场捡 1 个球</text>

  <line x1="360" y1="100" x2="360" y2="120" stroke="#444" stroke-width="2" marker-end="url(#ar)"/>
  <line x1="360" y1="200" x2="360" y2="220" stroke="#444" stroke-width="2" marker-end="url(#ar)"/>
</svg>

只有"在当前半场内找到并消除一个球"这一最难的视觉感知 + 闭环运动控制部分交给 PPO 学习；
全场调度、绕网、巡视等几何明确的部分由规则代码处理，避免无谓增加 RL 学习负担。

## 📁 项目结构

```
Tennis_Collector/
├── scene/
│   ├── scene.ttt                          -- 可直接使用的场景文件
│   ├── tennis_scene_init.lua              -- 创建场景脚本 V1.0
│   ├── tennis_scene_latest.lua            -- 创建场景脚本 V2.0
│   └── Tennis_Generate.lua                -- 随机位置网球生成脚本
│
├── tennis_rl_env.py                       -- RL 环境 V1（7 动作）
├── tennis_rl_env2.py                      -- RL 环境 V2（9 动作 + 视野新鲜度）
│
├── train.py                               -- PPO 训练脚本 V1
├── train2.py                              -- PPO 训练脚本 V2（学习率/熵衰减）
│
├── Deploy_collector.py                    -- 部署脚本 V1
├── Deploy_Collector2.py                   -- 部署脚本 V2（绕网 + 半场巡视）
│
├── TennisGlobalLocating_elimate_with_dist.py  -- 初代全局坐标作弊版
├── Tennis_Spawner_Script_test.py          -- 网球生成lua脚本测试
├── async_saver.py                         -- 异步模型保存工具
│
├── logs/  logs_v2/                        -- TensorBoard 训练日志
├── models/  models_v2/                    -- 训练缓存与最佳模型
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

| 依赖 | 版本要求 |
|------|----------|
| CoppeliaSim | 4.10.0+（[官方下载](https://www.coppeliarobotics.com/)） |
| Python | 3.11.15（开发测试版本） |
| 操作系统 | Windows / Linux |

```bash
# 克隆仓库
git clone https://github.com/uMemory/CollectTennisBalls_RL_CoppeliaSim.git
cd CollectTennisBalls_RL_CoppeliaSim

# (推荐) 创建独立环境
conda create -n tennis python=3.11.15 -y
conda activate tennis

# 安装依赖
pip install -r requirements.txt

# 如需 GPU 版 PyTorch (CPU 训练已足够，可跳过)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 2. 启动 CoppeliaSim 场景

1. 打开 CoppeliaSim，加载 `scene/scene.ttt`
2. **挂载网球生成脚本（关键步骤，否则训练中无法自动重生成网球）**：
   - 在场景层级中找到 **收集箱（`Bin_Entry` / `Bin_Base`）** 对象
   - 右键 → `Add` → `Associated child script` → 选择 **`Customization`** 类型 + **`Non-threaded`** 模式 + **`Lua`** 语言
   - 该脚本必须作为收集箱的 **子 subframe**（关联子脚本），脚本内容粘贴 `scene/Tennis_Generate.lua`
   - 这样 Python 端才能通过 `sim.callScriptFunction('spawnBalls', ...)` 在训练运行时远程调用生成函数，可通过运行Tennis_Spawner_Script_test.py测试是否可用。
   - **仓库自带 `scene.ttt` 已预挂载该脚本，直接使用可跳过本步**
3. 点 **Play ▶** 启动仿真（仿真处于运行中即可，stepping 模式由 Python 端自动接管）

> 💡 **注意必须是 Customization + Non-threaded + Lua**：
> - **Customization Script**：随场景持久保存，每次 Play 都会自动加载，且能被 `sim.callScriptFunction` 跨进程调用（普通 child script 重启后会丢失）
> - **Non-threaded**：与仿真主循环同步执行，避免多线程下 `sim.createPrimitiveShape` 等 API 时序竞争导致网球生成失败
> - **Lua**：CoppeliaSim 内置原生支持，`sim.*` API 调用零开销；Python 端通过 ZMQ 异步触发，性能远优于 Python 脚本回调

### 3. 启动训练

```bash
python train2.py
```

> ⚠️ **首次训练建议**：从头训练约 300k~500k 步，单环境 CPU 速度约 4~7 fps，预估 24~40 小时。可中途 `Ctrl+C` 安全保存并 resume。
>
> 训练过程中可在另一终端运行 `tensorboard --logdir ./logs_v2 --port 6006` 监控曲线。
> 评估、resume、部署等其他命令见文末 [🗒️ 常用命令速查](#-常用命令速查)。

## 📋 项目概述（技术摘要）

- **仿真平台**：CoppeliaSim 4.10.0 + ZMQ Remote API（stepping 同步模式）
- **RL 算法**：Stable-Baselines3 PPO（MlpPolicy）
- **感知方式**：单目 visionSensor + HSV 颜色分割（无深度信息，仅靠像素面积估算距离）
- **最终效果**：成功率稳定在 **88%~92%**，支持长时间 resume 训练，具备较好的鲁棒性

项目从**全局坐标作弊版**开始，经过多次重大迭代，最终形成一套**工程化程度较高**的纯视觉 RL 解决方案。

---

## 🧠 核心环境设计（TennisCollectorEnv）

环境实现遵循 Gymnasium API，单环境直连 CoppeliaSim（不使用 VecEnv，避免多 ZMQ 连接冲突），仿真使用 stepping 同步模式保证 agent 与物理引擎严格对齐。

### 1. 状态空间（Observation Space）

观测空间为 **10 维单帧语义特征 × 3 帧堆叠 = 30 维**连续向量（`Box(low=-1, high=1, shape=(30,))`），用 3 帧堆叠让网络感知到运动趋势（如球在画面中变大 = 正在靠近）。

不直接喂图像（CNN 训练慢、连续 ZMQ 取图开销大），而是先通过 OpenCV 提取语义特征，使 PPO 网络可以使用轻量 MLP（`pi=[128,128]`、`vf=[128,128]`）。

**10 维单帧特征**：

| 维度 | 名称              | 含义                                               | 取值范围   |
|------|-------------------|----------------------------------------------------|------------|
| 0    | `ball_detected`   | 活跃半场内是否检测到网球                            | {0, 1}     |
| 1    | `ball_angle`      | 最近网球在图像中的归一化水平偏角                    | [-1, 1]    |
| 2    | `ball_size`       | 像素面积归一化（距离的反向代理）                    | [0, 1]     |
| 3    | `ball_count`      | 活跃半场内可见网球数量（按 `NORM_COUNT=6` 归一化） | [0, 1]     |
| 4    | `ball_reachable`  | 估计球世界坐标是否在己方半场（宽松判据，容忍 0.5m）| {0, 1}     |
| 5    | `norm_rx`         | YouBot X 坐标归一化（`/HALF_COURT_X_MAX≈18.3`）   | [-1, 1]    |
| 6    | `norm_ry`         | YouBot Y 坐标归一化（`/HALF_COURT_Y_MAX≈9.1`）    | [-1, 1]    |
| 7    | `norm_yaw`        | YouBot 航向角归一化（`/π`）                        | [-1, 1]    |
| 8    | `norm_net`        | 到球网（X=0）的归一化距离                          | [0, 1]     |
| 9    | `norm_bound`      | 到场地最近边界的归一化距离（按 3.0 m 截断）         | [0, 1]     |

**关键点**：
- `ball_angle` 由轮廓质心 `cx` 直接通过 `(cx - W/2)/(W/2)` 得到，不受图像垂直裁剪影响
- `ball_size` = 轮廓像素面积 / (614 × 1024 × 0.05)，作为距离代理
- 半场过滤使用**严格判据** (`est_bx > 0`)，避免对面半场球污染观测
- `ball_reachable` 使用**宽松判据** (`est_bx > -0.5`) 给可达标志，避免误杀网边球

### 2. 动作空间（Action Space）

离散动作空间，每个动作连续执行 `ACTION_REPEAT=4` 个仿真步，给电机充分时间产生位移，避免单步动作信号过短被物理引擎吞掉。

底层运动学采用麦克纳姆轮"差速 + 推力"组合，每个动作映射为四个轮速：

```
v_fl = forward + turn       v_fr = forward - turn
v_rl = forward + turn       v_rr = forward - turn
```

#### 环境 V1（ `tennis_rl_env.py`）：7 个离散动作

| ID | 名称              | forward | turn | 说明           |
|----|-------------------|---------|------|----------------|
| 0  | FORWARD           | 5~6     | 0.0  | 直行前进       |
| 1  | FORWARD_LEFT      | 4.0     | +1.5 | 左前弧线前进   |
| 2  | FORWARD_RIGHT     | 4.0     | -1.5 | 右前弧线前进   |
| 3  | TURN_LEFT         | 0.0     | +2.5 | 小角度左原地转 |
| 4  | TURN_RIGHT        | 0.0     | -2.5 | 小角度右原地转 |
| 5  | TURN_LEFT_LARGE   | 0.0     | +5.0 | 大角度左原地转 |
| 6  | TURN_RIGHT_LARGE  | 0.0     | -5.0 | 大角度右原地转 |

#### 环境 V2（`tennis_rl_env2.py`）：9 个离散动作

V2 在 V1 基础上**新增 2 个后退动作**，解决"球在背后时只能摇头不能后退"的局部最优：

| ID | 名称              | forward | turn | 说明           |
|----|-------------------|---------|------|----------------|
| 0~6| 同 V1             |         |      |                |
| 7  | BACKWARD          | -3.0    | 0.0  | 后退（V2 新增）|
| 8  | BACKWARD_TURN     | -3.0    | +1.5 | 后退 + 小转向  |

> **注意**：V2 的 9 维动作空间无法加载 V1 的 7 维模型 resume，需从头训练。

### 3. 奖励函数（Reward Shaping）

奖励函数是本项目最关键的工程难点。采用 **稠密引导 + 稀疏目标 + 多重惩罚** 三段式设计：

#### 单步奖励判定流程

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 540" width="880" height="540">
  <defs>
    <marker id="rar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" fill="#444"/>
    </marker>
  </defs>

  <!-- 入口：本步动作执行后 -->
  <rect x="340" y="10" width="200" height="40" rx="20" fill="#3949ab" stroke="#1a237e" stroke-width="2"/>
  <text x="440" y="35" font-family="Consolas,monospace" font-size="13" fill="#fff" text-anchor="middle">step() 调用</text>

  <!-- 判定 1：是否消除球 -->
  <polygon points="440,70 600,110 440,150 280,110" fill="#ffe082" stroke="#ff8f00" stroke-width="2"/>
  <text x="440" y="108" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#5d4037" text-anchor="middle">距离 &lt; 0.40 m？</text>
  <text x="440" y="125" font-family="Consolas,monospace" font-size="11" fill="#5d4037" text-anchor="middle">(消除一个球)</text>
  <line x1="440" y1="50" x2="440" y2="70" stroke="#444" stroke-width="2" marker-end="url(#rar)"/>

  <!-- yes -> +100 终止 -->
  <rect x="640" y="90" width="220" height="40" rx="6" fill="#43a047" stroke="#1b5e20" stroke-width="2"/>
  <text x="750" y="115" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#fff" text-anchor="middle">return +100  (terminated)</text>
  <line x1="600" y1="110" x2="640" y2="110" stroke="#444" stroke-width="2" marker-end="url(#rar)"/>
  <text x="610" y="105" font-family="Consolas,monospace" font-size="11" fill="#2e7d32">YES</text>

  <!-- 否 -> 基础项 -->
  <line x1="440" y1="150" x2="440" y2="180" stroke="#444" stroke-width="2" marker-end="url(#rar)"/>
  <text x="450" y="170" font-family="Consolas,monospace" font-size="11" fill="#666">NO</text>

  <rect x="290" y="180" width="300" height="55" rx="6" fill="#eceff1" stroke="#546e7a" stroke-width="1.5"/>
  <text x="440" y="202" font-family="Consolas,monospace" font-size="12" fill="#263238" text-anchor="middle">reward = -0.1  (基础时间惩罚)</text>
  <text x="440" y="222" font-family="Consolas,monospace" font-size="12" fill="#263238" text-anchor="middle">if 位移 &lt; 0.02m: reward -= 1.0</text>

  <!-- 视觉引导分支 -->
  <line x1="440" y1="235" x2="440" y2="260" stroke="#444" stroke-width="2" marker-end="url(#rar)"/>
  <rect x="290" y="260" width="300" height="35" rx="6" fill="#bbdefb" stroke="#1565c0" stroke-width="2"/>
  <text x="440" y="283" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#0d47a1" text-anchor="middle">视觉状态？</text>

  <!-- 三个分支 -->
  <line x1="340" y1="295" x2="120" y2="335" stroke="#444" stroke-width="1.5" marker-end="url(#rar)"/>
  <line x1="440" y1="295" x2="440" y2="335" stroke="#444" stroke-width="1.5" marker-end="url(#rar)"/>
  <line x1="540" y1="295" x2="760" y2="335" stroke="#444" stroke-width="1.5" marker-end="url(#rar)"/>

  <!-- 分支 A：活跃半场内有球 -->
  <rect x="20" y="335" width="220" height="100" rx="6" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2"/>
  <text x="130" y="355" font-family="Consolas,monospace" font-size="12" font-weight="bold" fill="#1b5e20" text-anchor="middle">(A) 看到活跃半场球</text>
  <text x="30" y="375" font-family="Consolas,monospace" font-size="11" fill="#1b5e20">+ 0.5  (移动奖励)</text>
  <text x="30" y="392" font-family="Consolas,monospace" font-size="11" fill="#1b5e20">+ 1.0·(1 - |angle|)  (对齐)</text>
  <text x="30" y="409" font-family="Consolas,monospace" font-size="11" fill="#1b5e20">+ 5.0·Δsize  (接近)</text>
  <text x="30" y="426" font-family="Consolas,monospace" font-size="11" fill="#1b5e20">no_ball_steps = 0</text>

  <!-- 分支 B：只有对面半场球 -->
  <rect x="330" y="335" width="220" height="100" rx="6" fill="#ffe0b2" stroke="#e65100" stroke-width="2"/>
  <text x="440" y="355" font-family="Consolas,monospace" font-size="12" font-weight="bold" fill="#bf360c" text-anchor="middle">(B) 只有对面半场球</text>
  <text x="340" y="378" font-family="Consolas,monospace" font-size="11" fill="#bf360c">- 0.5  (注意错误目标)</text>
  <text x="340" y="398" font-family="Consolas,monospace" font-size="11" fill="#bf360c">no_ball_steps += 1</text>

  <!-- 分支 C：完全看不到球 -->
  <rect x="640" y="335" width="220" height="100" rx="6" fill="#ffcdd2" stroke="#b71c1c" stroke-width="2"/>
  <text x="750" y="355" font-family="Consolas,monospace" font-size="12" font-weight="bold" fill="#b71c1c" text-anchor="middle">(C) 完全看不到球</text>
  <text x="650" y="378" font-family="Consolas,monospace" font-size="11" fill="#b71c1c">no_ball_steps += 1</text>
  <text x="650" y="398" font-family="Consolas,monospace" font-size="11" fill="#b71c1c">- min(0.3 + 0.01·n, 1.5)</text>
  <text x="650" y="416" font-family="Consolas,monospace" font-size="10" fill="#b71c1c">(V2 视野新鲜度线性递增)</text>

  <!-- 汇总 -->
  <line x1="130" y1="435" x2="430" y2="465" stroke="#444" stroke-width="1.5" marker-end="url(#rar)"/>
  <line x1="440" y1="435" x2="440" y2="465" stroke="#444" stroke-width="1.5" marker-end="url(#rar)"/>
  <line x1="750" y1="435" x2="450" y2="465" stroke="#444" stroke-width="1.5" marker-end="url(#rar)"/>

  <rect x="180" y="465" width="520" height="60" rx="6" fill="#e1f5fe" stroke="#0277bd" stroke-width="1.5"/>
  <text x="440" y="486" font-family="Consolas,monospace" font-size="11" fill="#01579b" text-anchor="middle">+ 边界惩罚 -3.0·(1.0 - dist_b)  (dist_b &lt; 1.0)</text>
  <text x="440" y="504" font-family="Consolas,monospace" font-size="11" fill="#01579b" text-anchor="middle">+ 球网惩罚 -4.0·(0.8 - dist_n)  (dist_n &lt; 0.8)</text>
  <text x="440" y="520" font-family="Consolas,monospace" font-size="11" fill="#01579b" text-anchor="middle">+ 摇头惩罚 -1.0  (yaw 变化 &gt; 0.3 且 位移 &lt; 0.02)</text>
</svg>

> 终止性大惩罚（越网 / 卡边界 / 卡网）在判定流程之外另行触发，每次额外 `-10`。


#### (a) 稀疏大奖励 — 任务核心目标

| 事件                              | 奖励         |
|-----------------------------------|--------------|
| 成功消除网球（距离 < 0.40 m）     | **+100**     |

#### (b) 每步基础项

| 项目                              | 数值         | 触发条件                              |
|-----------------------------------|--------------|---------------------------------------|
| 时间惩罚                          | `-0.1`       | 每步固定                              |
| 静止惩罚                          | `-1.0`       | 单步位移 < 0.02 m                     |

#### (c) 视觉引导奖励 — 三种视野状态分情况处理

| 视野状态                                                    | 奖励组成                                           |
|-------------------------------------------------------------|----------------------------------------------------|
| (i) 看到活跃半场内的球（`ball_in_half=True`）              | `+0.5` 移动奖励 `+ 1.0·(1-|angle|)` 角度对齐 `+ 5.0·Δsize` 接近增益 |
| (ii) 视野里只有对面半场的球（`any_visible && !ball_in_half`）| `-0.5` 强惩罚（比完全没看到更糟，避免 agent 注意错误目标） |
| (iii) 完全看不到球                                          | **V1**: `-0.3` 固定；**V2**: `-min(0.3 + 0.01·n, 1.5)` 视野新鲜度线性递增 |

> **V2 视野新鲜度（freshness penalty）**：连续 `n` 步看不到球，单步惩罚从 `-0.3` 线性增加到 `-1.5` 封顶。
> 设计目的：阻止 agent 通过"原地慢慢摇头"把单步惩罚摊薄到接近 0 的局部最优，强制其要么找到球要么主动探索。

#### (d) 边界 / 球网梯度惩罚

| 触发                              | 奖励公式                                  |
|-----------------------------------|-------------------------------------------|
| 距离边界 < 1.0 m                  | `-3.0 × (1.0 - dist_to_boundary)`        |
| 距离球网 < 0.8 m                  | `-4.0 × (0.8 - net_distance)`            |

梯度惩罚比硬性禁止更平滑，让 agent 学到"远离边界更安全"的连续偏好。

#### (e) 摇头惩罚（基于 frame stack 的滑动检测）

```
最近 3 帧 |yaw 变化| > 0.3 rad  且  位移 < 0.02 m  →  -1.0
```

#### (f) 终止性大惩罚

| 终止原因                          | 奖励      | 步数判据                |
|-----------------------------------|-----------|-------------------------|
| 越过球网（`crossed_net`）         | `-10`     | `|rx| 跨过 -0.3`        |
| 卡在边界                           | `-10`     | `dist_to_boundary < 0.3 且 单步位移 < 0.05` 连续 30 步 |
| 卡在网边                           | `-10`     | `net_distance < 0.3 且 单步位移 < 0.05` 连续 30 步 |
| 超时（无负值）                     | 0         | `step_count ≥ 500`     |

> 卡顿判据使用**双重条件**（位置接近 + 位移小），避免"追球过程中途经网边"被误判终止。

### 4. 终止与重置（Episode Lifecycle）

**Episode 终止条件**：
- 成功消除一个球（`terminated=True, success=True`）
- 越过球网 / 卡边界 / 卡网（`terminated=True, success=False`）
- 步数达到 500（`truncated=True`）

**Reset 流程**：
1. 检查活跃半场内球数；为 0 则切换到对面半场；全场都没球则调用 Lua `spawnBalls()` 重生成 12 个
2. YouBot 随机放置在活跃半场内（`x ∈ [2, X_MAX-1]`，`y ∈ [-Y_MAX+1, Y_MAX-1]`）
3. **姿态重置**：先恢复初始正常姿态 `_default_ori`（保证水平），再用四元数绕世界 Z 轴乘随机航向角 —— 彻底规避欧拉角重置导致的万向锁/翻车
4. 调用 `sim.resetDynamicObject()` 清除残余速度
5. 等 15 个仿真步让物理稳定，再用 3 个相同初始观测填满 frame buffer

**soft_reset（部署专用）**：不挪动 YouBot，仅清空 episode 状态与 frame buffer，供绕网/巡视后衔接 RL 捡球。

### 5. 视觉感知流水线

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 760" width="880" height="760">

  <rect x="240" y="20" width="400" height="60" rx="8" fill="#3949ab" stroke="#1a237e" stroke-width="2"/>
  <text x="440" y="46" font-family="Consolas,monospace" font-size="14" font-weight="bold" fill="#fff" text-anchor="middle">① visionSensor 取图</text>
  <text x="440" y="68" font-family="Consolas,monospace" font-size="12" fill="#e8eaf6" text-anchor="middle">RGB 图像  1024 × 1024</text>

  <line x1="440" y1="80" x2="440" y2="105" stroke="#444" stroke-width="2"/>
  <polygon points="436,101 444,101 440,109" fill="#444"/>

  <rect x="200" y="110" width="480" height="74" rx="8" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/>
  <text x="440" y="134" font-family="Consolas,monospace" font-size="14" font-weight="bold" fill="#0d47a1" text-anchor="middle">② 裁剪上方 40%</text>
  <text x="440" y="155" font-family="Consolas,monospace" font-size="11" fill="#0d47a1" text-anchor="middle">输出 614 × 1024（保留下方 60%）</text>
  <text x="440" y="173" font-family="Consolas,monospace" font-size="11" fill="#0d47a1" text-anchor="middle">理由：远处球集中在画面中下部，上方为天空 / 远景噪声</text>

  <line x1="440" y1="184" x2="440" y2="209" stroke="#444" stroke-width="2"/>
  <polygon points="436,205 444,205 440,213" fill="#444"/>

  <rect x="200" y="214" width="480" height="74" rx="8" fill="#e8f5e9" stroke="#2e7d32" stroke-width="2"/>
  <text x="440" y="238" font-family="Consolas,monospace" font-size="14" font-weight="bold" fill="#1b5e20" text-anchor="middle">③ HSV 阈值分割</text>
  <text x="440" y="259" font-family="Consolas,monospace" font-size="11" fill="#1b5e20" text-anchor="middle">H ∈ [25, 45]   S ∈ [80, 255]   V ∈ [80, 255]</text>
  <text x="440" y="277" font-family="Consolas,monospace" font-size="11" fill="#1b5e20" text-anchor="middle">输出二值掩码 (mask)，命中网球的荧光黄绿色</text>

  <line x1="440" y1="288" x2="440" y2="313" stroke="#444" stroke-width="2"/>
  <polygon points="436,309 444,309 440,317" fill="#444"/>

  <rect x="200" y="318" width="480" height="74" rx="8" fill="#fff3e0" stroke="#e65100" stroke-width="2"/>
  <text x="440" y="342" font-family="Consolas,monospace" font-size="14" font-weight="bold" fill="#bf360c" text-anchor="middle">④ 形态学去毛刺</text>
  <text x="440" y="363" font-family="Consolas,monospace" font-size="11" fill="#bf360c" text-anchor="middle">OPEN（5×5 椭圆核）→ 去散点 + CLOSE → 填充小空洞</text>
  <text x="440" y="381" font-family="Consolas,monospace" font-size="11" fill="#bf360c" text-anchor="middle">得到平滑的网球区域块</text>

  <line x1="440" y1="392" x2="440" y2="417" stroke="#444" stroke-width="2"/>
  <polygon points="436,413 444,413 440,421" fill="#444"/>

  <rect x="200" y="422" width="480" height="74" rx="8" fill="#f3e5f5" stroke="#6a1b9a" stroke-width="2"/>
  <text x="440" y="446" font-family="Consolas,monospace" font-size="14" font-weight="bold" fill="#4a148c" text-anchor="middle">⑤ 轮廓检测 + 面积过滤</text>
  <text x="440" y="467" font-family="Consolas,monospace" font-size="11" fill="#4a148c" text-anchor="middle">findContours(RETR_EXTERNAL)，丢弃 area &lt; 30 像素的小斑点</text>
  <text x="440" y="485" font-family="Consolas,monospace" font-size="11" fill="#4a148c" text-anchor="middle">按面积降序排序 → 最大 = 最近的球</text>

  <line x1="440" y1="496" x2="440" y2="521" stroke="#444" stroke-width="2"/>
  <polygon points="436,517 444,517 440,525" fill="#444"/>

  <rect x="80" y="526" width="720" height="120" rx="8" fill="#fce4ec" stroke="#ad1457" stroke-width="2"/>
  <text x="440" y="550" font-family="Consolas,monospace" font-size="14" font-weight="bold" fill="#880e4f" text-anchor="middle">⑥ 估计世界坐标 (est_bx, est_by, est_dist)</text>
  <text x="440" y="572" font-family="Consolas,monospace" font-size="11" fill="#880e4f" text-anchor="middle">ball_angle_rad = atan(angle_norm × tan(FOV/2))     ← 精确反三角，非小角度近似</text>
  <text x="440" y="592" font-family="Consolas,monospace" font-size="11" fill="#880e4f" text-anchor="middle">est_dist = (R_ball × W) / (2 × pixel_radius × tan(FOV/2))</text>
  <text x="440" y="612" font-family="Consolas,monospace" font-size="11" fill="#880e4f" text-anchor="middle">est_bx = robot_x + est_dist × cos(robot_yaw + ball_angle_rad)</text>
  <text x="440" y="632" font-family="Consolas,monospace" font-size="11" fill="#880e4f" text-anchor="middle">est_by = robot_y + est_dist × sin(robot_yaw + ball_angle_rad)</text>

  <line x1="440" y1="646" x2="440" y2="671" stroke="#444" stroke-width="2"/>
  <polygon points="436,667 444,667 440,675" fill="#444"/>

  <line x1="440" y1="671" x2="220" y2="690" stroke="#444" stroke-width="1.5"/>
  <polygon points="222,686 222,694 213,690" fill="#444"/>
  <line x1="440" y1="671" x2="660" y2="690" stroke="#444" stroke-width="1.5"/>
  <polygon points="658,686 658,694 667,690" fill="#444"/>

  <rect x="40" y="692" width="360" height="60" rx="8" fill="#c8e6c9" stroke="#2e7d32" stroke-width="2"/>
  <text x="220" y="715" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#1b5e20" text-anchor="middle">⑦-A 严格半场过滤</text>
  <text x="220" y="736" font-family="Consolas,monospace" font-size="11" fill="#1b5e20" text-anchor="middle">est_bx 在己方半场（&gt;0 / &lt;0）→ 用于观测特征</text>

  <rect x="480" y="692" width="360" height="60" rx="8" fill="#ffcdd2" stroke="#b71c1c" stroke-width="2"/>
  <text x="660" y="715" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#b71c1c" text-anchor="middle">⑦-B 全部检测结果</text>
  <text x="660" y="736" font-family="Consolas,monospace" font-size="11" fill="#b71c1c" text-anchor="middle">含对面半场球 → 用于「错误目标」惩罚判定</text>
</svg>

---

## ⚙️ PPO 训练配置

### 网络架构（MlpPolicy）

```
30 维 state → MLP[128, 128] → π(a|s)  (Actor, 输出 9 维 logits)
30 维 state → MLP[128, 128] → V(s)    (Critic, 输出标量)
```

参数量约 4 万级别，CPU 训练即可（视觉特征已经被 OpenCV 提取成 30 维向量，没必要上 GPU）。

### 超参数对照

| 超参数            | V1 (`train.py`)    | V2 (`train2.py`)                          |
|-------------------|--------------------|-------------------------------------------|
| `total_timesteps` | 500,000            | 500,000                                   |
| `learning_rate`   | `3e-4` 固定        | **`3e-4 → 1e-4` 线性衰减**                |
| `n_steps`         | 1024               | **2048**（更稳定的 advantage 估计）       |
| `batch_size`      | 64                 | 64                                        |
| `n_epochs`        | 10                 | 10                                        |
| `gamma`           | 0.99               | 0.99                                      |
| `gae_lambda`      | 0.95               | 0.95                                      |
| `clip_range`      | 0.2                | 0.2                                       |
| `ent_coef`        | `0.01` 固定        | **`0.01 → 0.003` 线性衰减**（自定义 Callback）|
| `vf_coef`         | 0.5                | 0.5                                       |
| `max_grad_norm`   | 0.5                | 0.5                                       |
| `device`          | cpu                | cpu                                       |
| `seed`            | 42                 | 42                                        |

### V2 关键改进：自定义 EntCoefScheduleCallback

SB3 原生不支持 `ent_coef` 作为 schedule（loss 公式要求标量），但 `PPO.train()` 每次都会重新读取 `self.ent_coef`，因此通过 `_on_rollout_start` 在每个 rollout 周期前动态写入新值即可实现衰减：

```python
def _on_rollout_start(self):
    progress = self.num_timesteps / self.total_timesteps
    self.model.ent_coef = self.start + (self.end - self.start) * progress
```

设计意图：训练初期高熵鼓励探索，后期低熵让策略收敛 —— 这是从 1e5 步以后才开始有效收益的关键 trick。

### 训练辅助回调

| Callback                  | 功能                                                      |
|---------------------------|-----------------------------------------------------------|
| `CheckpointCallback`      | 每 10,000 步保存一份 `ppo_tennis_*.zip`                  |
| `BestModelCallback`       | 每 5,000 步检查最近 50 局滚动平均奖励，超过历史最佳则覆盖保存 + 写 meta.json |
| `TrainingLogCallback`     | 每 10 episode 打印 Ep / mean / max / min / 成功率 / 当前 R / L |
| `EntCoefScheduleCallback` | V2 专属，按全局进度衰减 `ent_coef`                       |

所有 Callback 的状态（`episode_count` / `success_count` / `episode_rewards` / `best_mean_reward`）都通过 JSON 持久化，**resume 时全部继承**，统计数据跨多次 `learn()` 调用保持连续。

---

## 🔧 核心处理逻辑（环境内部）

- **视觉流水线**：图像 → 裁剪上方 40% → HSV 分割 → 形态学处理 → 轮廓检测 → 面积排序 → 距离/角度估计
- **半场训练机制**：固定活跃半场，无球时自动切换，全场无球时调用 Lua 脚本重新生成
- **终止条件**：成功消除、穿越球网、卡边界、卡网、超时（500 步）

---

## 🎮 运动控制方案的演进（关键技术挑战）

### 初期方案：麦克纳姆轮三元组控制
- 使用 `forward + turn` 两个参数生成四轮速度
- 优点：简单快速
- 缺点：转向侧滑严重，姿态控制不精确

### 中期方案：四元数直接控制
- 尝试直接设置 YouBot 朝向
- **重大问题**：**万向锁（Gimbal Lock）**，导致 YouBot 在某些角度剧烈抖动甚至翻车，训练极不稳定

### 最终方案（当前使用）：变换矩阵 + 四元数乘法
- 记录初始姿态（`_default_ori`）
- 重置时：**先恢复初始正常姿态**（保证水平）
- 再用**四元数乘法** `q_new = qz * quat` 只改变航向角
- 同时调用 `resetDynamicObject()` 清除残余速度
- 彻底解决万向锁问题，YouBot 重置姿态始终稳定

**初代全局坐标版本**（`TennisGlobalLocating_elimate_with_dist.py`）正是这一演进过程中的重要里程碑，它首次实现了**矩阵 yaw 提取 + 统一导航控制器**，为后续纯视觉版本奠定了可靠的运动控制基础。

---

## 🎾 自动网球生成脚本（解决模型迭代问题）

`scene/Tennis_Generate.lua` 必须**作为收集箱（`Bin_Entry`）的子 subframe 关联脚本**挂载，且具体属性如下：

| 配置项     | 取值              | 原因 |
|-----------|------------------|------|
| 脚本类型   | **Customization Script** | 随场景持久化保存，可被 `sim.callScriptFunction` 跨进程调用 |
| 执行模式   | **Non-threaded**         | 与仿真主循环同步，避免多线程下创建 Shape 出现时序竞争 |
| 语言       | **Lua**                  | CoppeliaSim 原生支持，`sim.*` API 零开销 |
| 宿主对象   | `Bin_Entry`（收集箱）    | Python 端通过 `sim.getObject('/Bin_Entry')` 获取脚本句柄 |

挂载后该脚本对外暴露 `spawnBalls(ball_count, seed)` 函数：

- Python 通过 `sim.callScriptFunction('spawnBalls', script_handle, [count, seed], [], [], '')` 远程调用
- 支持全场随机位置生成 + 避免网附近生成
- 每次全场无球时自动重新生成 12 个网球
- **核心意义**：彻底解决"固定位置训练导致过拟合"的问题，保证每次训练球的位置都不一样

> ⚠️ 若挂载方式不正确（例如用了 Threaded、用了普通 Child Script、或挂在了错误对象上），训练运行时会看到 `[rl_env] spawnBalls 调用失败` 报错，且全场无球后无法自动恢复。

---

## 🛰️ 部署阶段（Deploy_Collector2.py）

部署是规则代码 + RL Agent 的协作流程：

```
while 场上有球:
    1. RL Agent 在当前半场捡球（循环调用 env.step + model.predict）
    2. 连续看不到球或卡住 → 触发半场巡视
    3. 巡视确认当前半场清空 → 绕网切换到另一半场
    4. 重复
```

### 绕网（bypass_net）

| 段           | 路径点                       | 几何含义                            |
|--------------|------------------------------|-------------------------------------|
| 安全离网     | (sign·max(\|rx\|, 1.5), ±7.05) | 先离网 1.5m，再纵向退到走廊 Y=±7.05 |
| 横穿 X=0     | (target_half·3.0, ±7.05)     | 走廊位于网柱(\|Y\|=6.40)与椅子(\|Y\|≥7.72)之间 |

### 半场巡视（patrol_half）

S 形 6 路径点（近网 / 中场 / 底线 × 左右两端）单程扫描，从离 YouBot 更近的端点切入。
途中每 15 个仿真步检查视野，**严格阈值 `est_bx > 1.0` + 连续 3 次确认**才中断巡视（避免相机噪声 / 单帧误检）。

#### 场地俯视图（两侧半场巡视路径 + 绕网走廊 + 实际物体位置）

> 比例：X 轴 1m ≈ 22px，Y 轴 1m ≈ 17.5px；屏幕向上 = 世界 Y+。
> 物体位置严格依据 `scene/tennis_scene_latest.lua` 的真实坐标绘制。

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 480" width="880" height="480">

  <rect x="20" y="20" width="840" height="400" fill="#f5f5f5" stroke="#666" stroke-width="2" stroke-dasharray="6 4"/>
  <text x="28" y="35" font-family="Consolas,monospace" font-size="11" fill="#666">围栏 38.57 × 20.29 m</text>

  <rect x="40" y="40" width="800" height="320" fill="#7cb342" fill-opacity="0.20" stroke="#558b2f" stroke-width="1.5"/>
  <rect x="40" y="40" width="400" height="320" fill="#bbdefb" fill-opacity="0.30"/>
  <rect x="440" y="40" width="400" height="320" fill="#ffe082" fill-opacity="0.30"/>
  <text x="240" y="56" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#0d47a1" text-anchor="middle">X &lt; 0 半场（active_half = -1）</text>
  <text x="640" y="56" font-family="Consolas,monospace" font-size="13" font-weight="bold" fill="#bf360c" text-anchor="middle">X &gt; 0 半场（active_half = +1）</text>

  <rect x="180" y="104" width="520" height="192" fill="#7cb342" fill-opacity="0.45" stroke="#33691e" stroke-width="1.5"/>

  <line x1="40" y1="77" x2="840" y2="77" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="6 3"/>
  <line x1="40" y1="323" x2="840" y2="323" stroke="#1565c0" stroke-width="1.5" stroke-dasharray="6 3"/>
  <text x="48" y="73" font-family="Consolas,monospace" font-size="10" fill="#0d47a1">绕网走廊 Y=+7.05</text>
  <text x="48" y="335" font-family="Consolas,monospace" font-size="10" fill="#0d47a1">绕网走廊 Y=-7.05</text>

  <line x1="440" y1="40" x2="440" y2="360" stroke="#d32f2f" stroke-width="3"/>
  <text x="446" y="38" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#b71c1c">球网 X=0</text>
  <circle cx="440" cy="88" r="5" fill="#000"/>
  <circle cx="440" cy="312" r="5" fill="#000"/>
  <text x="448" y="92" font-family="Consolas,monospace" font-size="9" fill="#333">网柱 Y=+6.40</text>
  <text x="448" y="316" font-family="Consolas,monospace" font-size="9" fill="#333">网柱 Y=-6.40</text>

  <rect x="424" y="50" width="32" height="8" fill="#5d4037" stroke="#3e2723" stroke-width="1"/>
  <text x="460" y="58" font-family="Consolas,monospace" font-size="10" fill="#3e2723">长椅 #2  (X=0, Y=+8.345, 1.5×0.4m)</text>
  <rect x="424" y="342" width="32" height="8" fill="#5d4037" stroke="#3e2723" stroke-width="1"/>
  <text x="460" y="354" font-family="Consolas,monospace" font-size="10" fill="#3e2723">长椅 #1  (X=0, Y=-8.345, 1.5×0.4m)</text>

  <rect x="822" y="40" width="18" height="18" fill="#fb8c00" stroke="#e65100" stroke-width="1.5"/>
  <text x="822" y="35" font-family="Consolas,monospace" font-size="10" font-weight="bold" fill="#bf360c">📦 回收仓 Bin</text>
  <text x="690" y="68" font-family="Consolas,monospace" font-size="10" fill="#bf360c">中心 (17.885, 8.645) 0.8×1.0m</text>
  <line x1="780" y1="65" x2="822" y2="50" stroke="#bf360c" stroke-width="1" stroke-dasharray="2 2"/>

  <line x1="495" y1="322" x2="495" y2="78" stroke="#ff5722" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="495" y1="78"  x2="640" y2="78" stroke="#ff5722" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="640" y1="78"  x2="640" y2="322" stroke="#ff5722" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="640" y1="322" x2="807" y2="322" stroke="#ff5722" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="807" y1="322" x2="807" y2="78" stroke="#ff5722" stroke-width="2.5" stroke-opacity="0.9"/>
  <polygon points="803,90 811,90 807,80" fill="#ff5722"/>

  <circle cx="495" cy="322" r="11" fill="#ff5722" stroke="#fff" stroke-width="2"/>
  <text x="495" y="326" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">0</text>
  <circle cx="495" cy="78"  r="11" fill="#ff5722" stroke="#fff" stroke-width="2"/>
  <text x="495" y="82" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">1</text>
  <circle cx="640" cy="78"  r="11" fill="#ff5722" stroke="#fff" stroke-width="2"/>
  <text x="640" y="82" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">2</text>
  <circle cx="640" cy="322" r="11" fill="#ff5722" stroke="#fff" stroke-width="2"/>
  <text x="640" y="326" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">3</text>
  <circle cx="807" cy="322" r="11" fill="#ff5722" stroke="#fff" stroke-width="2"/>
  <text x="807" y="326" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">4</text>
  <circle cx="807" cy="78"  r="11" fill="#ff5722" stroke="#fff" stroke-width="2"/>
  <text x="807" y="82" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">5</text>

  <line x1="385" y1="322" x2="385" y2="78" stroke="#8e24aa" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="385" y1="78"  x2="240" y2="78" stroke="#8e24aa" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="240" y1="78"  x2="240" y2="322" stroke="#8e24aa" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="240" y1="322" x2="73"  y2="322" stroke="#8e24aa" stroke-width="2.5" stroke-opacity="0.9"/>
  <line x1="73"  y1="322" x2="73"  y2="78" stroke="#8e24aa" stroke-width="2.5" stroke-opacity="0.9"/>
  <polygon points="69,90 77,90 73,80" fill="#8e24aa"/>

  <circle cx="385" cy="322" r="11" fill="#8e24aa" stroke="#fff" stroke-width="2"/>
  <text x="385" y="326" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">0</text>
  <circle cx="385" cy="78"  r="11" fill="#8e24aa" stroke="#fff" stroke-width="2"/>
  <text x="385" y="82" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">1</text>
  <circle cx="240" cy="78"  r="11" fill="#8e24aa" stroke="#fff" stroke-width="2"/>
  <text x="240" y="82" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">2</text>
  <circle cx="240" cy="322" r="11" fill="#8e24aa" stroke="#fff" stroke-width="2"/>
  <text x="240" y="326" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">3</text>
  <circle cx="73"  cy="322" r="11" fill="#8e24aa" stroke="#fff" stroke-width="2"/>
  <text x="73"  y="326" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">4</text>
  <circle cx="73"  cy="78"  r="11" fill="#8e24aa" stroke="#fff" stroke-width="2"/>
  <text x="73"  y="82" font-family="Consolas,monospace" font-size="11" font-weight="bold" fill="#fff" text-anchor="middle">5</text>

  <circle cx="374" cy="200" r="6" fill="#1976d2"/>
  <text x="320" y="204" font-family="Consolas,monospace" font-size="10" fill="#0d47a1">绕网起点</text>
  <line x1="374" y1="200" x2="374" y2="80" stroke="#1976d2" stroke-width="2.5"/>
  <polygon points="370,84 378,84 374,76" fill="#1976d2"/>
  <line x1="374" y1="77" x2="504" y2="77" stroke="#1976d2" stroke-width="2.5"/>
  <polygon points="500,73 500,81 508,77" fill="#1976d2"/>
  <text x="320" y="140" font-family="Consolas,monospace" font-size="10" fill="#0d47a1">① 上走廊</text>
  <text x="430" y="70" font-family="Consolas,monospace" font-size="10" fill="#0d47a1">② 横穿球网</text>

  <rect x="20" y="430" width="840" height="44" fill="#fafafa" stroke="#bdbdbd" stroke-width="1"/>
  <circle cx="36" cy="445" r="7" fill="#ff5722"/>
  <text x="50" y="449" font-family="Consolas,monospace" font-size="11" fill="#333">X&gt;0 半场巡视 (0→1→2→3→4→5)</text>
  <circle cx="276" cy="445" r="7" fill="#8e24aa"/>
  <text x="290" y="449" font-family="Consolas,monospace" font-size="11" fill="#333">X&lt;0 半场巡视（镜像）</text>
  <circle cx="468" cy="445" r="6" fill="#1976d2"/>
  <text x="482" y="449" font-family="Consolas,monospace" font-size="11" fill="#333">绕网路径</text>
  <rect x="566" y="441" width="14" height="8" fill="#5d4037"/>
  <text x="586" y="449" font-family="Consolas,monospace" font-size="11" fill="#333">长椅</text>
  <rect x="624" y="439" width="12" height="12" fill="#fb8c00" stroke="#e65100"/>
  <text x="642" y="449" font-family="Consolas,monospace" font-size="11" fill="#333">回收仓</text>
  <text x="36" y="466" font-family="Consolas,monospace" font-size="10" fill="#555">巡视规则：每次进入半场时，比较 YouBot 到点 0 与点 5 的距离，从更近端点开始单程走完 6 个点；途中检测到当前半场内的球立即中断，交还 RL Agent。</text>
</svg>

### 卡顿兜底（三层保护）

部署阶段所有规则导航与 RL 衔接处都有兜底机制，按触发场景分为三层：

#### ① RL 原地摇头检测

部署层维护一个 80 步的位置滑动窗口，若窗口内 X / Y 范围 < 0.20 m，判定 RL agent 卡住（典型场景：球藏在车背后，agent 倾向原地摇头），主动终止 RL 循环触发巡视打破僵局。

#### ② 进入规则阶段前的贴墙脱困（`_escape_from_obstacle`）

`patrol_half()` 与 `bypass_net()` 入口先调用 `_escape_from_obstacle(safe_margin=1.5)`，确保 YouBot 离 4 边界 + 球网都 ≥ 1.5 m 才开始规则导航。

| 触发场景 | 处理 |
|---------|------|
| 上一轮 RL 因 `stuck_at_boundary` / `stuck_at_net` / `crossed_net` 终止 | YouBot 物理上仍贴墙，直接进入 `navigate_to` 会顶墙转空轮 |
| 脱困策略 | 朝场地中心 (0,0)：车头偏差 < 60° → 前进修正；偏差 ≥ 60° → 后退 + 边退边转 |
| 安全阀 | 上限 200 sim.step（≈10s 仿真），超时仍记录脱困失败但不阻断后续流程 |

#### ③ 导航途中卡死的自愈脱困

`navigate_to()` 内部每 60 sim.step（≈3s 仿真）检查累计位移：若 < 0.20 m 判定途中卡住（例如规则导航半路碰到球网），自动调用一次 `_escape_from_obstacle` 后继续向原目标推进。

| 参数 | 值 | 说明 |
|------|----|------|
| `STUCK_CHECK_EVERY` | 60 sim.step | 检查频率 |
| `STUCK_DISPLACEMENT` | 0.20 m | 60 步内位移阈值 |
| `MAX_INFLIGHT_ESCAPES` | 3 | 单次 navigate_to 内最多脱困次数，超过即放弃该点 |

**早期放弃**：单点最坏 4 × 60 = 240 步即可识别"无解死角"，约 12s，避免耗尽 NAV_MAX_ITER=2000 步（~100s）。

---

## 📊 训练日志指标详细解释

下表列出训练日志中所有重要指标，按出现顺序与重要程度分为三大类，便于查阅。

### 1. rollout/ 指标（数据收集阶段）

| 指标              | 含义说明                                       | 理想情况               |
|-------------------|-----------------------------------------------|------------------------|
| **ep_len_mean**   | 平均每个 episode 的步数长度                    | 稳定在 200~230 步      |
| **ep_rew_mean**   | 平均每个 episode 的总奖励（最重要整体指标）    | 持续上升或稳定在高位   |

### 2. time/ 指标（时间与效率）

| 指标               | 含义说明                                       | 关注点         |
|--------------------|-----------------------------------------------|-------------|
| **fps**            | 每秒处理的仿真步数（训练速度）                  | 当前 4~7，瓶颈明显 |
| **iterations**     | 当前第几次策略更新（PPO 内部计数）              | 持续增长        |
| **time_elapsed**   | 从本次 `learn()` 开始累计的真实经过秒数         | 训练总用时       |
| **total_timesteps**| 全局总交互步数（跨 resume 累计）                | 持续增长        |

### 3. train/ 指标（策略更新时的损失与训练状态）

| 指标                     | 含义说明                                                       | 理想范围 / 关注点      |
|--------------------------|---------------------------------------------------------------|------------------------|
| **approx_kl**            | 新旧策略差异程度（KL 散度）                                    | < 0.02（越小越稳定）   |
| **clip_fraction**        | 被 clip 的样本比例（PPO 防止更新过大）                         | 0.03 ~ 0.15 正常       |
| **entropy_loss**         | 策略熵（负值），值越高越探索，值越低越利用                      | 缓慢下降为正常         |
| **explained_variance**   | 价值函数（Critic）对真实回报的解释程度                          | 目标 > 0.75~0.85       |
| **learning_rate**        | 当前学习率                                                     | V1 固定 3e-4；V2 衰减  |
| **loss**                 | 总损失（policy_loss + value_loss + entropy_loss 的加权和）      | 波动较大属正常         |
| **n_updates**            | 累计进行的参数更新次数                                          | 持续增长               |
| **policy_gradient_loss** | Actor（策略）的梯度损失                                         | 接近 0 为好            |
| **value_loss**           | Critic（价值函数）的损失（项目中重点关注的指标之一）             | 稳定在 10~40 较好      |

### 4. 自定义 TrainingLogCallback 指标（每 10 episode 打印）

| 指标                  | 含义说明                                     | 理想范围 / 关注点      |
|-----------------------|---------------------------------------------|------------------------|
| **Ep**                | 当前总 episode 数量（跨 resume 继承）        | 持续增长               |
| **最近10局 mean**     | 最近 10 个 episode 的平均奖励（重点指标）    | 越高越好               |
| **最近10局 max / min**| 最近 10 个 episode 的最高 / 最低奖励         | min 不要过低           |
| **成功率**            | 全局成功消除网球的比例（最核心指标）         | 目标 90%+              |
| **当前 R**            | 当前 episode 已获得的总奖励                  | —                      |
| **L**                 | 当前 episode 的长度（步数）                  | 200~250 较理想         |

### 快速判断训练状态

- **成功率** + **ep_rew_mean** → 看整体表现（最重要）
- **explained_variance** > 0.75 且稳定 → 价值函数学得不错
- **value_loss** 长期 < 40 且不爆炸 → 训练健康
- **approx_kl** < 0.02 → 更新平稳
- **entropy_loss** 缓慢下降 → 正常从探索转向利用

---

## 🕰️ 版本迭代历史

**V0（初代全局坐标版）**  
文件：`TennisGlobalLocating_elimate_with_dist.py`  
- 使用全局坐标 + 全局距离消除
- 首次实现变换矩阵提取 yaw + 统一导航控制器
- 引入低通滤波 + 卡顿检测 + 简化绕网路径
- 奠定了后续纯视觉版本的运动控制基础

**V1（早期视觉版）**  
- 引入 HSV 视觉检测
- 仍依赖全局坐标辅助消除

**V2（纯视觉过渡版）**  
- 完全去除全局坐标
- 图像裁剪 + 稠密奖励塑造 + 半场切换

**V3（当前成熟版）**  
- 四元数姿态控制 + 动态网球生成 + 跨 resume 统计继承
- 仅消除判定用全局坐标，HSV 距离用于感知/奖励
- `BestModelCallback` 自动保存最佳模型
- Checkpoint 规范化 + 持久化训练状态

**V4（train2.py / tennis_rl_env2.py）**  
- 动作空间 7 → 9（新增 2 个后退动作）
- 奖励加视野新鲜度线性递增惩罚
- 学习率 3e-4 → 1e-4 线性衰减
- 熵系数 0.01 → 0.003 线性衰减（EntCoefScheduleCallback）
- n_steps 1024 → 2048

---

## 💡 开发过程中的主要困难与解决方案

1. **视觉噪声严重** → 图像裁剪上方 40% + 严格半场过滤
2. **Agent 频繁卡网/卡边界** → 梯度惩罚 + stuck 检测
3. **resume 时成功率统计归零** → `TrainingLogCallback` 增加 JSON 持久化
4. **运动控制万向锁** → 最终采用变换矩阵 + 四元数乘法
5. **网球位置固定导致过拟合** → bin 挂载 Lua 脚本动态生成
6. **TensorBoard 日志不连续** → 明确指定 `tb_log_name="PPO"`
7. **背后球只能摇头的局部最优** → V2 加入后退动作 + 视野新鲜度线性递增惩罚
8. **巡视阶段单帧误检触发误中断** → 严格 `est_bx` 阈值 + 连续 3 次确认双重保险
9. **绕网撞椅子** → 经过几何分析确定走廊 \|Y\|=7.05（网柱 6.40 与椅子 7.72 之间）
10. **部署时 V1 env 加载 V2 (9 动作) 模型 → `KeyError: 7`** → 在 `deploy()` 加 `model.action_space.n == env.action_space.n` 兼容性断言，加载阶段直接报错
11. **RL 因 stuck_at_boundary 终止后规则巡视立即顶墙超时** → `patrol_half` / `bypass_net` 入口先调 `_escape_from_obstacle(safe_margin=1.5)` 把 YouBot 拉离边界 ≥ 1.5m
12. **导航半路撞网卡死** → `navigate_to` 内每 60 步检查累计位移，途中卡住自动触发脱困并继续，单点最多 3 次脱困后早期放弃避免无限循环

---

## 🎯 最终效果

- 成功率稳定在 **88%~92%**
- 平均 episode 长度约 220 步
- 支持长时间 resume 训练 + 自动保存历史最佳模型
- 完整验证了纯视觉 RL 在复杂仿真环境中的可行性

---

## 🗒️ 常用命令速查

| 任务                | 命令                                                                 |
|---------------------|----------------------------------------------------------------------|
| 继续训练（resume）  | `python train2.py resume <checkpoint.zip>`                           |
| 评估已训练模型      | `python train2.py eval <model.zip>`                                  |
| 完整部署（RL+绕网+巡视）| `python Deploy_Collector2.py --model ./models_v2/best_model/best_model.zip` |
| TensorBoard 监控    | `tensorboard --logdir ./logs_v2 --port 6006`                         |
| 仅测试 RL 环境      | `python tennis_rl_env2.py`                                           |
| 全局坐标作弊版（V0）| `python TennisGlobalLocating_elimate_with_dist.py`                   |
| 训练 V1（旧动作空间）| `python train.py`                                                   |

> 从头训练命令在 [🚀 快速开始](#-快速开始) 第 3 步。

## 📄 开源协议（License）

本项目采用 **MIT License** 开源，详见仓库根目录的 `LICENSE` 文件。
你可以自由地使用、修改、再发布本项目代码，包括用于商业用途，仅需保留版权声明。

## 📚 引用（Citation）

如果本项目对你的研究/项目有帮助，欢迎引用：

```bibtex
@misc{tennis_collector_2026,
  author       = {uMemory},
  title        = {Tennis_Collector: Vision-only Reinforcement Learning for YouBot Tennis Ball Collection in CoppeliaSim},
  year         = {2026},
  howpublished = {\url{https://github.com/uMemory/CollectTennisBalls_RL_CoppeliaSim}}
}
```

## 📬 联系方式（Contact）

- **作者**：uMemory
- **完成时间**：2026 年 4 月
- **核心技术栈**：CoppeliaSim + ZMQ + Gymnasium + Stable-Baselines3 PPO + OpenCV
- **Issue / 讨论**：欢迎在 GitHub Issues 中提出问题或建议
