## 本项目强化学习应用详解

### 整体框架

```
CoppeliaSim 仿真环境（4.10.0）
        ↕ ZMQ通信
Python RL训练 (PPO算法)
```



### 项目结构

```
CollectTennisBall_RL_CoppeliaSim/
│───.gitignore
├───logs			-- 训练日志
├───models			-- 训练缓存
└───scene/
│   ├───backup.ttt	-- 可直接使用的场景文件
│   ├───bin.lua		-- 优化后的网球收集箱脚本
│   ├───scene.ttt	-- 可直接使用的场景文件
│   ├───Tennis_Generate.lua			-- 随机位置网球生成脚本
│   ├───tennis_scene_init.lua		-- 创建场景脚本Version 1.0（内容与scene.ttt一致）
│   └───tennis_scene_latest.lua		-- 创建场景脚本Version 2.0（内容与backup.ttt一致）
│       

│───global_Locate_elimate_with_dis.py	-- 简约版
│───testCollect.py			-- 实现直接获取网球全局坐标，依赖距离判定消除网球（注释版）
│───test_yoloWorld.py
│───tmp.py				
│───train.py				-- 
│───YoubotMovement.py		-- 调试代码，测试环境及代码是否正常
│───requirements.txt
│───README.md

```



## 项目概述

本项目实现了一个**纯视觉驱动的 YouBot 网球收集强化学习智能体**，在 CoppeliaSim 真实比例（1:1）网球场环境中完成**半场自主发现、接近并消除网球，最终送回收仓**的任务。

- **仿真平台**：CoppeliaSim 4.10.0 + ZMQ Remote API
- **RL 算法**：Stable-Baselines3 PPO（MlpPolicy）
- **感知方式**：单目 visionSensor + HSV 颜色分割（无真实深度，仅靠像素面积估算距离）
- **最终效果**：成功率稳定在 **85%~92%**，支持长时间 resume 训练，具备较好的鲁棒性

项目从**全局坐标作弊版**开始，经过多次重大迭代，最终形成一套**工程化程度较高**的纯视觉 RL 解决方案。

## 核心环境设计（TennisCollectorEnv）

### 1. 状态空间（Observation Space）
单帧 10 维语义特征 + 3 帧堆叠 = **30 维**连续向量。

**10 维单帧特征**：
- `ball_detected`：活跃半场内是否检测到网球（0/1）
- `ball_angle`：网球在图像中的归一化水平偏角（-1 ~ 1）
- `ball_size`：基于像素面积的归一化大小（距离代理）
- `ball_count`：当前半场内可见网球数量（归一化）
- `ball_reachable`：估计位置是否在可达范围内（宽松判据）
- `norm_rx`、`norm_ry`、`norm_yaw`：机器人归一化位置与航向角
- `norm_net`：到球网的归一化距离
- `norm_bound`：到场地最近边界的归一化距离

### 2. 动作空间（Action Space）
7 个离散动作（`Discrete(7)`），每个动作持续执行 4 步仿真：
- 0：直行前进
- 1/2：左/右前弧线前进
- 3/4：小角度左/右原地转
- 5/6：大角度左/右原地转

### 3. 奖励函数（Reward Shaping）
采用**稠密引导 + 稀疏目标 + 多重惩罚**的设计：

- **稀疏大奖励**：成功消除网球 → `+100`
- **稠密引导**：看到活跃半场内的球并靠近 → 正向奖励 + 角度对齐奖励 + 大小增大奖励
- **惩罚项**：
  - 视野中只有对面半场的球 → `-0.5`
  - 完全看不到球 → `-0.3`
  - 靠近边界/球网 → 梯度惩罚
  - 静止不动 → `-1.0`
  - 卡边界/卡网 → 终止并惩罚 `-10`
  - 穿越球网 → 终止并惩罚 `-10`

### 4. 核心处理逻辑
- **视觉流水线**：图像 → 裁剪上方 40% → HSV 分割 → 形态学处理 → 轮廓检测 → 面积排序 → 距离/角度估计
- **半场训练机制**：固定活跃半场，无球时自动切换，全场无球时调用 Lua 脚本重新生成
- **终止条件**：成功消除、穿越球网、卡边界、卡网、超时（500 步）

## 运动控制方案的演进（关键技术挑战）

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

## 自动网球生成脚本（解决模型迭代问题）

在 `Bin_Entry` 对象上挂载 `bin_Customization_Script.lua`，提供 `spawnBalls(ball_count, seed)` 函数。

- Python 通过 `sim.callScriptFunction` 远程调用
- 支持全场随机位置生成 + 避免网附近生成
- 每次全场无球时自动重新生成 12 个网球
- **核心意义**：彻底解决“固定位置训练导致过拟合”的问题，保证每次训练球的位置都不一样

## 训练日志指标详细解释

### 自定义 TrainingLogCallback 指标
- **Ep**：当前总 episode 数量（跨 resume 继承）
- **最近10局 mean / max / min**：最近 10 个 episode 的奖励统计
- **成功率**：全局成功消除网球的比例
- **当前 R**：当前 episode 总奖励
- **L**：当前 episode 长度（步数）

### Stable-Baselines3 PPO 指标

**rollout/**（数据收集阶段）：
- `ep_len_mean`：平均 episode 长度（当前约 210~230 步）
- `ep_rew_mean`：平均 episode 奖励（最重要的整体表现指标）

**time/**（时间相关）：
- `fps`：每秒处理的步数（当前约 4~6）
- `time_elapsed`：从训练开始到现在累计的真实秒数（训练总用时）
- `iterations`：当前第几次策略更新
- `total_timesteps`：总交互步数（全局累计）

**train/**（策略更新时的损失）：
- `approx_kl`：新旧策略差异（理想 < 0.02）
- `clip_fraction`：被 clip 的样本比例
- `entropy_loss`：策略熵（逐渐下降表示从探索转向利用）
- `explained_variance`：价值函数解释方差的比例（目标 > 0.8）
- `loss`：总损失
- `n_updates`：累计参数更新次数
- `policy_gradient_loss`：策略梯度损失
- `value_loss`：价值函数损失（偶尔较高是当前主要关注点）

## 版本迭代历史

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
- `BestModelCallback` 自动保存最佳模型
- Checkpoint 规范化 + 持久化训练状态

## 开发过程中的主要困难与解决方案

1. **视觉噪声严重** → 图像裁剪上方 40% + 严格半场过滤
2. **Agent 频繁卡网/卡边界** → 梯度惩罚 + stuck 检测
3. **resume 时成功率统计归零** → `TrainingLogCallback` 增加 JSON 持久化
4. **运动控制万向锁** → 最终采用变换矩阵 + 四元数乘法
5. **网球位置固定导致过拟合** → bin 挂载 Lua 脚本动态生成
6. **TensorBoard 日志不连续** → 明确指定 `tb_log_name="PPO"`

## 最终效果

- 成功率稳定在 **85%~92%**
- 平均 episode 长度约 220 步
- 支持长时间 resume 训练 + 自动保存历史最佳模型
- 完整验证了纯视觉 RL 在复杂仿真环境中的可行性

**项目作者**：Mory  
**完成时间**：2026 年 4 月  
**核心技术栈**：CoppeliaSim + ZMQ + Gymnasium + Stable-Baselines3 PPO
