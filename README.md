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
│───collectTennis.py				-- 测试文件
│───depthForTennisElimination.py	-- 待定
│───get.py							-- 配置自检
│───global_Locate_elimate_with_dis.py	-- 简约版
│───script.lua
│───testCollect.py			-- 实现直接获取网球全局坐标，依赖距离判定消除网球（注释版）
│───test_yoloWorld.py
│───tmp.py				
│───train.py				-- 待定
│───yolov8s-worldv2.pt		-- 模型
│───YoubotMovement.py		-- 调试代码，测试环境及代码是否正常
│───requirements.txt
│───README.md

```



#### 1. 智能体 (Agent)
**YouBot 机器人**，需要学会自主捡球并送回回收仓。

#### 2. 环境 (Environment)
**TennisEnv 类**，封装了 CoppeliaSim 仿真场景，实现了标准 Gym 接口：
```
reset() → 开始新一局
step()  → 执行一个动作，返回观测+奖励
```

#### 3. 状态/观测空间 (State，11维)
