import time
import random
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ================= 1. 初始化与连接 =================
client = RemoteAPIClient()
sim = client.require('sim')

print("✅ 连接成功！开始启动随机行走程序...")

# 获取四个轮子关节句柄
fl = sim.getObject('/rollingJoint_fl')
fr = sim.getObject('/rollingJoint_fr')
rl = sim.getObject('/rollingJoint_rl')
rr = sim.getObject('/rollingJoint_rr')


# ================= 2. 定义控制与动作空间 =================
def set_motors(vfl, vfr, vrl, vrr):
    """设置四个电机的目标速度"""
    sim.setJointTargetVelocity(fl, vfl)
    sim.setJointTargetVelocity(fr, vfr)
    sim.setJointTargetVelocity(rl, vrl)
    sim.setJointTargetVelocity(rr, vrr)


def stop():
    """停止机器人"""
    set_motors(0, 0, 0, 0)


# 定义 YouBot 的基础动作向量 [FL, FR, RL, RR]
# 依据建场景lua代码：前进为 [正, 负, 正, 负]，左转为 [负, 负, 负, 负]
ACTIONS = {
    'forward': [1, -1, 1, -1],
    'backward': [-1, 1, -1, 1],
    'turn_left': [-1, -1, -1, -1],
    'turn_right': [1, 1, 1, 1],
    'strafe_left': [-1, -1, 1, 1],  # 麦克纳姆轮向左平移
    'strafe_right': [1, 1, -1, -1]  # 麦克纳姆轮向右平移
}

# ================= 3. 随机行走主循环 =================
print("🚀 进入随机行走模式，按 Ctrl+C 停止运行...")

try:
    while True:
        # 1. 随机选择一个动作
        action_name, action_vector = random.choice(list(ACTIONS.items()))

        # 2. 随机生成基础速度 (例如 2.0 到 5.0 rad/s 之间)
        speed = random.uniform(2.0, 5.0)

        # 3. 计算最终的轮子速度
        vfl, vfr, vrl, vrr = [v * speed for v in action_vector]

        # 4. 随机生成该动作的持续时间 (例如 0.5 到 2.5 秒)
        duration = random.uniform(1, 4)

        print(f"👉 执行动作: {action_name:<12} | 速度: {speed:.2f} | 持续时间: {duration:.2f}s")

        # 执行运动
        set_motors(vfl, vfr, vrl, vrr)
        time.sleep(duration)

        # 可选：在切换动作前加入短暂的停止，避免电机瞬间反转过载
        stop()
        time.sleep(0.2)

except KeyboardInterrupt:
    # ================= 4. 安全退出机制 =================
    print("\n🛑 接收到退出信号 (Ctrl+C)，正在停止机器人...")
    stop()
    print("✅ 机器人已停止，程序退出。")