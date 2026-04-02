import math
import time
import random
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


class YouBotTennisCollector:
    def __init__(self):
        # 1. 连接 API
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        # 2. 获取机器人句柄
        self.robot_h = self.sim.getObject('/youBot')

        # 3. 获取四个轮子关节句柄
        self.wheels = {
            'fl': self.sim.getObject('/rollingJoint_fl'),
            'fr': self.sim.getObject('/rollingJoint_fr'),
            'rl': self.sim.getObject('/rollingJoint_rl'),
            'rr': self.sim.getObject('/rollingJoint_rr')
        }

        # 4. 初始化网球列表
        self.ball_handles = []
        self._find_balls()

        # 5. 设置参数
        self.collect_distance = 0.4

        self.actions = {
            'forward': [1, -1, 1, -1],
            'backward': [-1, 1, -1, 1],
            'turn_left': [-1, -1, -1, -1],
            'turn_right': [1, 1, 1, 1],
            'strafe_left': [-1, -1, 1, 1],
            'strafe_right': [1, 1, -1, -1]
        }

    def _find_balls(self):
        self.ball_handles = []
        for i in range(1, 16):
            try:
                name = f'/TennisBall_{i:02d}'
                h = self.sim.getObject(name)
                self.ball_handles.append(h)
            except:
                continue
        print(f"🎾 扫描完成：共发现 {len(self.ball_handles)} 个网球。")

    def set_velocity(self, v_list):
        self.sim.setJointTargetVelocity(self.wheels['fl'], v_list[0])
        self.sim.setJointTargetVelocity(self.wheels['fr'], v_list[1])
        self.sim.setJointTargetVelocity(self.wheels['rl'], v_list[2])
        self.sim.setJointTargetVelocity(self.wheels['rr'], v_list[3])

    def stop(self):
        self.set_velocity([0, 0, 0, 0])

    def check_collection(self):
        """检查并收集网球"""
        if not self.ball_handles:
            return

        r_pos = self.sim.getObjectPosition(self.robot_h, -1)

        for i in range(len(self.ball_handles) - 1, -1, -1):
            h = self.ball_handles[i]
            try:
                b_pos = self.sim.getObjectPosition(h, -1)
                dist = math.sqrt((r_pos[0] - b_pos[0]) ** 2 + (r_pos[1] - b_pos[1]) ** 2)

                if dist < self.collect_distance:
                    # ✅ 修正点：使用 sim.removeObjects 传入一个列表 [h]
                    self.sim.removeObjects([h])
                    self.ball_handles.pop(i)
                    print(f"✨ 收集成功！剩余: {len(self.ball_handles)}")
            except Exception as e:
                # 如果对象已经由于某种原因不存在，直接从追踪列表移除
                self.ball_handles.pop(i)

    def run(self):
        print("🚀 启动 YouBot 随机行走...")
        self.sim.startSimulation()

        try:
            while len(self.ball_handles) > 0:
                act_name = random.choice(list(self.actions.keys()))
                act_vec = self.actions[act_name]
                duration = random.uniform(1.0, 2.5)
                speed = random.uniform(2.5, 4.5)

                vels = [v * speed for v in act_vec]
                self.set_velocity(vels)

                start_t = time.time()
                while time.time() - start_t < duration:
                    self.check_collection()
                    if not self.ball_handles: break
                    time.sleep(0.02)

                self.stop()
                time.sleep(0.1)

            print("🏆 球场清理完毕！")

        except KeyboardInterrupt:
            print("\n🛑 停止中...")
        finally:
            self.stop()
            self.sim.stopSimulation()


if __name__ == "__main__":
    collector = YouBotTennisCollector()
    collector.run()