import time
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ====================== 配置 ======================
MAX_BALLS_DETECT = 5
MAX_STEPS = 4000
BASE_SPEED = 7.0
TURN_SPEED = 3.2
ELIM_GLOBAL_DIST = 0.35
ACTION_DURATION = 0.15  # 每个动作持续时间


class TennisCollectEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        self.youbot_h = self.sim.getObject('/youBot') if self.sim.getObject('/youBot') != -1 else self.sim.getObject(
            '/youBot_base')
        self.sensor_h = self.sim.getObject('/visionSensor')

        self.fl = self.sim.getObject('/rollingJoint_fl')
        self.fr = self.sim.getObject('/rollingJoint_fr')
        self.rl = self.sim.getObject('/rollingJoint_rl')
        self.rr = self.sim.getObject('/rollingJoint_rr')

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)  # 保持16维

        self.collected_count = 0
        self.step_count = 0
        self.no_target_steps = 0

        self.lower_hsv = np.array([20, 80, 100])
        self.upper_hsv = np.array([40, 255, 255])

    def set_motors(self, vfl, vfr, vrl, vrr):
        self.sim.setJointTargetVelocity(self.fl, float(vfl))
        self.sim.setJointTargetVelocity(self.fr, float(vfr))
        self.sim.setJointTargetVelocity(self.rl, float(vrl))
        self.sim.setJointTargetVelocity(self.rr, float(vrr))

    def stop_robot(self):
        self.set_motors(0, 0, 0, 0)

    def get_youbot_pose(self):
        pos = self.sim.getObjectPosition(self.youbot_h, self.sim.handle_world)
        m = self.sim.getObjectMatrix(self.youbot_h, self.sim.handle_world)
        head_wx = -m[2]
        head_wy = -m[6]
        yaw = math.atan2(head_wy, head_wx)
        return pos[0], pos[1], yaw

    def get_rgb_image(self):
        img_buf, res = self.sim.getVisionSensorImg(self.sensor_h, 0)
        W, H = res[0], res[1]
        img_np = np.frombuffer(img_buf, dtype=np.uint8).reshape(H, W, 3)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def detect_balls(self):
        bgr = self.get_rgb_image()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 25:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])

            # 更新为正确的 FOV = 80°
            bearing = ((cx - 512) / 512.0) * (80 * math.pi / 180.0)

            # 改进距离估计（结合面积 + 远距离衰减）
            dist_est = max(0.4, 19.5 / math.sqrt(area + 1e-6))

            detections.append({'bearing': bearing, 'dist': dist_est, 'area': area})

        detections.sort(key=lambda x: x['dist'])
        return detections[:MAX_BALLS_DETECT]

    def reset(self, seed=None, options=None):
        self.collected_count = 0
        self.step_count = 0
        self.no_target_steps = 0
        self.stop_robot()
        time.sleep(0.6)

        DEFAULT_POS = [6.400, -0.100, 0.096]
        DEFAULT_ORI = [math.radians(-90.0), math.radians(0.1), math.radians(-90.0)]
        self.sim.setObjectPosition(self.youbot_h, DEFAULT_POS, self.sim.handle_world)
        self.sim.setObjectOrientation(self.youbot_h, DEFAULT_ORI, self.sim.handle_world)

        print("✅ Env reset 完成")
        time.sleep(1.0)
        return self._get_obs(), {"info": "reset"}

    def _get_obs(self):
        rx, _, ryaw = self.get_youbot_pose()
        detections = self.detect_balls()
        obs = np.zeros(16, dtype=np.float32)

        obs[0] = np.clip(rx / 15.0, -1.0, 1.0)
        obs[1] = np.clip(abs(rx) / 12.0, 0.0, 1.0)
        obs[2] = np.clip(ryaw / math.pi, -1.0, 1.0)
        obs[3] = np.clip(self.collected_count / 12.0, 0.0, 1.0)
        obs[4] = np.clip(self.no_target_steps / 200.0, 0.0, 1.0)
        obs[5] = np.clip(len(detections) / MAX_BALLS_DETECT, 0.0, 1.0)

        for i, det in enumerate(detections):
            idx = 6 + i * 2
            if idx + 1 < 16:  # 防止越界
                obs[idx] = np.clip(det['bearing'] / math.pi, -1.0, 1.0)
                obs[idx + 1] = np.clip(det['dist'] / 15.0, 0.0, 1.0)

        return obs

    def step(self, action):
        self.step_count += 1
        v_cmd = float(action[0])
        omega_cmd = float(action[1])

        v_linear = v_cmd * BASE_SPEED
        omega = omega_cmd * TURN_SPEED

        # 动作执行
        if abs(omega_cmd) > 0.65:
            turn = TURN_SPEED if omega_cmd > 0 else -TURN_SPEED
            self.set_motors(turn, -turn, turn, -turn)
        else:
            self.set_motors(
                v_linear + omega * 0.75,
                v_linear - omega * 0.75,
                v_linear + omega * 0.75,
                v_linear - omega * 0.75
            )

        time.sleep(ACTION_DURATION)
        self.set_motors(0, 0, 0, 0)
        time.sleep(0.02)

        # ==================== 改进后的奖励函数（关键修复）================
        reward = -0.05
        done = False

        rx, ry, _ = self.get_youbot_pose()
        detections = self.detect_balls()

        # 1. 收集奖励
        for i in range(1, 13):
            try:
                h = self.sim.getObject(f"/TennisBall_{i:02d}")
                bx, by = self.sim.getObjectPosition(h, self.sim.handle_world)[:2]
                dist = math.hypot(bx - rx, by - ry)
                if dist < ELIM_GLOBAL_DIST:
                    self.sim.removeObjects([h])
                    self.collected_count += 1
                    reward += 85.0
            except:
                continue

        if self.collected_count >= 12:
            reward += 200.0
            done = True

        # 2. 靠近球奖励
        if detections:
            nearest_dist = detections[0]['dist']
            reward += 3.0 * (1.0 - min(nearest_dist, 12.0) / 12.0)
            self.no_target_steps = 0
        else:
            self.no_target_steps += 1
            reward += 0.15 if abs(omega_cmd) > 0.4 else -0.1  # 转弯给正，不转给负

            # 【新增】强烈惩罚长时间不动或贴墙不动
        if self.no_target_steps > 40:
            reward -= 1.5 * (self.no_target_steps / 100.0)  # 越久惩罚越重

        # 3. 【新增】强烈边界惩罚 - 这是解决撞墙的核心！
        boundary_penalty = 0.0
        if abs(rx) > 17.0:  # 接近底线（X方向）
            boundary_penalty += (abs(rx) - 17.0) * 8.0
        if abs(ry) > 8.0:  # 接近侧边围栏（Y方向）
            boundary_penalty += (abs(ry) - 8.0) * 10.0

        reward -= boundary_penalty

        # 4. 撞网惩罚
        if abs(rx) < 0.9 and v_cmd > 0.3:
            reward -= 6.0

        # 5. 【新增】长时间卡顿惩罚（防止原地打转或死撞）
        if self.no_target_steps > 80:  # 连续80步没看到球
            reward -= 0.8

        # 终止条件
        if self.step_count >= MAX_STEPS:
            done = True

        return self._get_obs(), reward, done, False, {"collected": self.collected_count}


    def close(self):
        self.stop_robot()
