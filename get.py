import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')
print("✅ 连接成功")

# ── 测试1：获取YouBot句柄 ──
try:
    youbot = sim.getObject('/youBot')
    print(f"✅ YouBot句柄: {youbot}")
except Exception as e:
    print(f"❌ 获取YouBot失败: {e}")

# ── 测试2：获取YouBot位置 ──
try:
    pos = sim.getObjectPosition(youbot, sim.handle_world)
    print(f"✅ YouBot位置: {pos}")
except Exception as e:
    print(f"❌ 获取位置失败: {e}")

# ── 测试3：逐个尝试获取网球句柄 ──
print("\n--- 网球句柄搜索 ---")
found_balls = []
for i in range(1, 13):
    name = f'/TennisBall_{i:02d}'
    try:
        h = sim.getObject(name)
        print(f"✅ {name} → 句柄: {h}")
        found_balls.append(h)
    except Exception as e:
        print(f"❌ {name} → {e}")

print(f"\n共找到 {len(found_balls)} 个网球")


# ── 测试4：获取摄像头句柄 ──
try:
    cam = sim.getObject('/youBot/visionSensor')
    print(f"✅ visionSensor句柄: {cam}")
except Exception as e:
    print(f"❌ 获取摄像头失败: {e}")

# ── 测试5：读取摄像头图像 ──
try:
    img, resolution = sim.getVisionSensorImg(cam)
    print(f"✅ 图像读取成功，分辨率: {resolution}")
except Exception as e:
    print(f"❌ 读取图像失败: {e}")

# ── 测试6：读取深度图 ──
try:
    depth_buf, _ = sim.getVisionSensorDepth(cam)
    depth_np = np.frombuffer(depth_buf, dtype=np.float32)
    print(f"✅ 深度图读取成功，数据长度: {len(depth_np)}，范围: [{depth_np.min():.3f}, {depth_np.max():.3f}]")
except Exception as e:
    print(f"❌ 读取深度图失败: {e}")