from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def test_spawn_balls():
    print("正在连接 CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.require('sim')
    print(f"连接成功，仿真状态: {sim.getSimulationState()}")

    try:
        spawner_obj = sim.getObject('/Bin_Entry')
        print(f"Bin_Entry 句柄: {spawner_obj}")
    except Exception as e:
        print(f"找不到 Bin_Entry: {e}")
        return

    # 获取 Customization script 句柄
    try:
        script_handle = sim.getScript(sim.scripttype_customizationscript, spawner_obj)
        print(f"Customization script 句柄: {script_handle}")
    except Exception as e:
        print(f"找不到 Customization script: {e}")
        return

    # 调用 spawnBalls
    print("\n正在调用 spawnBalls(count=12, seed=0)...")
    try:
        result = sim.callScriptFunction(
            'spawnBalls',
            script_handle,
            12, 0               # 直接传参: ball_count=12, seed=0
        )
        print(f"调用成功，返回值: {result}")
    except Exception as e:
        print(f"callScriptFunction 失败: {e}")
        return

    # 验证球是否存在
    print("\n验证场景中的网球对象...")
    found = 0
    for i in range(1, 13):
        name = f"TennisBall_{i:02d}"
        try:
            h = sim.getObject(f'/{name}')
            pos = sim.getObjectPosition(h, sim.handle_world)
            print(f"{name}: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")
            found += 1
        except Exception:
            print(f"{name}: 不存在")

    print(f"\n{'✅ 测试通过' if found == 12 else '⚠️ 测试部分通过'} | 找到 {found}/12 个网球")

if __name__ == '__main__':
    test_spawn_balls()