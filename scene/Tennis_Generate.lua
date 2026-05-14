-- ============================================================
--  spawn_tennis_balls.lua
--  独立网球生成脚本 — 在已有场景中生成随机位置的网球
--
--  用法：在 CoppeliaSim 脚本编辑器中运行，或作为 Add-on 加载
--  功能：
--    1. 清除场景中所有已有的 TennisBall_xx 对象
--    2. 在球场范围内随机生成指定数量的缝线网球
--
--  参数（修改下方 CONFIG 区域）：
--    BALL_COUNT  : 生成数量（默认 12）
--    SEED        : 随机种子（nil 表示每次不同）
-- ============================================================

local W = sim.handle_world
local cleanIntrinsicBalls = true
-- ┌─────────────────────────────────────────────────────────┐
-- │  CONFIG — 在这里修改参数                                   │
-- └─────────────────────────────────────────────────────────┘
local BALL_COUNT = 12       -- 生成网球数量
local SEED       = nil      -- 随机种子，nil=每次随机，填数字=可复现

-- ┌─────────────────────────────────────────────────────────┐
-- │  球场尺寸常量）                │
-- └─────────────────────────────────────────────────────────┘
local CL  = 23.77   -- 球场长（底线到底线）
local SW  = 8.23    -- 单打宽
local DW  = 10.97   -- 双打宽
local OL = CL + 6.40 * 2   -- 36.57
local OW = DW + 3.66 * 2   -- 18.29

-- ── 清除旧网球 ─────────────────────────────────────────────
local function cleanBalls()
    local removed = 0
    local allObj = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for _, h in ipairs(allObj) do
        local ok, alias = pcall(sim.getObjectAlias, h, 0)
        if ok and alias and alias:sub(1, 11) == "TennisBall_" then
            pcall(sim.removeObjects, {h})
            removed = removed + 1
        end
    end
    if removed > 0 then
        print(string.format("🗑️  已清除 %d 个旧网球", removed))
    end
end

-- ── 缝线网球生成器（与场景脚本中的 createTennisBall 完全一致）──
local function createTennisBall(name, pos)
    local R      = 0.1  --  0.0335 原真比例的半径
    local mass   = 0.057
    local seamR  = 0.0018
    local A      = 0.38
    local N      = 20

    local ballC    = {0.85, 0.92, 0.10}
    local seamC    = {0.96, 0.96, 0.93}
    local feltSpec = {0.04, 0.04, 0.02}
    local feltEmit = {0.06, 0.07, 0.01}

    local d = R * 2
    local main = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {d, d, d}, 0)
    sim.setObjectPosition(main, pos, W)
    sim.setObjectInt32Param(main, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(main, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(main, nil, sim.colorcomponent_ambient_diffuse, ballC)
    sim.setShapeColor(main, nil, sim.colorcomponent_specular, feltSpec)
    sim.setShapeColor(main, nil, sim.colorcomponent_emission, feltEmit)

    local parts = {main}
    local Rsurf = R * 0.97

    for t_idx = 0, N - 1 do
        local t   = (t_idx / N) * 2 * math.pi
        local phi = A * math.sin(2 * t)
        local sd  = seamR * 2

        local x1 = pos[1] + Rsurf * math.cos(t) * math.cos(phi)
        local y1 = pos[2] + Rsurf * math.sin(t) * math.cos(phi)
        local z1 = pos[3] + Rsurf * math.sin(phi)
        local s1 = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {sd, sd, sd}, 0)
        sim.setObjectPosition(s1, {x1, y1, z1}, W)
        sim.setShapeColor(s1, nil, sim.colorcomponent_ambient_diffuse, seamC)
        sim.setObjectInt32Param(s1, sim.shapeintparam_respondable, 0)
        sim.setObjectInt32Param(s1, sim.shapeintparam_static, 0)
        table.insert(parts, s1)

        local x2 = pos[1] - Rsurf * math.sin(t) * math.cos(phi)
        local y2 = pos[2] + Rsurf * math.cos(t) * math.cos(phi)
        local z2 = pos[3] + Rsurf * math.sin(phi)
        local s2 = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {sd, sd, sd}, 0)
        sim.setObjectPosition(s2, {x2, y2, z2}, W)
        sim.setShapeColor(s2, nil, sim.colorcomponent_ambient_diffuse, seamC)
        sim.setObjectInt32Param(s2, sim.shapeintparam_respondable, 0)
        sim.setObjectInt32Param(s2, sim.shapeintparam_static, 0)
        table.insert(parts, s2)
    end

    if #parts > 1 then
        local grouped = sim.groupShapes(parts, false)
        sim.setObjectAlias(grouped, name)
        sim.setShapeMass(grouped, mass)
        sim.setObjectInt32Param(grouped, sim.shapeintparam_respondable, 1)
        return grouped
    else
        sim.setObjectAlias(main, name)
        sim.setShapeMass(main, mass)
        return main
    end
end

-- ── 随机位置生成──────────────────────
--
--  分三组分布，确保球散布在球场各个区域：
--    组 1（前 50%）：单打区内随机（主区域）
--    组 2（接下来 25%）：底线附近 + 双打区（边缘区域）
--    组 3（最后 25%）：中场区域（靠近球网两侧）
--
--  所有球避开球网碰撞墙区域（|X| < 0.3m 且 |Y| < 6.5m）
--
local function generateRandomPosition(i, total)
    local bx, by
    local ballR = 0.0335

    -- 分组比例
    local group1_end = math.floor(total * 0.5)
    local group2_end = math.floor(total * 0.75)

    if i <= group1_end then
        -- 组 1：单打区内随机
        bx = (math.random() - 0.5) * (OL - 2.0)
        by = (math.random() - 0.5) * (OW - 1.0)
    elseif i <= group2_end then
        -- 组 2：底线附近 + 双打区
        bx = (math.random() > 0.5 and 1 or -1) * (OL/2 - math.random() * 3.0)
        by = (math.random() - 0.5) * (OW - 2.0)
    else
        -- 组 3：中场区域
        bx = (math.random() - 0.5) * 10.0
        by = (math.random() - 0.5) * (OW - 2.0)
    end

    -- 安全检查：避开球网碰撞墙区域
    if math.abs(bx) < 0.3 and math.abs(by) < 6.5 then
        -- 推到离网 1m 的位置
        bx = (bx >= 0 and 1.0 or -1.0)
    end

    return bx, by, ballR + 0.003
end

-- ══════════════════════════════════════════════════════════════
--  主执行
-- ══════════════════════════════════════════════════════════════

-- 设置随机种子
if SEED then
    math.randomseed(SEED)
    print(string.format("🎲 随机种子: %d", SEED))
else
    math.randomseed(os.time())
    print("🎲 随机种子: 基于当前时间")
end

-- 清除旧球
if cleanIntrinsicBalls then
    cleanBalls()
end

-- 生成新球
print(string.format("🎾 正在生成 %d 个缝线网球...", BALL_COUNT))

for i = 1, BALL_COUNT do
    local bx, by, bz = generateRandomPosition(i, BALL_COUNT)

    createTennisBall(
        string.format("TennisBall_%02d", i),
        {bx, by, bz}
    )

    if i % 4 == 0 or i == BALL_COUNT then
        print(string.format("    已生成 %d/%d 个网球...", i, BALL_COUNT))
    end
end

print("═══════════════════════════════════════════════════════")
print(string.format("✅  网球生成完毕：%d 个缝线网球", BALL_COUNT))
print("═══════════════════════════════════════════════════════")