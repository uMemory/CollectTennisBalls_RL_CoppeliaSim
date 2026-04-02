-- ============================================================
--  tennis_scene_v7.lua  1:1 真实比例网球场
--  CoppeliaSim 4.10.0
--  全部尺寸为真实世界米制，无任何缩放
--  球场 23.77m×10.97m  |  网球直径 6.7cm  |  球网高 0.914m
-- ============================================================

local W = sim.handle_world

-- ── 工具函数 ─────────────────────────────────────────────────
local function box(name, pos, size, color, static, mass)
    local h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectPosition(h, pos, W)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, static and 1 or 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(h, nil, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectAlias(h, name)
    if mass and not static then sim.setShapeMass(h, mass) end
    return h
end

local function cyl(name, pos, ori, r, height, color, static, mass)
    local h = sim.createPrimitiveShape(sim.primitiveshape_cylinder, {r*2, r*2, height}, 0)
    sim.setObjectPosition(h, pos, W)
    if ori then sim.setObjectOrientation(h, ori, W) end
    sim.setObjectInt32Param(h, sim.shapeintparam_static, static and 1 or 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(h, nil, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectAlias(h, name)
    if mass and not static then sim.setShapeMass(h, mass) end
    return h
end

-- ────────────────────────────────────────────────────────────
--  仿真缝线网球生成器
--  主球体 + 沿参数曲线放置的微型白色球体 → 合并为复合体
--  缝线曲线: θ(t)=t, φ(t)=A·sin(2t), t∈[0,2π]
-- ────────────────────────────────────────────────────────────
local function createTennisBall(name, pos)
    local R      = 0.0335                    -- 半径 33.5mm (ITF: 6.54~6.86cm 直径)
    local mass   = 0.057                     -- 57g
    local seamR  = 0.0018                    -- 缝线粒子半径
    local A      = 0.38                      -- 缝线振幅 (弧度)
    local N      = 20                        -- 每条缝线采样点数

    local ballC  = {0.85, 0.92, 0.10}       -- ITF "optic yellow"
    local seamC  = {0.96, 0.96, 0.93}       -- 缝线白色
    local feltSpec = {0.04, 0.04, 0.02}     -- 低镜面 → 毛毡感
    local feltEmit = {0.06, 0.07, 0.01}     -- 微发光 → 荧光感

    -- 主球体
    local d = R * 2
    local main = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {d, d, d}, 0)
    sim.setObjectPosition(main, pos, W)
    sim.setObjectInt32Param(main, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(main, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(main, nil, sim.colorcomponent_ambient_diffuse, ballC)
    sim.setShapeColor(main, nil, sim.colorcomponent_specular, feltSpec)
    sim.setShapeColor(main, nil, sim.colorcomponent_emission, feltEmit)

    -- 缝线粒子
    local parts = {main}
    local Rsurf = R * 0.97

    for t_idx = 0, N - 1 do
        local t   = (t_idx / N) * 2 * math.pi
        local phi = A * math.sin(2 * t)
        local sd  = seamR * 2

        -- 缝线 1
        local x1 = pos[1] + Rsurf * math.cos(t) * math.cos(phi)
        local y1 = pos[2] + Rsurf * math.sin(t) * math.cos(phi)
        local z1 = pos[3] + Rsurf * math.sin(phi)
        local s1 = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {sd, sd, sd}, 0)
        sim.setObjectPosition(s1, {x1, y1, z1}, W)
        sim.setShapeColor(s1, nil, sim.colorcomponent_ambient_diffuse, seamC)
        sim.setObjectInt32Param(s1, sim.shapeintparam_respondable, 0)
        sim.setObjectInt32Param(s1, sim.shapeintparam_static, 0)
        table.insert(parts, s1)

        -- 缝线 2 (绕 Z 轴旋转 90°)
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

    -- 合并为复合体
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

-- ── 清理旧对象 ────────────────────────────────────────────────
local oldPrefixes = {
    "Court_", "Line_", "Net_", "Fence_", "Light_",
    "Bench_", "Bin_", "Outer_", "Ground_", "TennisBall_",
}
local function cleanOld()
    local allObj = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for _, h in ipairs(allObj) do
        local ok, alias = pcall(sim.getObjectAlias, h, 0)
        if ok and alias then
            for _, prefix in ipairs(oldPrefixes) do
                if alias:sub(1, #prefix) == prefix then
                    pcall(sim.removeObjects, {h})
                    break
                end
            end
        end
    end
end
cleanOld()

-- ══════════════════════════════════════════════════════════════
--  真实尺寸定义 (单位: 米, 1:1 无缩放)
--
--  坐标约定:
--    X 轴 = 球场长轴 (底线到底线方向)
--    Y 轴 = 球场宽轴 (边线到边线方向)
--    原点 = 球场正中心 (球网处)
-- ══════════════════════════════════════════════════════════════
local CL   = 23.77   -- 球场长: 底线到底线
local SW   = 8.23    -- 单打宽
local DW   = 10.97   -- 双打宽
local SLD  = 6.40    -- 发球线距网距离
local NH   = 0.914   -- 球网中心高度
local NPH  = 1.067   -- 球网柱高度

-- 缓冲区 (ITF 推荐)
local RUN_BACK = 6.40
local RUN_SIDE = 3.66

local OL = CL + RUN_BACK * 2   -- 外场总长 ≈36.57m
local OW = DW + RUN_SIDE * 2   -- 外场总宽 ≈18.29m
local FL = OL + 2.0            -- 围栏长 ≈38.57m
local FW = OW + 2.0            -- 围栏宽 ≈20.29m

-- ── 颜色 ─────────────────────────────────────────────────────
local C = {
    ground     = {0.35, 0.42, 0.32},
    outer      = {0.45, 0.63, 0.40},
    inner      = {0.22, 0.45, 0.72},
    line       = {0.96, 0.96, 0.96},
    net_post   = {0.75, 0.78, 0.75},
    net_band   = {0.92, 0.92, 0.92},
    net_mesh   = {0.12, 0.12, 0.12},
    fence_post = {0.18, 0.40, 0.22},
    fence_mesh = {0.20, 0.45, 0.25},
    fence_rail = {0.22, 0.48, 0.28},
    light_pole = {0.50, 0.52, 0.50},
    light_head = {0.60, 0.62, 0.58},
    light_lamp = {0.95, 0.95, 0.85},
    bin_body   = {0.25, 0.55, 0.30},
    bin_rim    = {0.80, 0.82, 0.78},
    bench      = {0.20, 0.45, 0.25},
}

-- ══════════════════════════════════════════════════════════════
--  ① 地面: 基底 → 外场(绿) → 内场(蓝)
-- ══════════════════════════════════════════════════════════════
box("Ground_Base",   {0, 0, -0.06}, {FL + 4, FW + 4, 0.06}, C.ground, true)
box("Outer_Court",   {0, 0, -0.02}, {OL, OW, 0.03},         C.outer,  true)
box("Court_Surface", {0, 0, -0.002},{CL, DW, 0.02},          C.inner,  true)

-- ══════════════════════════════════════════════════════════════
--  ② 白线 (ITF 标准, 线宽 5cm)
-- ══════════════════════════════════════════════════════════════
local LT = 0.05
local LE = 0.005
local LZ = 0.008
local LC = C.line

-- 底线: X = ±CL/2, 平行 Y, 长度 = DW
box("Line_Baseline_E",   { CL/2, 0, LZ}, {LT, DW, LE}, LC, true)
box("Line_Baseline_W",   {-CL/2, 0, LZ}, {LT, DW, LE}, LC, true)

-- 双打边线: Y = ±DW/2, 平行 X, 长度 = CL
box("Line_Double_N",     {0,  DW/2, LZ}, {CL, LT, LE}, LC, true)
box("Line_Double_S",     {0, -DW/2, LZ}, {CL, LT, LE}, LC, true)

-- 单打边线: Y = ±SW/2, 平行 X, 长度 = CL
box("Line_Single_N",     {0,  SW/2, LZ}, {CL, LT, LE}, LC, true)
box("Line_Single_S",     {0, -SW/2, LZ}, {CL, LT, LE}, LC, true)

-- 发球线: X = ±SLD, 平行 Y, 长度 = SW
box("Line_Service_E",    { SLD, 0, LZ}, {LT, SW, LE}, LC, true)
box("Line_Service_W",    {-SLD, 0, LZ}, {LT, SW, LE}, LC, true)

-- 中线: 连接两条发球线
box("Line_Center",       {0, 0, LZ}, {SLD * 2, LT, LE}, LC, true)

-- 底线中心标记 (15cm 短线)
box("Line_CenterMark_E", { CL/2 - 0.075, 0, LZ}, {0.15, LT, LE}, LC, true)
box("Line_CenterMark_W", {-CL/2 + 0.075, 0, LZ}, {0.15, LT, LE}, LC, true)

-- ══════════════════════════════════════════════════════════════
--  ③ 球网
-- ══════════════════════════════════════════════════════════════
local netW = DW + 0.914 * 2   -- ≈12.80m

-- 网柱
cyl("Net_Post_N", {0,  netW/2, NPH/2}, nil, 0.04, NPH, C.net_post, true)
cyl("Net_Post_S", {0, -netW/2, NPH/2}, nil, 0.04, NPH, C.net_post, true)

-- 网柱顶帽
cyl("Net_Cap_N",  {0,  netW/2, NPH + 0.015}, nil, 0.05, 0.03, C.net_post, true)
cyl("Net_Cap_S",  {0, -netW/2, NPH + 0.015}, nil, 0.05, 0.03, C.net_post, true)

-- 网顶白带
box("Net_Band", {0, 0, NH + 0.025}, {0.04, netW, 0.06}, C.net_band, true)

-- 网体横线
local meshRows = 8
for i = 1, meshRows do
    local z = 0.10 + (NH - 0.15) * (i - 1) / (meshRows - 1)
    box(string.format("Net_HLine_%d", i),
        {0, 0, z}, {0.004, netW - 0.10, 0.012}, C.net_mesh, true)
end

-- 网体竖线 (约每 32cm 一根)
local vCount = 40
for i = 1, vCount do
    local y = -netW/2 + 0.10 + (netW - 0.20) * (i - 1) / (vCount - 1)
    box(string.format("Net_VLine_%d", i),
        {0, y, NH/2 + 0.02}, {0.004, 0.004, NH - 0.08}, C.net_mesh, true)
end

-- 网顶钢丝
cyl("Net_Cable", {0, 0, NH + 0.055}, {math.pi/2, 0, 0}, 0.005, netW, C.net_post, true)
-- 球网碰撞墙 (不可见但阻挡通过)
local netWall = box("Net_Collision_Wall", {0, 0, NH/2}, {0.08, netW, NH}, {0,0,0}, true)
sim.setObjectInt32Param(netWall, sim.objintparam_visibility_layer, 0)

-- ══════════════════════════════════════════════════════════════
--  ④ 围栏 (绿色, 高 3m)
-- ══════════════════════════════════════════════════════════════
local FH  = 3.0
local FpR = 0.04

local function fencePost(name, x, y)
    cyl(name, {x, y, FH/2}, nil, FpR, FH, C.fence_post, true)
    cyl(name.."_Cap", {x, y, FH + 0.02}, nil, FpR + 0.008, 0.04, C.fence_post, true)
end

local nPostsLong = math.floor(FL / 3) + 1
for i = 0, nPostsLong - 1 do
    local x = -FL/2 + i * (FL / (nPostsLong - 1))
    fencePost(string.format("Fence_Post_N%02d", i+1), x,  FW/2)
    fencePost(string.format("Fence_Post_S%02d", i+1), x, -FW/2)
end

local nPostsShort = math.floor(FW / 3) + 1
for i = 1, nPostsShort - 2 do
    local y = -FW/2 + i * (FW / (nPostsShort - 1))
    fencePost(string.format("Fence_Post_E%02d", i),  FL/2, y)
    fencePost(string.format("Fence_Post_W%02d", i), -FL/2, y)
end

-- 围栏网片
box("Fence_Mesh_N", {0,  FW/2, FH/2}, {FL, 0.02, FH}, C.fence_mesh, true)
box("Fence_Mesh_S", {0, -FW/2, FH/2}, {FL, 0.02, FH}, C.fence_mesh, true)
box("Fence_Mesh_E", { FL/2, 0, FH/2}, {0.02, FW, FH}, C.fence_mesh, true)
box("Fence_Mesh_W", {-FL/2, 0, FH/2}, {0.02, FW, FH}, C.fence_mesh, true)

-- 顶部横杆
cyl("Fence_Rail_Top_N", {0,  FW/2, FH}, {0, 0, math.pi/2}, 0.02, FL, C.fence_rail, true)
cyl("Fence_Rail_Top_S", {0, -FW/2, FH}, {0, 0, math.pi/2}, 0.02, FL, C.fence_rail, true)
cyl("Fence_Rail_Top_E", { FL/2, 0, FH}, {math.pi/2, 0, 0}, 0.02, FW, C.fence_rail, true)
cyl("Fence_Rail_Top_W", {-FL/2, 0, FH}, {math.pi/2, 0, 0}, 0.02, FW, C.fence_rail, true)

-- 底部横杆
cyl("Fence_Rail_Bot_N", {0,  FW/2, 0.12}, {0, 0, math.pi/2}, 0.015, FL, C.fence_rail, true)
cyl("Fence_Rail_Bot_S", {0, -FW/2, 0.12}, {0, 0, math.pi/2}, 0.015, FL, C.fence_rail, true)
cyl("Fence_Rail_Bot_E", { FL/2, 0, 0.12}, {math.pi/2, 0, 0}, 0.015, FW, C.fence_rail, true)
cyl("Fence_Rail_Bot_W", {-FL/2, 0, 0.12}, {math.pi/2, 0, 0}, 0.015, FW, C.fence_rail, true)

-- ══════════════════════════════════════════════════════════════
--  ⑤ 灯柱 (四角, 高 8m)
-- ══════════════════════════════════════════════════════════════
local lightH = 8.0
local lOff   = 1.0
local lightPos = {
    { FL/2 - lOff,  FW/2 - lOff},
    { FL/2 - lOff, -FW/2 + lOff},
    {-FL/2 + lOff,  FW/2 - lOff},
    {-FL/2 + lOff, -FW/2 + lOff},
}
for i, lp in ipairs(lightPos) do
    cyl(string.format("Light_Pole_%d", i),
        {lp[1], lp[2], lightH/2}, nil, 0.06, lightH, C.light_pole, true)
    box(string.format("Light_Head_%d", i),
        {lp[1], lp[2], lightH + 0.10}, {0.50, 0.25, 0.08}, C.light_head, true)
    box(string.format("Light_Lamp_%d", i),
        {lp[1], lp[2], lightH + 0.04}, {0.45, 0.22, 0.03}, C.light_lamp, true)
end

-- ══════════════════════════════════════════════════════════════
--  ⑥ 长椅
-- ══════════════════════════════════════════════════════════════
local function makeBench(idx, x, y)
    box(string.format("Bench_Seat_%d", idx),
        {x, y, 0.45}, {1.5, 0.40, 0.05}, C.bench, true)
    box(string.format("Bench_Leg1_%d", idx),
        {x - 0.55, y, 0.22}, {0.05, 0.38, 0.44}, C.bench, true)
    box(string.format("Bench_Leg2_%d", idx),
        {x + 0.55, y, 0.22}, {0.05, 0.38, 0.44}, C.bench, true)
    box(string.format("Bench_Back_%d", idx),
        {x, y - 0.18, 0.70}, {1.5, 0.04, 0.45}, C.bench, true)
end

makeBench(1,  0, -OW/2 + 0.8)
makeBench(2,  0,  OW/2 - 0.8)

-- ══════════════════════════════════════════════════════════════
--  ⑦ 回收仓
-- ══════════════════════════════════════════════════════════════
local BX, BY = CL/2 + 2.5, OW/2 - 2.0

box("Bin_Base",    {BX, BY, 0.02},                {0.80, 0.80, 0.03}, C.bin_body, true)
box("Bin_Back",    {BX + 0.39, BY, 0.25},         {0.04, 0.80, 0.46}, C.bin_body, true)
box("Bin_L",       {BX, BY + 0.39, 0.25},         {0.80, 0.04, 0.46}, C.bin_body, true)
box("Bin_R",       {BX, BY - 0.39, 0.25},         {0.80, 0.04, 0.46}, C.bin_body, true)
box("Bin_Front_L", {BX - 0.39, BY + 0.22, 0.14}, {0.04, 0.30, 0.24}, C.bin_body, true)
box("Bin_Front_R", {BX - 0.39, BY - 0.22, 0.14}, {0.04, 0.30, 0.24}, C.bin_body, true)
box("Bin_Rim",     {BX, BY, 0.48},                {0.84, 0.84, 0.02}, C.bin_rim, true)

local bd = sim.createDummy(0.08)
sim.setObjectPosition(bd, {BX - 0.50, BY, 0.25}, W)
sim.setObjectAlias(bd, "Bin_Entry")

-- ══════════════════════════════════════════════════════════════
--  ⑧ 网球 (12 个, 含缝线纹理)
-- ══════════════════════════════════════════════════════════════
math.randomseed(42)
local ballR = 0.0335
print("🎾 正在生成仿真缝线网球...")

for i = 1, 12 do
    local bx, by
    if i <= 6 then
        bx = (math.random() - 0.5) * (CL - 2.0)
        by = (math.random() - 0.5) * (SW - 1.0)
    elseif i <= 9 then
        bx = (math.random() > 0.5 and 1 or -1) * (CL/2 - math.random() * 3.0)
        by = (math.random() - 0.5) * (DW - 1.0)
    else
        bx = (math.random() - 0.5) * 6.0
        by = (math.random() - 0.5) * (SW - 0.5)
    end

    createTennisBall(
        string.format("TennisBall_%02d", i),
        {bx, by, ballR + 0.003}
    )

    if i % 4 == 0 then
        print(string.format("    已生成 %d/12 个网球...", i))
    end
end

-- ══════════════════════════════════════════════════════════════
--  完成
-- ══════════════════════════════════════════════════════════════
print("═══════════════════════════════════════════════════════")
print("✅  真实比例网球场景 v7 生成完毕 (1:1 无缩放)")
print("─────────────────────────────────────────────────────")
print(string.format("  🏟️  球场:  %.2fm × %.2fm", CL, DW))
print(string.format("  🟢  外场:  %.1fm × %.1fm", OL, OW))
print(string.format("  🔲  围栏:  %.1fm × %.1fm  高 %.1fm", FL, FW, FH))
print(string.format("  🥅  球网:  高 %.3fm  宽 %.2fm", NH, netW))
print(string.format("  🎾  网球:  直径 %.1fmm × 12 个 (含缝线)", ballR * 2 * 1000))
print(string.format("  💡  灯柱:  %.1fm × 4 根", lightH))
print(string.format("  📦  回收仓: (%.1f, %.1f)", BX, BY))
print("─────────────────────────────────────────────────────")
print("  📌  手动加载 KUKA YouBot: Model Browser → robots → mobile")
print("  📌  File → Save Scene As → tennis_court_v7.ttt")
print("═══════════════════════════════════════════════════════")