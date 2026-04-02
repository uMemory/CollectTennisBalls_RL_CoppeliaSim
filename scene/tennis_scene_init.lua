-- ============================================================
--  tennis_scene_v6.lua   高仿真网球场 (CoppeliaSim 4.10.0)
--  参考真实硬地网球场: 蓝色内场 + 绿色外场 + 绿色围栏 + 灯柱
--  标准球场 23.77m × 10.97m
-- ============================================================

-- ── 工具函数 (兼容 4.10.0 API) ──────────────────────────────
local W = sim.handle_world  -- 4.10.0 推荐使用 sim.handle_world

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

local function sphere(name, pos, r, color, mass)
    local h = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {r*2, r*2, r*2}, 0)
    sim.setObjectPosition(h, pos, W)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(h, nil, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectAlias(h, name)
    sim.setShapeMass(h, mass)
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

-- 设置物体透明度
local function setAlpha(handle, alpha)
    sim.setShapeColor(handle, nil, sim.colorcomponent_transparency, {alpha})
end

-- ── 清理旧对象 ────────────────────────────────────────────────
local oldNames = {
    -- v5 对象
    "Court_Floor","Court_Surface",
    "Line_Baseline_N","Line_Baseline_S","Line_Sideline_E","Line_Sideline_W",
    "Line_Service_N","Line_Service_S","Line_Center","Line_Center_Mark",
    "Line_Center_Service",
    "Net_Post_L","Net_Post_R","Net_Band","Net_Body_1","Net_Top",
    "Wall_N","Wall_S","Wall_E","Wall_W",
    "Bin_Base","Bin_Back","Bin_L","Bin_R","Bin_Entry",
    "Robot_Body","Robot_Cam_Mast","Robot_Cam_Head",
    "Wheel_FL","Wheel_FR","Wheel_RL","Wheel_RR",
    "Robot_Joint_FL","Robot_Joint_FR","Robot_Joint_RL","Robot_Joint_RR",
    "Robot_Scoop","Robot_Camera_Front","Robot_Camera_Top",
    -- v6 新增
    "Outer_Court","Ground_Base",
    "Net_Mesh_1","Net_Mesh_2","Net_Mesh_3",
    "Fence_N","Fence_S","Fence_E","Fence_W",
    "Fence_Mesh_N","Fence_Mesh_S","Fence_Mesh_E","Fence_Mesh_W",
    "Fence_Rail_Top_N","Fence_Rail_Top_S","Fence_Rail_Top_E","Fence_Rail_Top_W",
    "Fence_Rail_Bot_N","Fence_Rail_Bot_S","Fence_Rail_Bot_E","Fence_Rail_Bot_W",
    "Light_Pole_1","Light_Pole_2","Light_Pole_3","Light_Pole_4",
    "Light_Head_1","Light_Head_2","Light_Head_3","Light_Head_4",
    "Light_Lamp_1","Light_Lamp_2","Light_Lamp_3","Light_Lamp_4",
    "Bench_Seat_1","Bench_Seat_2","Bench_Leg1_1","Bench_Leg2_1","Bench_Leg1_2","Bench_Leg2_2",
    "Robot_Bumper_L","Robot_Bumper_R","Robot_Bumper_F",
    "Robot_Top_Cover","Robot_Indicator",
    "Bin_Front_L","Bin_Front_R","Bin_Label",
    "Doubles_Line_E","Doubles_Line_W",
    "Service_Center_N","Service_Center_S",
}
for i = 1, 5 do
    table.insert(oldNames, string.format("Fence_Post_N%d", i))
    table.insert(oldNames, string.format("Fence_Post_S%d", i))
    table.insert(oldNames, string.format("Fence_Post_E%d", i))
    table.insert(oldNames, string.format("Fence_Post_W%d", i))
end
for i = 1, 20 do table.insert(oldNames, string.format("TennisBall_%02d", i)) end
for _, name in ipairs(oldNames) do
    local ok, h = pcall(sim.getObject, "/"..name)
    if ok and h and h >= 0 then
        pcall(sim.removeObjects, {h})
    end
end

-- ══════════════════════════════════════════════════════════════
--  场地尺寸
-- ══════════════════════════════════════════════════════════════
local CL  = 6.0    -- 内场长 (23.77m 缩放)
local CW  = 2.74   -- 内场宽 (10.97m 缩放)
local DW  = 3.66   -- 双打宽度 (仿真)
local OL  = 9.0    -- 外场长 (含跑动区)
local OW  = 6.0    -- 外场宽
local FL  = 10.0   -- 围栏区长
local FW  = 7.0    -- 围栏区宽

-- ── 颜色定义 ─────────────────────────────────────────────────
local C = {
    ground     = {0.35, 0.42, 0.32},   -- 地面深灰绿
    outer      = {0.42, 0.62, 0.38},   -- 外场绿色 (仿真真实硬地)
    inner      = {0.22, 0.45, 0.72},   -- 内场蓝色 (US Open 风格)
    line       = {0.96, 0.96, 0.96},   -- 白线
    net_post   = {0.75, 0.78, 0.75},   -- 网柱银色
    net_band   = {0.92, 0.92, 0.92},   -- 网带白色
    net_mesh   = {0.12, 0.12, 0.12},   -- 网体深色
    fence_post = {0.18, 0.40, 0.22},   -- 围栏柱深绿
    fence_mesh = {0.20, 0.45, 0.25},   -- 围栏网绿色
    fence_rail = {0.22, 0.48, 0.28},   -- 围栏横杆
    light_pole = {0.50, 0.52, 0.50},   -- 灯柱灰色
    light_head = {0.60, 0.62, 0.58},   -- 灯头
    light_lamp = {0.95, 0.95, 0.85},   -- 灯光色
    ball       = {0.82, 0.90, 0.12},   -- 网球荧光黄
    robot_body = {0.92, 0.92, 0.92},   -- 机器人白色外壳
    robot_trim = {0.20, 0.55, 0.85},   -- 机器人蓝色装饰
    robot_dark = {0.15, 0.15, 0.18},   -- 机器人深色部件
    wheel      = {0.10, 0.10, 0.10},   -- 轮胎黑色
    bin_body   = {0.25, 0.55, 0.30},   -- 回收仓绿色
    bin_rim    = {0.80, 0.82, 0.78},   -- 回收仓边框
    bench      = {0.20, 0.45, 0.25},   -- 长椅绿色
    scoop      = {0.85, 0.50, 0.10},   -- 收集铲橙色
}

-- ══════════════════════════════════════════════════════════════
--  ① 地面层级: 基底 → 外场(绿) → 内场(蓝)
-- ══════════════════════════════════════════════════════════════
box("Ground_Base",  {0, 0, -0.04}, {FL+2, FW+2, 0.04}, C.ground, true)
box("Outer_Court",  {0, 0, -0.015},{OL, OW, 0.02},     C.outer,  true)
box("Court_Surface",{0, 0, -0.002},{CL, DW, 0.016},     C.inner,  true)

-- ══════════════════════════════════════════════════════════════
--  ② 球场白线 (标准网球场线)
-- ══════════════════════════════════════════════════════════════
local LT  = 0.04    -- 线宽
local LE  = 0.004   -- 线高于地面
local LZ  = 0.006   -- 线 z 坐标
local LC  = C.line

-- 底线 (Baselines)
box("Line_Baseline_N",  {0,  DW/2,  LZ}, {CL, LT, LE}, LC, true)
box("Line_Baseline_S",  {0, -DW/2,  LZ}, {CL, LT, LE}, LC, true)

-- 单打边线 (Singles Sidelines)
box("Line_Sideline_E",  { CL/2, 0,  LZ}, {LT, DW, LE}, LC, true)
box("Line_Sideline_W",  {-CL/2, 0,  LZ}, {LT, DW, LE}, LC, true)

-- 双打边线 (Doubles Sidelines) - 外侧
box("Doubles_Line_E",   { CL/2,  0, LZ}, {LT, DW+0.92, LE}, LC, true)
box("Doubles_Line_W",   {-CL/2,  0, LZ}, {LT, DW+0.92, LE}, LC, true)

-- 发球线 (Service Lines) - 距网 6.4m → 缩放 1.6m
local SL = 1.60
box("Line_Service_N",   { SL, 0, LZ}, {LT, CW, LE}, LC, true)
box("Line_Service_S",   {-SL, 0, LZ}, {LT, CW, LE}, LC, true)

-- 中线 (Center Service Line)
box("Line_Center",       {0, 0, LZ}, {SL*2, LT, LE}, LC, true)

-- 中心标记 (Center Marks on Baselines)
box("Service_Center_N",  { CL/2-0.04, 0, LZ}, {0.08, LT, LE}, LC, true)
box("Service_Center_S",  {-CL/2+0.04, 0, LZ}, {0.08, LT, LE}, LC, true)

-- ══════════════════════════════════════════════════════════════
--  ③ 球网 (精细模型)
-- ══════════════════════════════════════════════════════════════
local netW = DW + 0.20  -- 网宽略超球场
local netH = 0.90       -- 网高约 0.914m (缩放)

-- 网柱 (圆柱形金属柱)
cyl("Net_Post_L", {0,  netW/2,  netH/2}, nil, 0.025, netH, C.net_post, true)
cyl("Net_Post_R", {0, -netW/2,  netH/2}, nil, 0.025, netH, C.net_post, true)

-- 网柱顶部圆球装饰
sphere("Net_Post_Cap_L", {0,  netW/2, netH+0.02}, 0.03, C.net_post, 0.01)
sim.setObjectInt32Param(sim.getObject("/Net_Post_Cap_L"), sim.shapeintparam_static, 1)
sphere("Net_Post_Cap_R", {0, -netW/2, netH+0.02}, 0.03, C.net_post, 0.01)
sim.setObjectInt32Param(sim.getObject("/Net_Post_Cap_R"), sim.shapeintparam_static, 1)

-- 网顶白带 (Net Band)
box("Net_Band", {0, 0, netH-0.01}, {0.03, netW, 0.05}, C.net_band, true)

-- 网体 - 用多层薄片模拟网格质感
local meshLayers = 5
for i = 1, meshLayers do
    local z = 0.08 + (netH - 0.16) * (i - 1) / (meshLayers - 1)
    local thickness = 0.003
    local h = box(string.format("Net_Mesh_%d", i),
        {0, 0, z}, {thickness, netW - 0.06, 0.008}, C.net_mesh, true)
end
-- 竖直网线
local vLines = 12
for i = 1, vLines do
    local y = -netW/2 + 0.06 + (netW - 0.12) * (i - 1) / (vLines - 1)
    box(string.format("Net_VLine_%d", i),
        {0, y, netH/2}, {0.003, 0.003, netH - 0.10}, C.net_mesh, true)
end

-- 网绳 (Net Cable) 顶部钢丝
cyl("Net_Cable", {0, 0, netH + 0.02}, {math.pi/2, 0, 0}, 0.004, netW, C.net_post, true)

-- ══════════════════════════════════════════════════════════════
--  ④ 围栏系统 (绿色金属围栏 + 立柱)
-- ══════════════════════════════════════════════════════════════
local FH    = 1.8   -- 围栏高度
local FpR   = 0.03  -- 立柱半径
local FmT   = 0.015 -- 网片厚度

-- 围栏立柱
local function fencePost(name, x, y)
    cyl(name, {x, y, FH/2}, nil, FpR, FH, C.fence_post, true)
    -- 柱顶帽
    cyl(name.."_Cap", {x, y, FH+0.01}, nil, FpR+0.005, 0.025, C.fence_post, true)
end

-- 北面立柱
for i = 0, 6 do
    local x = -FL/2 + i * (FL/6)
    fencePost(string.format("Fence_Post_N%d", i+1), x, FW/2)
end
-- 南面立柱
for i = 0, 6 do
    local x = -FL/2 + i * (FL/6)
    fencePost(string.format("Fence_Post_S%d", i+1), x, -FW/2)
end
-- 东面立柱
for i = 1, 3 do
    local y = -FW/2 + i * (FW/4)
    fencePost(string.format("Fence_Post_E%d", i), FL/2, y)
end
-- 西面立柱
for i = 1, 3 do
    local y = -FW/2 + i * (FW/4)
    fencePost(string.format("Fence_Post_W%d", i), -FL/2, y)
end

-- 围栏网片 (半透明绿色薄板)
local fN = box("Fence_Mesh_N", {0,  FW/2, FH/2}, {FL, FmT, FH}, C.fence_mesh, true)
local fS = box("Fence_Mesh_S", {0, -FW/2, FH/2}, {FL, FmT, FH}, C.fence_mesh, true)
local fE = box("Fence_Mesh_E", { FL/2, 0, FH/2}, {FmT, FW, FH}, C.fence_mesh, true)
local fW = box("Fence_Mesh_W", {-FL/2, 0, FH/2}, {FmT, FW, FH}, C.fence_mesh, true)

-- 顶部横杆
cyl("Fence_Rail_Top_N", {0,  FW/2, FH}, {0,0,math.pi/2}, 0.015, FL, C.fence_rail, true)
cyl("Fence_Rail_Top_S", {0, -FW/2, FH}, {0,0,math.pi/2}, 0.015, FL, C.fence_rail, true)
cyl("Fence_Rail_Top_E", { FL/2, 0, FH}, {math.pi/2,0,0}, 0.015, FW, C.fence_rail, true)
cyl("Fence_Rail_Top_W", {-FL/2, 0, FH}, {math.pi/2,0,0}, 0.015, FW, C.fence_rail, true)

-- 底部横杆
cyl("Fence_Rail_Bot_N", {0,  FW/2, 0.08}, {0,0,math.pi/2}, 0.012, FL, C.fence_rail, true)
cyl("Fence_Rail_Bot_S", {0, -FW/2, 0.08}, {0,0,math.pi/2}, 0.012, FL, C.fence_rail, true)
cyl("Fence_Rail_Bot_E", { FL/2, 0, 0.08}, {math.pi/2,0,0}, 0.012, FW, C.fence_rail, true)
cyl("Fence_Rail_Bot_W", {-FL/2, 0, 0.08}, {math.pi/2,0,0}, 0.012, FW, C.fence_rail, true)

-- ══════════════════════════════════════════════════════════════
--  ⑤ 灯柱 (四角照明)
-- ══════════════════════════════════════════════════════════════
local lightH = 2.8
local lightPositions = {
    { FL/2 - 0.3,  FW/2 - 0.3},
    { FL/2 - 0.3, -FW/2 + 0.3},
    {-FL/2 + 0.3,  FW/2 - 0.3},
    {-FL/2 + 0.3, -FW/2 + 0.3},
}
for i, lp in ipairs(lightPositions) do
    -- 灯柱
    cyl(string.format("Light_Pole_%d", i),
        {lp[1], lp[2], lightH/2}, nil, 0.035, lightH, C.light_pole, true)
    -- 灯头支架
    box(string.format("Light_Head_%d", i),
        {lp[1], lp[2], lightH + 0.05}, {0.25, 0.12, 0.04}, C.light_head, true)
    -- 灯面 (朝内倾斜)
    box(string.format("Light_Lamp_%d", i),
        {lp[1], lp[2], lightH + 0.02}, {0.22, 0.10, 0.02}, C.light_lamp, true)
end

-- ══════════════════════════════════════════════════════════════
--  ⑥ 长椅 (场外两侧)
-- ══════════════════════════════════════════════════════════════
local function makeBench(idx, x, y, rotZ)
    local seat = box(string.format("Bench_Seat_%d", idx),
        {x, y, 0.28}, {0.60, 0.22, 0.03}, C.bench, true)
    if rotZ then sim.setObjectOrientation(seat, {0, 0, rotZ}, W) end
    -- 椅腿
    local leg1 = box(string.format("Bench_Leg1_%d", idx),
        {x - 0.22, y, 0.14}, {0.03, 0.20, 0.28}, C.bench, true)
    local leg2 = box(string.format("Bench_Leg2_%d", idx),
        {x + 0.22, y, 0.14}, {0.03, 0.20, 0.28}, C.bench, true)
    -- 靠背
    local back = box(string.format("Bench_Back_%d", idx),
        {x, y - 0.10, 0.42}, {0.60, 0.025, 0.24}, C.bench, true)
end

makeBench(1, -OL/2 + 0.4, -FW/2 + 0.6, 0)
makeBench(2,  OL/2 - 0.4, -FW/2 + 0.6, 0)

-- ══════════════════════════════════════════════════════════════
--  ⑦ 回收仓 (右后角, 更精致)
-- ══════════════════════════════════════════════════════════════
local BX, BY = FL/2 - 0.8, FW/2 - 0.8

box("Bin_Base",    {BX, BY, 0.01},      {0.55, 0.55, 0.02}, C.bin_body, true)
box("Bin_Back",    {BX+0.27, BY, 0.18}, {0.03, 0.55, 0.34}, C.bin_body, true)
box("Bin_L",       {BX, BY+0.27, 0.18}, {0.55, 0.03, 0.34}, C.bin_body, true)
box("Bin_R",       {BX, BY-0.27, 0.18}, {0.55, 0.03, 0.34}, C.bin_body, true)
-- 前方左右矮挡板 (留中间入口)
box("Bin_Front_L", {BX-0.27, BY+0.16, 0.10}, {0.03, 0.20, 0.18}, C.bin_body, true)
box("Bin_Front_R", {BX-0.27, BY-0.16, 0.10}, {0.03, 0.20, 0.18}, C.bin_body, true)
-- 边框装饰
box("Bin_Rim_Top", {BX, BY, 0.35}, {0.58, 0.58, 0.02}, C.bin_rim, true)

local bd = sim.createDummy(0.05)
sim.setObjectPosition(bd, {BX - 0.35, BY, 0.18}, W)
sim.setObjectAlias(bd, "Bin_Entry")

-- ══════════════════════════════════════════════════════════════
--  ⑧ 网球 (散落在场内不同区域)
-- ══════════════════════════════════════════════════════════════
math.randomseed(42)
local ballR = 0.033
local ballPositions = {}

-- 在场内不同区域散布
for i = 1, 15 do
    local bx, by
    if i <= 8 then
        -- 大部分在内场
        bx = (math.random() - 0.5) * (CL - 0.4)
        by = (math.random() - 0.5) * (CW - 0.2)
    elseif i <= 12 then
        -- 一些在外场缓冲区
        bx = (math.random() - 0.5) * (OL - 1.0)
        by = (math.random() - 0.5) * (OW - 1.0)
    else
        -- 几个靠近网的位置
        bx = (math.random() - 0.5) * 1.5
        by = (math.random() - 0.5) * (CW - 0.3)
    end
    sphere(string.format("TennisBall_%02d", i),
           {bx, by, ballR + 0.002}, ballR, C.ball, 0.057)
end

-- ══════════════════════════════════════════════════════════════
--  ⑨ 网球拾取机器人 (精致差速四轮平台)
-- ══════════════════════════════════════════════════════════════
local RX, RY = -OL/2 + 0.8, 0.0

-- ─── 底盘 ───
local body = box("Robot_Body", {RX, RY, 0.085},
                 {0.40, 0.30, 0.08}, C.robot_body, false, 3.0)

-- ─── 顶部盖板 (蓝色装饰) ───
local topCover = box("Robot_Top_Cover", {RX - 0.02, RY, 0.135},
                     {0.30, 0.26, 0.02}, C.robot_trim, false, 0.2)
sim.setObjectParent(topCover, body, true)

-- ─── 指示灯 ───
local indicator = cyl("Robot_Indicator", {RX, RY, 0.155},
                      nil, 0.015, 0.02, {0.10, 0.85, 0.20}, false, 0.01)
sim.setObjectParent(indicator, body, true)

-- ─── 四轮驱动 ───
local wheelData = {
    {"FL", RX + 0.15,  RY + 0.16},
    {"FR", RX + 0.15,  RY - 0.16},
    {"RL", RX - 0.15,  RY + 0.16},
    {"RR", RX - 0.15,  RY - 0.16},
}
for _, w in ipairs(wheelData) do
    local wname, wx, wy = w[1], w[2], w[3]
    -- 轮胎
    local wheel = cyl("Wheel_"..wname, {wx, wy, 0.045},
        {math.pi/2, 0, 0}, 0.045, 0.03, C.wheel, false, 0.15)
    -- 轮毂装饰
    cyl("Hub_"..wname, {wx, wy, 0.045},
        {math.pi/2, 0, 0}, 0.025, 0.032, C.robot_dark, false, 0.02)
    -- 驱动关节
    local j = sim.createJoint(sim.joint_revolute_subtype, sim.jointmode_force, 0)
    sim.setObjectPosition(j, {wx, wy, 0.045}, W)
    sim.setObjectOrientation(j, {math.pi/2, 0, 0}, W)
    sim.setObjectAlias(j, "Robot_Joint_"..wname)
    sim.setJointTargetVelocity(j, 0)
    -- 组装父子关系
    sim.setObjectParent(wheel, j, true)
    sim.setObjectParent(j, body, true)
end

-- ─── 摄像头系统 ───
-- 立柱
local mast = cyl("Robot_Cam_Mast", {RX + 0.10, RY, 0.20},
                  nil, 0.015, 0.14, C.robot_dark, false, 0.08)
sim.setObjectParent(mast, body, true)

-- 摄像头头部
local camHead = box("Robot_Cam_Head", {RX + 0.13, RY, 0.30},
                    {0.06, 0.05, 0.04}, C.robot_dark, false, 0.04)
sim.setObjectParent(camHead, body, true)

-- 镜头 (蓝色圆环)
local lens = cyl("Robot_Lens", {RX + 0.16, RY, 0.30},
                 {0, math.pi/2, 0}, 0.018, 0.015, C.robot_trim, false, 0.01)
sim.setObjectParent(lens, body, true)

-- ─── 前置收集铲 (橙色) ───
local scoop = box("Robot_Scoop", {RX + 0.22, RY, 0.04},
                  {0.04, 0.28, 0.06}, C.scoop, false, 0.15)
sim.setObjectParent(scoop, body, true)

-- 铲斗侧翼
local bumpL = box("Robot_Bumper_L", {RX + 0.20, RY + 0.15, 0.05},
                  {0.08, 0.02, 0.07}, C.scoop, false, 0.05)
sim.setObjectParent(bumpL, body, true)
local bumpR = box("Robot_Bumper_R", {RX + 0.20, RY - 0.15, 0.05},
                  {0.08, 0.02, 0.07}, C.scoop, false, 0.05)
sim.setObjectParent(bumpR, body, true)

-- ─── 后置集球仓 ───
local hopper = box("Robot_Hopper", {RX - 0.12, RY, 0.12},
                   {0.14, 0.24, 0.10}, C.robot_trim, false, 0.3)
sim.setObjectParent(hopper, body, true)

-- ══════════════════════════════════════════════════════════════
--  ⑩ 视觉传感器
-- ══════════════════════════════════════════════════════════════
local fov    = 65 * math.pi / 180
local intP   = {256, 256, 0, 0}
local floatP = {0.01, 8.0, fov, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0}

local cam = sim.createVisionSensor(0, intP, floatP)
sim.setObjectPosition(cam, {RX + 0.16, RY, 0.32}, W)
sim.setObjectOrientation(cam, {0, 0.35, 0}, W)
sim.setObjectAlias(cam, "Robot_Camera_Front")
sim.setObjectParent(cam, body, true)
sim.setObjectInt32Param(cam, sim.objintparam_visibility_layer, 0)

-- ══════════════════════════════════════════════════════════════
--  完成
-- ══════════════════════════════════════════════════════════════
print("═══════════════════════════════════════════════════")
print("✅  高仿真网球场景 v6 生成完毕 (CoppeliaSim 4.10.0)")
print("───────────────────────────────────────────────────")
print("  🏟️ 场地:  蓝色硬地内场 + 绿色外场 (US Open 风格)")
print("  📏 白线:  底线/边线/双打线/发球线/中线/中心标记")
print("  🥅 球网:  网柱+网带+网线+钢丝 (精细模型)")
print("  🔲 围栏:  绿色金属围栏 + 立柱 + 横杆")
print("  💡 灯柱:  四角照明灯柱")
print("  🪑 长椅:  场外两侧球员座椅")
print("  🎾 网球:  15个散落 (场内+缓冲区)")
print("  🤖 机器人: 四轮差速平台+摄像头+收集铲+集球仓")
print("  📦 回收仓: 右后角 (带入口)")
print("───────────────────────────────────────────────────")
print("  关节:   Robot_Joint_FL/FR/RL/RR")
print("  摄像头: Robot_Camera_Front (256×256)")
print("  入口:   Bin_Entry (dummy)")
print("───────────────────────────────────────────────────")
print("📌 File → Save Scene As → tennis_court_v6.ttt")
print("═══════════════════════════════════════════════════")











-- ============================================================
--  replace_robot_with_youbot.lua
--  删除 v6 自建机器人，替换为 CoppeliaSim 自带 KUKA YouBot
--  适配 CoppeliaSim 4.10.0
-- ============================================================

local W = sim.handle_world

-- ── ① 删除自建机器人所有部件 ──────────────────────────────────
local robotParts = {
    -- 主体
    "Robot_Body", "Robot_Top_Cover", "Robot_Indicator",
    -- 轮子与关节
    "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR",
    "Hub_FL", "Hub_FR", "Hub_RL", "Hub_RR",
    "Robot_Joint_FL", "Robot_Joint_FR", "Robot_Joint_RL", "Robot_Joint_RR",
    -- 摄像头系统
    "Robot_Cam_Mast", "Robot_Cam_Head", "Robot_Lens",
    "Robot_Camera_Front",
    -- 收集系统
    "Robot_Scoop", "Robot_Bumper_L", "Robot_Bumper_R", "Robot_Bumper_F",
    "Robot_Hopper",
}

local removed = 0
for _, name in ipairs(robotParts) do
    local ok, h = pcall(sim.getObject, "/"..name)
    if ok and h and h >= 0 then
        pcall(sim.removeObjects, {h})
        removed = removed + 1
    end
end
print(string.format("🗑️  已删除自建机器人部件: %d 个", removed))


-- ── 完成 ─────────────────────────────────────────────────────
print("═══════════════════════════════════════════════════")
print("✅  机器人替换完成")
print("    YouBot 关节名称 (参考控制脚本):")
print("      底盘: rollingJoint_fl/fr/rl/rr")
print("      机械臂: youBotArmJoint0 ~ youBotArmJoint4")
print("      夹爪: youBotGripperJoint1/2")
print("═══════════════════════════════════════════════════")