-- ============================================================
--  bin_rebuild.lua
--  独立执行：删除旧球仓 → 生成新球仓
--  在 CoppeliaSim 脚本编辑器中直接运行即可
-- ============================================================

local W = sim.handle_world

-- ┌─────────────────────────────────────────────────────────┐
-- │  球场常量（与 tennis_scene_v7.lua 一致，内联避免依赖）   │
-- └─────────────────────────────────────────────────────────┘
local CL  = 23.77
local DW  = 10.97
local RUN_BACK = 6.40
local RUN_SIDE = 3.66
local OL = CL + RUN_BACK * 2   -- ≈36.57m
local OW = DW + RUN_SIDE * 2   -- ≈18.29m

-- ┌─────────────────────────────────────────────────────────┐
-- │  ① 删除旧球仓                                          │
-- └─────────────────────────────────────────────────────────┘
local removed = 0
local allObj = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
for _, h in ipairs(allObj) do
    local ok, alias = pcall(sim.getObjectAlias, h, 0)
    if ok and alias then
        if alias:sub(1, 4) == "Bin_" then
            pcall(sim.removeObjects, {h})
            removed = removed + 1
        end
    end
end
print(string.format("🗑️  已删除 %d 个旧球仓对象", removed))

-- ┌─────────────────────────────────────────────────────────┐
-- │  ② 工具函数                                             │
-- └─────────────────────────────────────────────────────────┘
local function box(name, pos, size, color, static)
    local h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectPosition(h, pos, W)
    sim.setObjectInt32Param(h, sim.shapeintparam_static, static and 1 or 0)
    sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 1)
    sim.setShapeColor(h, nil, sim.colorcomponent_ambient_diffuse, color)
    sim.setObjectAlias(h, name)
    return h
end

-- ┌─────────────────────────────────────────────────────────┐
-- │  ③ 新球仓参数                                           │
-- └─────────────────────────────────────────────────────────┘
local BIN_W     = 1.00    -- 宽度（Y 方向，开口方向）
local BIN_D     = 0.80    -- 深度（X 方向）
local BIN_WALL  = 0.46    -- 墙高
local BIN_THICK = 0.04    -- 墙厚
local GUIDE_LEN = 0.40    -- 导向墙长度

local C_body = {0.25, 0.55, 0.30}
local C_rim  = {0.80, 0.82, 0.78}

-- 紧贴外场角落（+X, +Y）
local BX = OL/2 - BIN_D/2
local BY = OW/2 - BIN_W/2

-- ┌─────────────────────────────────────────────────────────┐
-- │  ④ 生成新球仓                                           │
-- └─────────────────────────────────────────────────────────┘

-- 底板
box("Bin_Base",
    {BX, BY, 0.015},
    {BIN_D, BIN_W, 0.03},
    C_body, true)

-- 背墙（+X 侧，封闭）
box("Bin_Back",
    {BX + BIN_D/2, BY, BIN_WALL/2},
    {BIN_THICK, BIN_W, BIN_WALL},
    C_body, true)

-- 上侧墙（+Y 侧）
box("Bin_SideN",
    {BX, BY + BIN_W/2, BIN_WALL/2},
    {BIN_D, BIN_THICK, BIN_WALL},
    C_body, true)

-- 下侧墙（-Y 侧）
box("Bin_SideS",
    {BX, BY - BIN_W/2, BIN_WALL/2},
    {BIN_D, BIN_THICK, BIN_WALL},
    C_body, true)

-- 顶部边框
box("Bin_Rim",
    {BX, BY, BIN_WALL + 0.01},
    {BIN_D + 0.04, BIN_W + 0.04, 0.02},
    C_rim, true)

-- 导向墙（开口 -X 侧，喇叭形张开）
local gx1 = BX - BIN_D/2
local gx2 = gx1 - GUIDE_LEN * 0.7
local g_len = math.sqrt((GUIDE_LEN*0.7)^2 + (GUIDE_LEN*0.5)^2)

-- 上导向（向 +Y 张开）
local gy1_top = BY + BIN_W/2
local gy2_top = gy1_top + GUIDE_LEN * 0.5
local g_angle_top = math.atan2(gy2_top - gy1_top, gx2 - gx1)

local guide_n = box("Bin_GuideN",
    {(gx1+gx2)/2, (gy1_top+gy2_top)/2, BIN_WALL/2},
    {g_len, BIN_THICK, BIN_WALL},
    C_body, true)
sim.setObjectOrientation(guide_n, {0, 0, g_angle_top}, W)

-- 下导向（向 -Y 张开）
local gy1_bot = BY - BIN_W/2
local gy2_bot = gy1_bot - GUIDE_LEN * 0.5
local g_angle_bot = math.atan2(gy2_bot - gy1_bot, gx2 - gx1)

local guide_s = box("Bin_GuideS",
    {(gx1+gx2)/2, (gy1_bot+gy2_bot)/2, BIN_WALL/2},
    {g_len, BIN_THICK, BIN_WALL},
    C_body, true)
sim.setObjectOrientation(guide_s, {0, 0, g_angle_bot}, W)

-- 入口标记
local bd = sim.createDummy(0.08)
sim.setObjectPosition(bd, {gx1 - 0.50, BY, 0.25}, W)
sim.setObjectAlias(bd, "Bin_Entry")

print("═══════════════════════════════════════════════════════")
print(string.format("✅  新球仓生成完毕"))
print(string.format("    中心: (%.2f, %.2f)", BX, BY))
print(string.format("    尺寸: width=%.1fm × depth=%.1fm  墙高=%.2fm", BIN_W, BIN_D, BIN_WALL))
print(string.format("    开口朝 -X 方向，带喇叭导向墙"))
print("═══════════════════════════════════════════════════════")