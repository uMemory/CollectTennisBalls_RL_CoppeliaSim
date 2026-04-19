-- ============================================================
--  BallSpawner Customization Script
--  Mounted Object: Bin_Entry
--  Function: Provides the spawnBalls() function for remote
--  invocation by Python via ZMQ Remote API.
-- ============================================================

function sysCall_init()
    sim = require('sim')
    W = sim.handle_world
    print("BallSpawner is ready, waiting for Python calling spawnBalls()")
end

function sysCall_nonSimulation()
end

function sysCall_beforeSimulation()
end

function sysCall_afterSimulation()
end

function sysCall_cleanup()
end

-- ===============================================================
-- Helper: ensure sim and W are initialized
-- When callScriptFunction invokes spawnBalls from outside,
-- it runs in the script's global env but sysCall_init may
-- not have populated sim/W yet. This guard fixes that.
-- ===============================================================
function _ensureInit()
    if not sim then
        sim = require('sim')
    end
    if not W then
        W = sim.handle_world
    end
end

-- ===============================================================
-- Internal: Create a tennis ball (sphere + optional seam particles)
-- Ported from Tennis_Generate.lua createTennisBall()
-- MUST be global so callScriptFunction can reach it
-- ===============================================================
function _createTennisBall(name, pos)
    _ensureInit()

    local R      = 0.1
    local mass   = 0.057
    local seamR  = 0.0018
    local A      = 0.38
    local N      = 0   -- set to 20 to enable seam thread particles

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

-- ==========================================================
-- Internal: Remove all existing TennisBall_* objects
-- MUST be global
-- ==========================================================
function _cleanBalls()
    _ensureInit()

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
        print(string.format("Cleaned %d old tennis balls.", removed))
    end
    return removed
end

-- ===============================================================
-- Internal: Generate a random spawn position for ball index i
-- MUST be global
-- ===============================================================
function _randomPos(i, total)
    local OL = 36.57
    local OW = 18.29
    local g1 = math.floor(total * 0.5)
    local g2 = math.floor(total * 0.75)
    local bx, by

    if i <= g1 then
        bx = (math.random() - 0.5) * (OL - 2.0)
        by = (math.random() - 0.5) * (OW - 1.0)
    elseif i <= g2 then
        bx = (math.random() > 0.5 and 1 or -1) * (OL/2 - math.random() * 3.0)
        by = (math.random() - 0.5) * (OW - 2.0)
    else
        bx = (math.random() - 0.5) * 10.0
        by = (math.random() - 0.5) * (OW - 2.0)
    end

    -- Avoid the net collision wall area
    if math.abs(bx) < 0.3 and math.abs(by) < 6.5 then
        bx = (bx >= 0 and 1.0 or -1.0)
    end

    return bx, by, 0.1 + 0.003
end

-- ============================================================
-- External entry point: called by Python via callScriptFunction
--
-- ZMQ Remote API signature: spawnBalls(ball_count, seed)
--   ball_count: number of balls to spawn (default 12)
--   seed:       random seed (0 or nil = use os.time())
--
-- Returns: actual generated count
-- ============================================================
function spawnBalls(inInts, inFloats, inStrings, inBuffer)
    _ensureInit()

    local ball_count, seed
    if type(inInts) == "table" then
        ball_count = inInts[1]
        seed       = inInts[2]
    else
        ball_count = inInts
        seed       = inFloats   -- ????????
    end

    ball_count = (ball_count and ball_count > 0) and ball_count or 12
    seed       = (seed and seed > 0) and seed or os.time()

    math.randomseed(seed)
    print(string.format("spawnBalls called: count=%d seed=%d", ball_count, seed))

    _cleanBalls()

    for i = 1, ball_count do
        local bx, by, bz = _randomPos(i, ball_count)
        _createTennisBall(string.format("TennisBall_%02d", i), {bx, by, bz})
        if i % 4 == 0 or i == ball_count then
            print(string.format("  generated %d/%d tennis balls...", i, ball_count))
        end
    end

    print(string.format("spawnBalls completed: %d balls", ball_count))
    return {ball_count}, {}, {}, ''
end