-- ============================================================
--  ball_manager.lua  (v2)
--  Attached to youBot as simulation script (Non-threaded)
--
--  Features:
--  1. Detect YouBot base collision with tennis balls -> ball disappears
--  2. Track carried ball count (max 5)
--  3. When robot reaches bin with carried balls -> score points
--  4. Communicate with Python via integer signals
--  5. Reset only when Python explicitly sets episode_reset = 1
-- ============================================================

sim = require("sim")

-- Parameters
local NUM_BALLS   = 15
local PICKUP_DIST = 0.35
local BIN_X       = 4.5
local BIN_Y       = 2.5
local BIN_DIST    = 0.6
local MAX_CARRY   = 5

-- State variables
local carried     = 0
local totalScored = 0
local ballHandles = {}
local ballActive  = {}
local robotHandle = nil
local lastResetSignal = 0  -- track last reset signal value to avoid repeat reset

function sysCall_init()
    -- Get youBot base handle
    robotHandle = sim.getObject('/youBot')

    -- Find all tennis balls
    local found = 0
    for i = 1, NUM_BALLS do
        local name = string.format('/TennisBall_%02d', i)
        local h = sim.getObject(name, {noError=true})
        if h and h >= 0 then
            ballHandles[i] = h
            ballActive[i]  = true
            found = found + 1
        end
    end

    -- Initialize signals
    sim.setInt32Signal('carried_balls',   0)
    sim.setInt32Signal('scored_balls',    0)
    sim.setInt32Signal('remaining_balls', found)
    sim.setInt32Signal('episode_reset',   0)

    lastResetSignal = 0

    print('[BallManager] Init complete. Balls found: ' .. found)
end

function sysCall_actuation()
    -- Check reset signal from Python
    -- Only reset when signal changes from 0 to 1 (rising edge)
    local resetSignal = sim.getInt32Signal('episode_reset')
    if resetSignal == 1 and lastResetSignal == 0 then
        resetEpisode()
        sim.setInt32Signal('episode_reset', 0)
    end
    lastResetSignal = resetSignal

    -- Get robot position
    local rPos = sim.getObjectPosition(robotHandle, -1)
    local rx   = rPos[1]
    local ry   = rPos[2]

    -- Check ball pickup
    if carried < MAX_CARRY then
        for i = 1, NUM_BALLS do
            if ballHandles[i] and ballActive[i] then
                local bPos = sim.getObjectPosition(ballHandles[i], -1)
                local dx   = rx - bPos[1]
                local dy   = ry - bPos[2]
                local dist = math.sqrt(dx*dx + dy*dy)

                if dist < PICKUP_DIST then
                    -- Hide ball (move out of scene)
                    sim.setObjectPosition(ballHandles[i], -1, {99, 99, -1})
                    sim.setObjectInt32Param(ballHandles[i], sim.shapeintparam_static, 1)
                    ballActive[i] = false
                    carried       = carried + 1
                    print('[BallManager] Picked up ball ' .. i .. ', carrying: ' .. carried)
                end
            end
        end
    end

    -- Check bin deposit
    if carried > 0 then
        local dx   = rx - BIN_X
        local dy   = ry - BIN_Y
        local dist = math.sqrt(dx*dx + dy*dy)
        if dist < BIN_DIST then
            totalScored = totalScored + carried
            print('[BallManager] Deposited ' .. carried .. ' balls. Total score: ' .. totalScored)
            carried = 0
        end
    end

    -- Count remaining balls
    local remaining = 0
    for i = 1, NUM_BALLS do
        if ballActive[i] then
            remaining = remaining + 1
        end
    end

    -- Update signals for Python
    sim.setInt32Signal('carried_balls',   carried)
    sim.setInt32Signal('scored_balls',    totalScored)
    sim.setInt32Signal('remaining_balls', remaining)
end

function resetEpisode()
    -- Reset all balls to random positions on court
    math.randomseed(math.floor(sim.getSimulationTime() * 1000))
    for i = 1, NUM_BALLS do
        if ballHandles[i] then
            local bx = (math.random() - 0.5) * 7.0
            local by = (math.random() - 0.5) * 3.0
            sim.setObjectPosition(ballHandles[i], -1, {bx, by, 0.033})
            sim.setObjectInt32Param(ballHandles[i], sim.shapeintparam_static, 0)
            ballActive[i] = true
        end
    end

    -- Reset state
    carried     = 0
    totalScored = 0

    -- Reset signals
    sim.setInt32Signal('carried_balls',   0)
    sim.setInt32Signal('scored_balls',    0)
    sim.setInt32Signal('remaining_balls', NUM_BALLS)

    print('[BallManager] Episode reset complete')
end