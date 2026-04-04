import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

youbot = sim.getObject('/youBot')
bot_pos = sim.getObjectPosition(youbot, sim.handle_world)
print(f"YouBot位置: {bot_pos}")

for i in range(1, 13):
    h = sim.getObject(f'/TennisBall_{i:02d}')
    pos = sim.getObjectPosition(h, sim.handle_world)
    dist = np.sqrt((bot_pos[0]-pos[0])**2 + (bot_pos[1]-pos[1])**2)
    print(f"Ball_{i:02d} 位置: {pos} | 距离: {dist:.3f}m")
