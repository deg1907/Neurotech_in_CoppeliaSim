# test_bbox.py — запустить пока открыта симуляция
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
client = RemoteAPIClient()
sim = client.require('sim')
h = sim.getObject('/gaika')
print(dir(sim))  # покажет все доступные методы
try:
    print(sim.getShapeBoundingBox(h))
except Exception as e:
    print(f'getShapeBoundingBox: {e}')
