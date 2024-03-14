import mujoco
import mujoco_viewer
import dm_control.mujoco
import numpy as np
from dm_control import mjcf
import random
import matplotlib.pyplot as plt
import Cheetah

# Create a single creature
creature = Cheetah.Cheetah()

# Ask user for creatue features
legLength = float(input("Enter leg length: "))
if (legLength == 0):
    legLength = 0.3
    speed = 2.353442929327363
    gear = 15.49475551203141
    numLeftLegs = 3
    numRightLegs = 3
else: 
    speed = float(input("Enter speed: "))
    gear = float(input("Enter gear: "))
    numLeftLegs = int(input("Enter number of left legs: "))
    numRightLegs = int(input("Enter number of right legs: "))


for i in range(numLeftLegs):
    creature.createLeftLeg(legLength, gear)

for i in range(numRightLegs):
    creature.createRightLeg(legLength, gear)

# Creating the xml file
creatureString = creature.model.to_xml_string()

with open(f"XMLfiles/singleCreature.xml", "w") as f:
    f.write(creatureString)

mujoModel = dm_control.mujoco.MjModel.from_xml_string(creatureString)
mujoData = dm_control.mujoco.MjData(mujoModel)

mujoViewer = mujoco_viewer.MujocoViewer(mujoModel, mujoData)
mujoViewer.cam.distance = 20

# Params
timestep = 0.01
distanceTravelled = 0.0
lastPosition = np.zeros(1)

# Simulate the creture
for i in range(1000):
    for limb in range(creature.numLegs):
        mujoData.ctrl[limb] = 80.0 * np.sin(speed * i * timestep)

    if (mujoViewer.is_alive):
        dm_control.mujoco.mj_step(mujoModel, mujoData)
        mujoViewer.render()
    else:
        break

    distanceTravelled += np.linalg.norm(mujoData.qpos[:1] - lastPosition)
    lastPosition = np.copy(mujoData.qpos[:1])
    print(f"Distance travelled: {distanceTravelled:.2f} m")

mujoViewer.close()