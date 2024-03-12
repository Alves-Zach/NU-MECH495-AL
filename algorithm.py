import mujoco
import mujoco_viewer
import dm_control.mujoco
import numpy as np
from dm_control import mjcf
import random
import matplotlib.pyplot as plt

# Creating the robot
class Cheetah:
    def __init__(self):
        self.model = mjcf.RootElement()
        self.model.option.timestep = 0.01
        self.model.visual.headlight.ambient = [0.5, 0.5, 0.5]
        self.model.worldbody.add('light', diffuse='.5 .5 .5', pos='0 0 50', dir='0 0 -1')
        self.model.worldbody.add('geom', type='plane', size='25 25 0.01')

        # Creating the abdomen
        self.abdomen = self.model.worldbody.add('body', name='abdomen', pos='0 0 1.0')
        self.abdomen.add('joint', type='free')
        self.abdomen.add('geom', type='box', size='1.0 0.3 0.3', pos='0 0 0', rgba='1.0 0.0 0.0 1')

        # Arrays of legs
        self.leftLegs = np.array([])
        self.leftJoints = np.array([])
        self.rightLegs = np.array([])
        self.rightJoints = np.array([])

        self.numLegs = 0

    # Add a left leg to the robot
    def createLeftLeg(self, size = 0.4, gear = 0.5):
        # Adding a body to the left leg array
        self.leftLegs = np.append(self.leftLegs, self.abdomen.add('body',
                                                                  name=f'leftLeg{len(self.leftLegs)}'))

        # Adding geometry to the new left leg
        self.leftLegs[-1].add('geom', type='ellipsoid', size=f'0.1 0.1 {size}', rgba='0.0 1.0 0.0 1')

        # Creating the foot for the left leg
        self.leftLegs[-1].add('geom', type='ellipsoid', size='0.1 0.1 0.1', rgba='0.0 0.0 1.0 1', pos=f'0 0 {-size}')

        # Creating a joint between the abdomen and the left leg
        self.leftJoints = np.append(self.leftJoints, self.leftLegs[-1].add('joint',
                                                                           type='hinge',
                                                                           name=f'leftLeg{len(self.leftLegs)}Joint',
                                                                           pos='0 0 0',
                                                                           axis='0 1 0',
                                                                           limited='true',
                                                                           range='-100 100'))

        # For loops to update all of the other legs that were previously added
        j = 0
        for i in self.leftLegs:
            # Moving the left leg based on the number of left legs
            i.set_attributes(pos=f'{((2.0 / (len(self.leftLegs) + 1)) * (j + 1)) - 1.0} 0.4 -0.3')
            j += 1

        j = 0
        for i in self.leftJoints:
            # Moving the left leg based on the number of left legs
            i.set_attributes(pos='0.0 0.3 0.3')
            j += 1

        self.createLeftMotor(gear)
        
        self.numLegs += 1

    # Add a right leg to the robot
    def createRightLeg(self, size = 0.4, gear = 0.5):
        # Adding a body to the right leg array
        self.rightLegs = np.append(self.rightLegs, self.abdomen.add('body',
                                                                  name=f'rightLeg{len(self.rightLegs)}'))

        # Adding geometry to the new right leg
        self.rightLegs[-1].add('geom', type='ellipsoid', size=f'0.1 0.1 {size}', rgba='0.0 1.0 0.0 1')

        # Creating the foot for the right leg
        self.rightLegs[-1].add('geom', type='ellipsoid', size='0.1 0.1 0.1', rgba='0.0 0.0 1.0 1', pos=f'0 0 {-size}')

        # Creating a joint between the abdomen and the right leg
        self.rightJoints = np.append(self.rightJoints, self.rightLegs[-1].add('joint',
                                                                              type='hinge',
                                                                              name=f'rightLeg{len(self.rightLegs)}Joint',
                                                                              pos='0 0 0',
                                                                              axis='0 1 0',
                                                                              range='-100 100'))

        # For loop to update all of the other legs that were previously added
        j = 0
        for i in self.rightLegs:
            # Moving the right leg based on the number of right legs
            i.set_attributes(pos=f'{((2.0 / (len(self.rightLegs) + 1)) * (j + 1)) - 1.0} -0.4 -0.3')
            j += 1

        j = 0
        for i in self.rightJoints:
            # Moving the right joint based on the number of right joints
            i.set_attributes(pos='0.0 -0.3 0.3')
            j += 1

        self.createRightMotor(gear)
        
        self.numLegs += 1

    # Create a motor for the left leg
    def createLeftMotor(self, gear):
        self.model.actuator.add('motor', joint=self.leftJoints[-1].name,
                                name=f'leftMotor{len(self.leftJoints)}',
                                gear=f'{gear}', ctrllimited='true',
                                ctrlrange='-90 90')
        
    # Create a motor for the right leg
    def createRightMotor(self, gear):
        self.model.actuator.add('motor', joint=self.rightJoints[-1].name,
                                name=f'rightMotor{len(self.rightJoints)}',
                                gear=f'{gear}', ctrllimited='true',
                                ctrlrange='-90 90')

numChildren = 3
numGenerations = 3

legSize = 0.45
speed = 5.0
gear = 10.0

numLeftLegs = 4 + random.randint(-1, 1)
numRightLegs = 4 + random.randint(-1, 1)

# Keeping track of features over the generations
generationalDistances = np.array([])
bestPerGeneration = np.array([])
averagePerGeneration = np.array([])
bestDistances = np.array([])
bestGears = np.array([])
bestSpeeds = np.array([])
bestLengths = np.array([])
bestNumLeftLegs = np.array([])
bestNumRightLegs = np.array([])

for g in range(numGenerations):
    # Clear distances
    distances = np.array([])
    childrenArray = np.array([])
    legLengths = np.array([])
    speeds = np.array([])
    gears = np.array([])
    numLeftLegsArray = np.array([])
    numRightLegsArray = np.array([])
    print("Generation: ", g)

    for c in range(numChildren):
        # Creating leg lengths but keeping it positive
        nextLegLength = legSize + np.random.uniform(-0.5, 0.5)
        if nextLegLength < 0.1:
            nextLegLength = 0.1

        

        # Mutating values going into each child
        childrenArray = np.append(childrenArray, Cheetah())
        legLengths = np.append(legLengths, nextLegLength)
        gears = np.append(gears, gear + np.random.uniform(-1, 1))
        speeds = np.append(speeds, speed + np.random.uniform(-1, 1))
        numLeftLegsArray = np.append(numLeftLegsArray, numLeftLegs + random.randint(-1, 1))
        numRightLegsArray = np.append(numRightLegsArray, numRightLegs + random.randint(-1, 1))

        # Making sure the creature has at least one leg per side
        if numLeftLegsArray[c] < 1:
            numLeftLegsArray[c] = 1
        if numRightLegsArray[c] < 1:
            numRightLegsArray[c] = 1

        for i in range(numLeftLegs):
            childrenArray[c].createLeftLeg(legLengths[c], gears[c])
        
        for i in range(numRightLegs):
            childrenArray[c].createRightLeg(legLengths[c], gears[c])

        cheetahXMLString = childrenArray[c].model.to_xml_string()

        # Create an XML based on the robot
        with open(f"cheetah{c}.xml", "w") as file:
            file.write(cheetahXMLString)

        mujoModel = dm_control.mujoco.MjModel.from_xml_path(f"cheetah{c}.xml")
        mujoData = dm_control.mujoco.MjData(mujoModel)

        mujoViewer = mujoco_viewer.MujocoViewer(mujoModel, mujoData)
        mujoViewer.cam.distance = 20

        # Params
        timestep = 0.01  
        distanceTravelled = 0.0
        lastPosition = np.zeros(1)
        numChild = 0

        # Simulate the creture
        for i in range(1000):
            for limb in range(childrenArray[c].numLegs):
                mujoData.ctrl[limb] = 80.0 * np.sin(speed * i * timestep)

            if (mujoViewer.is_alive):
                dm_control.mujoco.mj_step(mujoModel, mujoData)
                mujoViewer.render()
            else:
                break

            distanceTravelled += np.linalg.norm(mujoData.qpos[:1] - lastPosition)
            lastPosition = np.copy(mujoData.qpos[:1])

        # Save distance traveled to file
        distances = np.append(distances, distanceTravelled)
    
        mujoViewer.close()

    bestChild = np.argmax(distances)

    bestPerGeneration = np.append(bestPerGeneration, distances[bestChild])
    averagePerGeneration = np.append(averagePerGeneration, np.mean(distances))
    bestGears = np.append(bestGears, gears[bestChild])
    bestSpeeds = np.append(bestSpeeds, speeds[bestChild])
    bestLengths = np.append(bestLengths, legLengths[bestChild])
    bestNumLeftLegs = np.append(bestNumLeftLegs, numLeftLegsArray[bestChild])
    bestNumRightLegs = np.append(bestNumRightLegs, numRightLegsArray[bestChild])

    print(f"Best child: {bestChild} with distance: {distances[bestChild]}")
    
    if(generationalDistances.size == 0):
        generationalDistances = distances
    else:
        generationalDistances = np.vstack((generationalDistances, distances))


    # Setting next generation features based on best child
    numLeftLegs = childrenArray[bestChild].leftLegs.shape[0]
    numRightLegs = childrenArray[bestChild].rightLegs.shape[0]
    legLengths = legLengths[bestChild]
    speed = speeds[bestChild]
    gear = gears[bestChild]

# Save the distances of each generation to a file
with open("distances.txt", "w") as file:
    for i in range(generationalDistances.shape[0]):
        file.write(f"Generation: {i}\n")
        row = generationalDistances[i]
        for j in range(len(row)):
            file.write(f"\tChild {j}: {row[j]}\n")
        file.write("\n")

    file.close()

# Create a plot of the distances
p1 = plt.figure(1)
plt.plot(bestPerGeneration, label="Best")
plt.plot(averagePerGeneration, label="Average")
plt.xlabel("Generation")
plt.ylabel("Distance")
plt.title("Distance per Generation")
plt.legend()

# Create another plot of the best features of each generation
p2 = plt.figure(2)
plt.plot(bestGears, label="Gears")
plt.plot(bestSpeeds, label="Speeds")
plt.plot(bestLengths, label="Leg Lengths")
plt.plot(bestNumLeftLegs, label="Left Legs")
plt.plot(bestNumRightLegs, label="Right Legs")
plt.xlabel("Generation")
plt.ylabel("Value")
plt.title("Best Features per Generation")
plt.legend()

plt.show()