import mujoco
import mujoco_viewer
import dm_control.mujoco
import numpy as np
from dm_control import mjcf
import random
import matplotlib.pyplot as plt
import Cheetah

numChildren = 10
numGenerations = 300

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
        childrenArray = np.append(childrenArray, Cheetah.Cheetah())
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

# Create a plot of the distances
p1 = plt.figure(1)
plt.plot(bestPerGeneration, label="Best")
plt.plot(averagePerGeneration, label="Average")
plt.plot(bestDistances, label="Best Distance")
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

# Save the distances of each generation to a file
with open("parameters.txt", "w") as file:
    for i in range(generationalDistances.shape[0]):
        file.write(f"Generation: {i}\n")
        row = generationalDistances[i]
        for j in range(len(row)):
            file.write(f"\tChild {j}: {row[j]}\n")
        file.write("\n")

        file.write(f"Best distance: {bestPerGeneration[i]}\n\n")

        file.write(f"Best legLength: {bestLengths[i]}\n")
        file.write(f"Best speed: {bestSpeeds[i]}\n")
        file.write(f"Best gear: {bestGears[i]}\n")
        file.write(f"Best numLeftLegs: {bestNumLeftLegs[i]}\n")
        file.write(f"Best numRightLegs: {bestNumRightLegs[i]}\n\n")

    file.close()