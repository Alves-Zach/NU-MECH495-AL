import mujoco
import mujoco_viewer
import dm_control.mujoco
import numpy as np
from dm_control import mjcf

# Creating the robot
class Cheetah:
    def __init__(self):
        self.model = mjcf.RootElement()
        self.model.option.timestep = 0.01
        self.model.visual.headlight.ambient = [0.5, 0.5, 0.5]
        self.model.worldbody.add('light', diffuse='.5 .5 .5', pos='0 0 5', dir='0 0 -1')
        self.model.worldbody.add('geom', type='plane', size='25 25 0.01')

        # Creating the abdomen
        self.abdomen = self.model.worldbody.add('body', name='abdomen', pos='0 0 1.0')
        self.abdomen.add('joint', type='free')
        self.abdomen.add('geom', type='ellipsoid', size='1.0 0.3 0.3', pos='0 0 0', rgba='1.0 0.0 0.0 1')

        # Arrays of legs
        self.leftLegs = np.array([])
        self.leftJoints = np.array([])
        self.rightLegs = np.array([])
        self.rightJoints = np.array([])

        self.numLegs = 0

    # Add a left leg to the robot
    def createLeftLeg(self):
        # Adding a body to the left leg array
        self.leftLegs = np.append(self.leftLegs, self.abdomen.add('body',
                                                                  name=f'leftLeg{len(self.leftLegs)}'))

        # Adding geometry to the new left leg
        self.leftLegs[-1].add('geom', type='ellipsoid', size='0.1 0.1 0.4', rgba='0.0 1.0 0.0 1')

        # Creating a joint between the abdomen and the left leg
        self.leftJoints = np.append(self.leftJoints, self.leftLegs[-1].add('joint',
                                                                           type='hinge',
                                                                           name=f'leftLeg{len(self.leftLegs)}Joint',
                                                                           pos='0 0 0',
                                                                           axis='0 1 0',
                                                                           limited='true',
                                                                           range='-270 270'))

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

        self.createLeftMotor()
        
        self.numLegs += 1

    # Add a right leg to the robot
    def createRightLeg(self):
        # Adding a body to the right leg array
        self.rightLegs = np.append(self.rightLegs, self.abdomen.add('body',
                                                                  name=f'rightLeg{len(self.rightLegs)}'))

        # Adding geometry to the new right leg
        self.rightLegs[-1].add('geom', type='ellipsoid', size='0.1 0.1 0.4', rgba='0.0 1.0 0.0 1')

        # Creating a joint between the abdomen and the right leg
        self.rightJoints = np.append(self.rightJoints, self.rightLegs[-1].add('joint',
                                                                              type='hinge',
                                                                              name=f'rightLeg{len(self.rightLegs)}Joint',
                                                                              pos='0 0 0',
                                                                              axis='0 1 0',
                                                                              range='-360 360'))

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

        self.createRightMotor()
        
        self.numLegs += 1

    # Create a motor for the left leg
    def createLeftMotor(self):
        self.model.actuator.add('motor', joint=self.leftJoints[-1].name,
                                name=f'leftMotor{len(self.leftJoints)}',
                                gear='5', ctrllimited='true',
                                ctrlrange='-90 90')
        
    # Create a motor for the right leg
    def createRightMotor(self):
        self.model.actuator.add('motor', joint=self.rightJoints[-1].name,
                                name=f'rightMotor{len(self.rightJoints)}',
                                gear='5', ctrllimited='true',
                                ctrlrange='-90 90')

cheetahModel = Cheetah()
for i in range(4):
    cheetahModel.createLeftLeg()
    cheetahModel.createRightLeg()

cheetahXMLString = cheetahModel.model.to_xml_string()

# Create an XML based on the robot
with open("cheetah.xml", "w") as file:
    file.write(cheetahXMLString)

mujoModel = dm_control.mujoco.MjModel.from_xml_path("cheetah.xml")
mujoData = dm_control.mujoco.MjData(mujoModel)

mujoViewer = mujoco_viewer.MujocoViewer(mujoModel, mujoData)

# Params
timestep = 0.01  

for i in range(100000):
    for limb in range(cheetahModel.numLegs):
        mujoData.ctrl[limb] = 80.0 * np.sin(i * timestep)

    if (mujoViewer.is_alive):
        dm_control.mujoco.mj_step(mujoModel, mujoData)
        mujoViewer.render()
    else:
        break

# Closing the viewer
mujoViewer.close()