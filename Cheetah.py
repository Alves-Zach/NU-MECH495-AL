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
