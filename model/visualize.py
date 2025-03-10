import pybullet as p
import pybullet_data
import os
from os.path import dirname, join, abspath
import time
from icecream import ic
import pinocchio
from sys import argv
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)


plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
p.changeDynamics(plane, -1, lateralFriction=1.0)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hopper.urdf")
model = p.loadURDF(model_path, [0, 0, 1], [0, 0, 0, 1])

# for i in range(p.getNumJoints(model)):
#     ic(p.getJointInfo(model, i))

JOINT_ST_LIST = (0, 1, 2, 4, 6, 8)

q0 = np.array([0, 0, 0, np.pi * 75 / 180, np.pi * -120 / 180, np.pi * 60 / 180])
for idx in range(len(q0)):
    p.resetJointState(model, JOINT_ST_LIST[idx], q0[idx])

while 1:
    # pass
    p.stepSimulation()
    #     # print(len(p.getContactPoints(bodyA=1, linkIndexA=3)))
    time.sleep(0.001)
