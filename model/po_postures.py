import pybullet as p
import pybullet_data
import os
import numpy as np
import time
import model_matrices
from datetime import datetime
from PIL import Image

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load the plane
plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
p.changeDynamics(plane, -1, lateralFriction=1.0)

# Load the robot
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hopper.urdf")
axis_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xyz.urdf")

model = p.loadURDF(model_path, [0, 0, 0], [0, 0, 0, 1])
ax1 = p.loadURDF(axis_path, [0, 0, 0], [0, 0, 0, 1])
ax2 = p.loadURDF(axis_path, [0, 0, 0], [0, 0, 0, 1])
ax3 = p.loadURDF(axis_path, [0, 0, 0], [0, 0, 0, 1])

# Joint indices
JOINT_ST_LIST = (0, 1, 2, 4, 6, 8)

# Model Matrices
md = model_matrices.ModelMatrices()

# Joint positions (converted to radians)
joint_obj = (
    np.array(
        [
            [-32.5, 60, -25],
            [-70, 110, -40],
            [-30, 120, -45],
            [-32, 60, -10],
            [-70, 120, -50],
            # [-70, 120, -30],
            # [-32.5, 40, 20],
            # [-50, 90, -30],
        ]
    )
    * np.pi
    / 180
)

# Adjust the camera: xz plane, y pointing inside the screen
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.2])

# Create folder for images
image_folder = os.path.join(os.getcwd(), f"screenshots_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
os.makedirs(image_folder, exist_ok=True)


# Simulation loop
for idx in range(len(joint_obj)):
    print("---------------")
    md.update_robot_states(q=joint_obj[idx], dq=[0])

    md.update_kinematics()
    r = md.update_com_pos().reshape(3, 1)
    t = (md.HT_toe[0:3, 3]).reshape(3, 1)
    h = (md.HT_heel[0:3, 3]).reshape(3, 1)
    a = (md.HT_ankle[0:3, 3]).reshape(3, 1)

    theta = sum(joint_obj[idx])
    rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    b = -rot @ a

    if idx == 6:
        b = b + np.array([[0], [0], [0.2]])
    else:
        b = b + np.array([[0], [0], [0.06]])

    q = [b[0, 0], b[2, 0], theta, joint_obj[idx][0], joint_obj[idx][1], joint_obj[idx][2]]

    for j in range(len(q)):
        p.resetJointState(model, JOINT_ST_LIST[j], q[j])

    p.resetBasePositionAndOrientation(ax1, b + rot @ r, [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(ax2, b + rot @ h, [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(ax3, b + rot @ t, [0, 0, 0, 1])

    time.sleep(0.5)
    print(f"Saving screenshot {idx + 1}/{len(joint_obj)}")

    # Capture and save image
    file_path = os.path.join(image_folder, f"phase_{idx + 1:03d}.png")
    width, height, rgb, _, _ = p.getCameraImage(width=1200, height=1000)
    img = Image.fromarray(np.reshape(rgb, (height, width, 4))[:, :, :3])
    img.save(file_path)

    # Small delay to visualize properly
    time.sleep(0.5)

# print(f"All screenshots saved in: {image_folder}")

# # Keep simulation open
# input("Press Enter to exit...")
# p.disconnect()
