import pybullet as p
import pybullet_data
import time
import os

# Connect to PyBullet in GUI mode
p.connect(p.GUI)

# Set the search path for PyBullet's default URDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the URDF file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hopper.urdf")
robot_id = p.loadURDF(model_path, [0, 0, 0], [0, 0, 0, 1])
# robot_id = p.loadURDF("hopper2.urdf")

# Get the number of joints (including links)
num_joints = p.getNumJoints(robot_id)

# Parameters
for link_index in range(num_joints):  # Loop through each link (excluding base)
    # Get dynamics info for the link
    dynamics_info = p.getDynamicsInfo(robot_id, link_index)
    mass = dynamics_info[0]
    local_inertia_pos = dynamics_info[3]  # Local position of the inertia frame
    local_inertia_ori = dynamics_info[4]  # Orientation of the inertia frame as a quaternion

    # Get the world position and orientation of the link
    link_state = p.getLinkState(robot_id, link_index)
    world_pos = link_state[4]
    world_ori = link_state[5]

    if mass != 0:
        # Debugging: Print local inertia position and orientation
        print(
            f"Link {link_index}: Local Inertia Position = {local_inertia_pos}, Local Inertia Orientation = {local_inertia_ori}"
        )

        # Convert local inertia position to world frame
        inertia_world_pos, inertia_world_ori = p.multiplyTransforms(
            world_pos, world_ori, local_inertia_pos, local_inertia_ori
        )

        # Debugging: Print transformed world inertia position and orientation
        print(
            f"Link {link_index}: World Inertia Position = {inertia_world_pos}, World Inertia Orientation = {inertia_world_ori}"
        )

        # Visualize inertia frame using a small sphere
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(
            baseVisualShapeIndex=sphere_visual, basePosition=inertia_world_pos, baseOrientation=inertia_world_ori
        )

        # Print mass and position for reference
        print(
            f"Link {link_index}: Mass = {mass}, Inertia Position = {inertia_world_pos}"
        )  # Debugging: Print local inertia position and orientation
        print(
            f"Link {link_index}: Local Inertia Position = {local_inertia_pos}, Local Inertia Orientation = {local_inertia_ori}"
        )

        # Convert local inertia position to world frame
        inertia_world_pos, inertia_world_ori = p.multiplyTransforms(
            world_pos, world_ori, local_inertia_pos, local_inertia_ori
        )

        # Debugging: Print transformed world inertia position and orientation
        print(
            f"Link {link_index}: World Inertia Position = {inertia_world_pos}, World Inertia Orientation = {inertia_world_ori}"
        )

        # Visualize inertia frame using a small sphere
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(
            baseVisualShapeIndex=sphere_visual, basePosition=inertia_world_pos, baseOrientation=inertia_world_ori
        )

        # Print mass and position for reference
        print(f"Link {link_index}: Mass = {mass}, Inertia Position = {inertia_world_pos}")

while 1:  # Simulate for 1000 steps
    p.stepSimulation()
    time.sleep(1.0 / 240.0)  # Simulate at 240 Hz (adjust if needed)

# Disconnect the PyBullet client after the simulation
p.disconnect()
