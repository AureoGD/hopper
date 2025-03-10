import pybullet as p
import os
import sys

import pybullet as p


def visualize_frames(model_id, frame_length=0.1, frame_thickness=0.01):
    """
    Visualizes all frames of a PyBullet model by drawing coordinate axes at each link's origin.

    Parameters:
    - model_id (int): The ID of the PyBullet model.
    - frame_length (float): The length of the axes to visualize.
    - frame_thickness (float): The thickness of the axes lines.
    """
    # Visualize the base frame
    base_position, base_orientation = p.getBasePositionAndOrientation(model_id)

    # Convert orientation to rotation matrix for the base
    base_rotation_matrix = p.getMatrixFromQuaternion(base_orientation)
    base_x_axis = [base_rotation_matrix[0], base_rotation_matrix[3], base_rotation_matrix[6]]
    base_y_axis = [base_rotation_matrix[1], base_rotation_matrix[4], base_rotation_matrix[7]]
    base_z_axis = [base_rotation_matrix[2], base_rotation_matrix[5], base_rotation_matrix[8]]

    # Scale and draw base axes
    base_x_end = [base_position[i] + frame_length * base_x_axis[i] for i in range(3)]
    base_y_end = [base_position[i] + frame_length * base_y_axis[i] for i in range(3)]
    base_z_end = [base_position[i] + frame_length * base_z_axis[i] for i in range(3)]
    p.addUserDebugLine(base_position, base_x_end, [1, 0, 0], frame_thickness)  # X-axis (red)
    p.addUserDebugLine(base_position, base_y_end, [0, 1, 0], frame_thickness)  # Y-axis (green)
    p.addUserDebugLine(base_position, base_z_end, [0, 0, 1], frame_thickness)  # Z-axis (blue)

    # Visualize all link frames
    num_joints = p.getNumJoints(model_id)
    for link_index in range(num_joints):
        # Get the link's world position and orientation
        link_state = p.getLinkState(model_id, link_index, computeForwardKinematics=True)
        link_position = link_state[4]  # World position of the link
        link_orientation = link_state[5]  # World orientation of the link (quaternion)

        # Convert orientation to rotation matrix
        rotation_matrix = p.getMatrixFromQuaternion(link_orientation)
        x_axis = [rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]]
        y_axis = [rotation_matrix[1], rotation_matrix[4], rotation_matrix[7]]
        z_axis = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]

        # Scale the axes by the desired frame length
        x_end = [link_position[i] + frame_length * x_axis[i] for i in range(3)]
        y_end = [link_position[i] + frame_length * y_axis[i] for i in range(3)]
        z_end = [link_position[i] + frame_length * z_axis[i] for i in range(3)]

        # Draw the axes using PyBullet's debug line feature
        p.addUserDebugLine(link_position, x_end, [1, 0, 0], frame_thickness)  # X-axis (red)
        p.addUserDebugLine(link_position, y_end, [0, 1, 0], frame_thickness)  # Y-axis (green)
        p.addUserDebugLine(link_position, z_end, [0, 0, 1], frame_thickness)  # Z-axis (blue)


# Example usage
# Assuming PyBullet is initialized, and a model is loaded
p.connect(p.GUI)
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hopper.urdf")
model_id = p.loadURDF(model_path)  # Replace with your model path
visualize_frames(model_id)

while True:
    pass
