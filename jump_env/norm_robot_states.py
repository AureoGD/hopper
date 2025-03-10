import numpy as np
from jump_model import RobotStates


class NormRobotStates:
    def __init__(self, _robot_states: RobotStates):
        self.robot_states = _robot_states
        self.robot_states_norm = RobotStates()

        # Define min and max values for each state attribute
        self.min_values = {
            "b_pos": np.array([[-100], [0]]),
            "b_vel": np.array([[-5], [-5]]),
            "r_pos": np.array([[-100], [0]]),
            "r_vel": np.array([[-5], [-5]]),
            "th": np.array([[-np.pi]]),
            "dth": np.array([[-5]]),
            "q": np.array([[-0.5], [-2.2], [-1.1]]),
            "dq": np.array([[-10], [-10], [-10]]),
            "qr": np.array([[-0.5], [-2.2], [-1.1]]),
            "dqr": np.array([[-10], [-10], [-10]]),
            "qrh": np.array([[-0.5], [-2.2], [-1.1]]),
            "tau": np.array([[-50], [-50], [-50]]),
            "toe_pos": np.array([[-100], [0]]),
            "toe_vel": np.array([[-5], [-5]]),
            "toe_cont": np.array([[0]]),
            "heel_pos": np.array([[-100], [0]]),
            "heel_vel": np.array([[-5], [-5]]),
            "heel_cont": np.array([[0]]),
            "ankle_pos": np.array([[-100], [0]]),
            "ankle_vel": np.array([[-5], [-5]]),
            "knee_pos": np.array([[-100], [-0]]),
            "knee_vel": np.array([[-5], [-5]]),
        }

        self.max_values = {
            "b_pos": np.array([[100], [2]]),
            "b_vel": np.array([[5], [5]]),
            "r_pos": np.array([[100], [2]]),
            "r_vel": np.array([[5], [5]]),
            "th": np.array([[np.pi]]),
            "dth": np.array([[5]]),
            "q": np.array([[1.4], [0.5], [1.1]]),
            "dq": np.array([[5], [5], [5]]),
            "qr": np.array([[1.4], [0.5], [1.1]]),
            "dqr": np.array([[5], [5], [5]]),
            "qrh": np.array([[1.4], [0.5], [1.1]]),
            "tau": np.array([[50], [50], [50]]),
            "toe_pos": np.array([[100], [2]]),
            "toe_vel": np.array([[5], [5]]),
            "toe_cont": np.array([[1]]),
            "heel_pos": np.array([[100], [2]]),
            "heel_vel": np.array([[5], [5]]),
            "heel_cont": np.array([[1]]),
            "ankle_pos": np.array([[100], [2]]),
            "ankle_vel": np.array([[5], [5]]),
            "knee_pos": np.array([[100], [2]]),
            "knee_vel": np.array([[5], [5]]),
        }

    def normalize(self):
        """Normalize all states automatically to range [0,1]."""
        for attr, min_val in self.min_values.items():
            max_val = self.max_values[attr]
            raw_val = getattr(self.robot_states, attr)

            # Normalize safely
            norm_val = np.clip((raw_val - min_val) / (max_val - min_val), 0, 1)
            setattr(self.robot_states_norm, attr, norm_val)

    def denormalize(self):
        """Denormalize states back to their original range."""
        for attr, min_val in self.min_values.items():
            max_val = self.max_values[attr]
            norm_val = getattr(self.robot_states_norm, attr)

            raw_val = norm_val * (max_val - min_val) + min_val
            setattr(self.robot_states, attr, raw_val)

    def get_normalized_states(self):
        """Retrieve the normalized state values."""
        return self.robot_states_norm
