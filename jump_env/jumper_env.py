import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
from icecream import ic
import time
import pickle
from datetime import datetime
import os

from jump_model import RobotStates, JumpModel
import jump_ml_fcns


class JumperEnv(gym.Env):
    def __init__(self, render=False, render_every=False, render_interval=10, log_interval=10):
        super(JumperEnv, self).__init__()

        self.render_every = render_every
        self.render_interval = render_interval

        self.robot_states = RobotStates()
        self.robot_states.dqr = np.zeros((3, 1), dtype=np.float64)
        self.robot_mdl = JumpModel(_robot_states=self.robot_states)
        self.robot_ml_fcns = jump_ml_fcns.JumpMLFcns(_robot_states=self.robot_states)

        self._last_frame_time = 0.0
        self._time_step = self.robot_mdl.sim_dt
        self._is_render = render
        self.interations = self.robot_mdl.rgc_dt / self.robot_mdl.sim_dt

        # Logging setup inside "Data" folder
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.data_root = "Data"  # Main folder
        os.makedirs(self.data_root, exist_ok=True)  # Ensure "Data/" exists

        self.log_folder = os.path.join(self.data_root, f"data-{current_time}")
        os.makedirs(self.log_folder, exist_ok=True)  # Create timestamped folder

        self.ml_file_name = os.path.join(self.log_folder, f"ml_data-{current_time}")

        self.log_interval = log_interval  # Number of resets before logging starts
        self.enable_logging = False  # Logging flag
        self.logged_states = []  # Store robot states

        # Initialize the physics client
        self.physics_client = p.connect(p.GUI if self._is_render else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the plane
        self._load_plane()

        # Load the robot
        self.model, self.num_joints = self._load_robot()
        q0 = self.robot_mdl.randon_joint_pos()
        self._initialize_joint_states(q0)

        # Gym variables
        self.action_space = spaces.Discrete(self.robot_ml_fcns.NUM_ACTIONS)
        # self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=self.robot_ml_fcns.OBS_LOW_VALUE,
            high=self.robot_ml_fcns.OBS_HIGH_VALUE,
            shape=(self.robot_ml_fcns.NUM_OBS_STATES,),
            dtype=np.float64,
        )

        self.current_step = 0
        self.ep = 0
        self.q_aux = np.zeros((self.robot_mdl.JOINT_MODEL_NUM, 1), dtype=np.float64)
        self.dq_aux = np.zeros((self.robot_mdl.JOINT_MODEL_NUM, 1), dtype=np.float64)
        self.f_cont = np.zeros((len(self.robot_mdl.CONTACT_JOINTS), 1), dtype=np.float64)

        self.ml_data_list = []

    def _load_plane(self):
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self._time_step)
        plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        p.changeDynamics(plane, -1, lateralFriction=1.0)

    def _load_robot(self):
        model = p.loadURDF(self.robot_mdl.model_path, [0, 0, 0], [0, 0, 0, 1])
        number_joints = p.getNumJoints(model)
        p.changeDynamics(model, number_joints - 1, lateralFriction=1.5)
        return model, number_joints

    def _initialize_joint_states(self, q):
        for idx in range(len(q)):
            p.resetJointState(self.model, self.robot_mdl.JOINT_ST_LIST[idx], q[idx, 0])
            p.setJointMotorControl2(self.model, self.robot_mdl.JOINT_ST_LIST[idx], p.VELOCITY_CONTROL, force=0)

        self.robot_mdl.init_qr(q[3:])
        # p.enableJointForceTorqueSensor(self.model, self.num_joints - 1)

    def step(self, action):
        self.robot_mdl.new_action(action)
        # Simulate the environment for N iterations
        for _ in range(int(self.interations)):
            # read the robot variables
            self._get_model_st()
            self.robot_mdl.update_robot_states(self.q_aux, self.dq_aux, self.f_cont)

            # Append data every step if logging is enable
            if self.enable_logging:
                self.logged_states.append(self.robot_states)

            # compute the torque
            tau = self.robot_mdl.command_torque()
            # apply the torque at the joint
            p.setJointMotorControlArray(self.model, self.robot_mdl.AC_JOINT_LIST, p.TORQUE_CONTROL, forces=tau)
            # step simulation
            p.stepSimulation()
            if self._is_render:
                time_spent = time.time() - self._last_frame_time
                self._last_frame_time = time.time()
                time_to_sleep = self._time_step - time_spent
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)

        ###########################################################################################
        self._get_model_st()
        self.robot_mdl.update_robot_states(self.q_aux, self.dq_aux, self.f_cont)
        self.robot_mdl.ml_states()

        obs, reward, terminated = self.robot_ml_fcns.end_of_step()
        ###########################################################################################

        obs = np.array(obs)
        truncated = terminated
        info = {}

        self.current_step += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Save the collected states if logging is enabled
        self._save_data()

        self.robot_mdl.reset_variables()
        self.robot_ml_fcns.reset_vars()

        p.resetSimulation()

        self._load_plane()

        self.model, self.num_joints = self._load_robot()

        q = self.robot_mdl.randon_joint_pos()
        self._initialize_joint_states(q)

        if self.render_every:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        else:
            if self.ep % self.render_interval:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            else:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        time.sleep(0.01)

        info = {"ep": self.ep, "Episod reward: ": self.robot_ml_fcns.episode_reward}

        self.ep += 1

        self.current_step = 0

        self.tau = self.robot_mdl.command_torque()

        self._get_model_st()
        self.robot_mdl.init_qr(q[3:])
        self.robot_mdl.update_robot_states(self.q_aux, self.dq_aux, self.f_cont)
        self.robot_mdl.ml_states()
        obs = self.robot_ml_fcns.observation()

        self._last_frame_time = 0
        return obs, info

    def _get_model_st(self):
        for idx in range(self.robot_mdl.JOINT_MODEL_NUM):
            self.q_aux[idx], self.dq_aux[idx], forces, tau = p.getJointState(
                self.model, self.robot_mdl.JOINT_ST_LIST[idx]
            )

        for idx in range(len(self.robot_mdl.CONTACT_JOINTS)):
            if len(p.getContactPoints(bodyA=self.model, linkIndexA=self.robot_mdl.CONTACT_JOINTS[idx] - 1)) > 0:
                self.f_cont[idx, 0] = 1
            else:
                self.f_cont[idx, 0] = 0

    def _save_data(self):
        """Saves robot states and ML data inside the structured log folder."""

        # Save Robot States (Logging)
        if self.enable_logging and self.logged_states:
            file_path = os.path.join(self.log_folder, f"robot-log-ep-{self.ep}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(self.logged_states, f)
            self.logged_states = []  # Reset state log

        # Enable/Disable Logging Based on Log Interval
        self.enable_logging = self.ep % self.log_interval == 0

        #  Save ML Data
        self.ml_data_list.append(self.robot_ml_fcns.episode_reward)
        self.ml_data_list.append(self.robot_ml_fcns.n_jumps)
        self.ml_data_list.append(self.robot_ml_fcns.ac_total_changes)
        self.ml_data_list.append(self.robot_ml_fcns.inter)

        print(f"---------- Data from episode {self.ep} ----------")
        print(f" - Ep reward: {self.robot_ml_fcns.episode_reward}")
        print(f" - Number of jumps: {self.robot_ml_fcns.n_jumps}")
        print(f" - Total action changes: {self.robot_ml_fcns.ac_total_changes}")
        print(f" - Number of interactions: {self.robot_ml_fcns.inter}")
        print("-" * (len(f"---------- Data from episode {self.ep} ----------")))

        # Since `self.ml_file_name` already includes the full path, no need for `os.path.join()`
        file_path = self.ml_file_name

        # Load existing ML data
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                try:
                    data_list = pickle.load(f)
                    if not isinstance(data_list, list):  # Ensure it's a list
                        data_list = []
                except EOFError:  # Handle empty file case
                    data_list = []
        else:
            data_list = []

        # Append new ML data
        data_list.append(self.ml_data_list)

        # Save updated ML data back to the Pickle file
        with open(file_path, "wb") as f:
            pickle.dump(data_list, f)

        # Reset ML data list for next episode
        self.ml_data_list = []
