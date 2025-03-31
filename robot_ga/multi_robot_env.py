# jumper_env_multi.py

import pybullet as p
import pybullet_data
import numpy as np
from jump_model_ga import JumpModel, RobotStates
import jump_ga_fcns
import time
from concurrent.futures import ThreadPoolExecutor
class MultiRobotEnv:

    def __init__(self, render=False, render_every=False, max_interations=2500, render_interval=10, log_interval=10):
        self.render_every = render_every
        self.render_interval = render_interval
        self._is_render = render
        self.log_interval = log_interval

        self.robots = []  # List to hold each robot's state, model, id, etc.
        self._last_frame_time = 0.0
        self._time_step = 0.001
        self.interations = 10  # Will be overridden per robot
        self.current_time = 0
        self.current_step = 0
        self.ep = 0
        self.ga_data_list = []

        self.q_aux = np.zeros((6, 1), dtype=np.float64)
        self.dq_aux = np.zeros((6, 1), dtype=np.float64)
        self.f_cont = np.zeros((2, 1), dtype=np.float64)
        
        # Connect and load terrain
        self.physics_client = p.connect(p.GUI if self._is_render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._load_plane()
    
    def _load_plane(self):
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self._time_step)
        plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        p.changeDynamics(plane, -1, lateralFriction=1.0)
    
    def preload_robots(self, num_robots):
        self.robots.clear()
        spacing = 0.5

        for i in range(num_robots):
            y_offset = i * spacing

            robot_states = RobotStates()
            model = JumpModel(robot_states)
            model_id = p.loadURDF(model.model_path, [0, y_offset, 0])

            ga_fcns = jump_ga_fcns.JumpGAFcns(robot_states)

            self.robots.append({
                "id": model_id,
                "model": model,
                "states": robot_states,
                "ga": ga_fcns,
                "gene": None,
                "reward": 0.0,
                "y_offset": y_offset
            })

    def reset_generation(self, gene_list):
        if len(self.robots) != len(gene_list):
            p.resetSimulation()
            self._load_plane()
            self.preload_robots(len(gene_list))

        for i, (robot, gene) in enumerate(zip(self.robots, gene_list)):
            model = robot["model"]
            model.set_genes(gene)
            model.reset_variables()
            robot["ga"].reset_vars()
            robot["gene"] = gene
            robot["reward"] = 0.0

            q0 = model.randon_joint_pos()
            q0[0, 0] = 0.0
            q0[1, 0] = 0.95
            q0[2, 0] = 0.0

            for idx in range(len(q0)):
                p.resetJointState(robot["id"], model.JOINT_ST_LIST[idx], q0[idx, 0])
                p.setJointMotorControl2(robot["id"], model.JOINT_ST_LIST[idx], p.VELOCITY_CONTROL, force=0)

        self.current_time = 0
        self.current_step = 0
        self.ep += 1

    def collect_rewards(self):
        all_rewards = []
        all_dones = []

        for robot in self.robots:
            ga_fcns = robot["ga"]

            # Compute reward + check termination
            reward, done = ga_fcns.end_of_step()

            # Accumulate total reward
            robot["reward"] += reward

            all_rewards.append(reward)
            all_dones.append(done)

        return all_rewards, all_dones   

    def run_generation_rollout(self, max_steps=1500):
        # Track which robots are done
        terminated = [False] * len(self.robots)
        steps = 0

        while not all(terminated) and steps < max_steps:
            self.step_all()

            _, dones = self.collect_rewards()
            terminated = [t or d for t, d in zip(terminated, dones)]  # persist True once done

            steps += 1

        # Return final total reward (fitness) per robot
        fitnesses = [robot["reward"] for robot in self.robots]
        return fitnesses
    
    def step_all(self):
        # 1. Update QR once per control step (not per sim step)
        for robot in self.robots:
            robot["model"].update_qr(self.current_time)

        # 2. Run simulation for N fine-grained steps
        N = self.interations
        for _ in range(int(N)):
            with ThreadPoolExecutor() as executor:
                # Compute all torque commands in parallel
                results = list(executor.map(self._compute_robot_torque, self.robots))

            # Apply all torques
            for model_id, joint_ids, forces in results:
                p.setJointMotorControlArray(
                    bodyUniqueId=model_id,
                    jointIndices=joint_ids,
                    controlMode=p.TORQUE_CONTROL,
                    forces=forces
                )

            # Step simulation
            p.stepSimulation()

            # Optional: Real-time sleep
            if self._is_render:
                time_spent = time.time() - self._last_frame_time
                self._last_frame_time = time.time()
                sleep_time = self._time_step - time_spent
                if sleep_time > 0:
                    time.sleep(sleep_time)

        # 3. Update time (only once per qr update)
        self.current_time += self.robots[0]["model"].qr_dt
        self.current_step += 1

    def _compute_robot_torque(self, robot):
        model = robot["model"]
        model_id = robot["id"]

        # Local copies for thread safety
        q_aux = np.zeros((model.JOINT_MODEL_NUM, 1), dtype=np.float64)
        dq_aux = np.zeros((model.JOINT_MODEL_NUM, 1), dtype=np.float64)
        f_cont = np.zeros((len(model.CONTACT_JOINTS), 1), dtype=np.float64)

        # Get joint states
        for idx in range(model.JOINT_MODEL_NUM):
            joint_id = model.JOINT_ST_LIST[idx]
            q, dq, _, _ = p.getJointState(model_id, joint_id)
            q_aux[idx, 0] = q
            dq_aux[idx, 0] = dq

        # Get contact points
        for idx, link_id in enumerate(model.CONTACT_JOINTS):
            contact = p.getContactPoints(bodyA=model_id, linkIndexA=link_id)
            f_cont[idx, 0] = 1 if contact else 0

        # Update internal robot state
        model.update_robot_states(q_aux, dq_aux, f_cont)

        # Compute torque
        tau = model.command_torque()

        return model_id, model.AC_JOINT_LIST, tau.flatten().tolist()
