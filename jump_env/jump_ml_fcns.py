import numpy as np
import time
from collections import deque


class JumpMLFcns:
    def __init__(self, _robot_states):
        self.N = 15

        self.actions = deque([-1] * self.N, maxlen=self.N)
        self.transition_history = 0

        # Gym variables
        self.OBS_LOW_VALUE = 0
        self.OBS_HIGH_VALUE = 1
        self.NUM_OBS_STATES = 22
        self.NUM_ACTIONS = 8

        # The observation state are: [r, dr,th, dth, q, dq, qr, qrh, tau, ag_action, trans_history, contact_, stagnation_metric]

        self.obs_st_max = [
            100,
            2,
            5,
            5,
            np.pi,
            5,
            1.20,
            0.5,
            1.1,
            30,
            30,
            30,
            1.20 * 1.1,
            0.5 * 1.1,
            1.1 * 1.1,
            50,
            50,
            50,
            self.NUM_ACTIONS,
            self.N,
            3,
            1,
        ]
        self.obs_st_min = [
            -100,
            0.15,
            -5,
            -5,
            -np.pi,
            -5,
            -0.50,
            -2.2,
            -1.1,
            -30,
            -30,
            -30,
            -0.50 * 1.1,
            -2.2 * 1.1,
            -1.1 * 1.1,
            -50,
            -50,
            -50,
            0,
            0,
            0,
            0,
        ]

        self.alfa = np.zeros((self.NUM_OBS_STATES, self.NUM_OBS_STATES), dtype=np.float64)
        self.beta = np.zeros((self.NUM_OBS_STATES, 1), dtype=np.float64)
        self._setup_normalization()

        # Reward variables
        self.episode_reward = 0
        self.min_reward = -250

        self.max_inter = 1500
        self.inter = 0

        self.rewards = np.zeros((12, 1), dtype=np.double)

        self.stagnation_metric = 0

        self.robot_states = _robot_states

        self.foot_contact_state = 0

        self.contact_penalty_steps = 0
        self.grace_period = 5

        self.ac_total_changes = 0
        self.n_jumps = 0
        self.time_jump = 0
        self.delta_jump = 0
        self.first_landing = False
        self.jump_height_threshold = 0.1
        self.jump_time_threshold = 0.05

        self.jump_weight = 2.15 / 100
        self.jump_height_weight = 2.15 / 100
        self.transition_weight = 1.5
        self.const_violation_weight = 3
        self.prohibited_po_weight = 2.1
        self.joint_error_weight = 0.75
        self.body_weight = 3.25
        self.delta_x_weight = 100
        self.survive_weight = 2
        self.body_orietation_weight = 1.2
        self.j_val_weight = 1
        self.stagnation_penalty_weight = 1.1

        self.last_b_x = 0
        self.delta_x = 0

        self.stagnation_steps = 0  # Count of how long the agent is "stuck"
        self.stagnation_threshold = 25  # How many steps before penalty
        self.prev_states = None  # Store previous states for comparison
        # Weight for stagnation penalty

        self.first_interation = True

    def end_of_step(self):
        obs = self._observation()
        reward = self._reward()
        terminated = self._done()
        self.inter += 1

        return obs, reward, terminated

    def observation(self):
        return self._observation()

    def _observation(self):
        """This funtion is responsable to evalaute the observed states of the system.
        First, a vector is created with all the interested states, then it is normalized in the interval of 0 to 1
        The observation state are: [r, dr,th, dth, q, dq, qr, qrh, tau, ag_action, trans_history, contact_]
        """
        # append the new action to the list of last N actions and evaluate the history of the transation
        self.transition_history = self._check_transition()
        self.foot_contact_state = self.robot_states.toe_cont + self.robot_states.heel_cont * 2
        states = np.vstack(
            (
                self.robot_states.r_vel,
                self.robot_states.r_pos,
                self.robot_states.th,
                self.robot_states.dth,
                self.robot_states.q,
                self.robot_states.dq,
                self.robot_states.qr,
                self.robot_states.tau,
                self.robot_states.ag_act,
                self.transition_history,
                self.foot_contact_state,
                np.array([[self.stagnation_metric]]),
            ),
        )

        return self._normalize_states(states).reshape((self.NUM_OBS_STATES,))

    def _reward(self):
        self.rewards[:] = 0
        # Joint error reward
        # self.rewards[0] = self._compute_joint_error_reward()
        self.rewards[0] = self._po_j_val()

        # # Air time reward
        self.rewards[1] = self._compute_air_time_reward()

        # # Check if the choosed PO is aproprieted
        self.rewards[2] = self._compute_contact_penalty()

        # # Check if the PO was solved
        self.rewards[3] = self._check_rgc_violation()

        # # Knee ground collision check
        self.rewards[4] = self._check_body_ground_collision()

        # # Base position in x direction
        self.rewards[5] = self._compute_delta_x()

        # # Verify the control mode changes
        self.rewards[6] = self._transition_reward()

        self.rewards[7] = self._check_foot_high()

        self.rewards[8] = self._survive_reward()

        self.rewards[9] = self._body_orietation_reward()

        self.rewards[10] = self._stagnation_penalty()

        reward = self.rewards.sum()
        # # Check if the foot is inside the ground
        self.rewards[11] = self._foot_inside_ground(reward)
        reward = self.rewards[11]

        self.episode_reward += reward

        return reward

    def _done(self):
        if self.episode_reward <= self.min_reward:
            return True

        if self.robot_states.toe_pos[1, 0] < -0.025:
            return True

        if self.robot_states.heel_pos[1, 0] < -0.025:
            return True

<<<<<<< Updated upstream
<<<<<<< Updated upstream
        # # check if the base postion in Z direction is near the ground
=======
        # check if the base postion in Z direction is near the ground
>>>>>>> Stashed changes
=======
        # check if the base postion in Z direction is near the ground
>>>>>>> Stashed changes
        if self.robot_states.b_pos[1, 0] < 0.35:
            return True

        if (self.n_jumps + 1) * self.max_inter <= self.inter:
            return True

        # otherwise continue
        return False

    ############################

    def _setup_normalization(self):
        for idx in range(self.NUM_OBS_STATES):
            self.alfa[idx, idx] = 1 / (self.obs_st_max[idx] - self.obs_st_min[idx])
            self.beta[idx, 0] = -self.obs_st_min[idx] / (self.obs_st_max[idx] - self.obs_st_min[idx])

    def _normalize_states(self, states):
        return np.clip(
            self.alfa @ states + self.beta,
            self.OBS_LOW_VALUE,
            self.OBS_HIGH_VALUE,
        )

    def reset_vars(self):
        self.actions = deque([-1] * self.N, maxlen=self.N)
        self.first_landing = False
        self.episode_reward = 0
        self.n_jumps = 0
        self.ac_total_changes = 0
        self.rewards[:, :] = 0
        self.contact_penalty_steps = 0
        self.inter = 0
        self.first_interation = True

    #############################

    def _check_transition(self):
        self.actions.appendleft(self.robot_states.ag_act)

        if (self.actions[0] != self.actions[1]) and (self.actions[1] != -1):
            self.ac_total_changes += 1

        return sum(
            1
            for i in range(len(self.actions) - 1, 0, -1)
            if self.actions[i] != self.actions[i - 1] and self.actions[i] != -1
        )

    #############################

    # def _po_j_val(self):
    #     if np.isnan(self.robot_states.j_val[0, 0]):
    #         return 0
    #     else:
    #         return (
    #             -self.j_val_weight * self.robot_states.j_val[0, 0] / (np.sqrt(1 + self.robot_states.j_val[0, 0] ** 2))
    #         )

    def _po_j_val(self):
        j_val = self.robot_states.j_val[0, 0]
        if np.isnan(j_val):
            return 0
        return -self.j_val_weight * np.sign(j_val) * np.log1p(abs(j_val))

    def _body_orietation_reward(self):
        th = self.robot_states.th[0, 0]
        if th < -0.8 or th > 0.8:
            return -self.body_orietation_weight * 100 * (abs(th) - 0.8)
        else:
            return 0

    def _survive_reward(self):
        return self.survive_weight

    def _compute_joint_error_reward(self):
        # joint_errors = self.qr - self.q
        joint_errors = self.robot_states.qrh - self.robot_states.q
        joint_error_abs = np.abs(joint_errors)
        epsilon = 1e-6  # Avoid division issues
        return sum(np.clip(self.joint_error_weight / (joint_error_abs + epsilon), 0, 10))

    def _compute_contact_penalty(self):
        """
        Gradually penalizes the agent for maintaining contact when it shouldn't,
        based on how long it stays in that mode (e.g., actions 6 or 7).
        """
        prohibited_actions = [6, 7]
        foot_in_contact = self.robot_states.toe_cont or self.robot_states.heel_cont
        current_action = self.actions[0]

        # If contact exists and we're in a prohibited PO, increase penalty steps
        if foot_in_contact and current_action in prohibited_actions:
            self.contact_penalty_steps += 1
        else:
            self.contact_penalty_steps = 0  # Reset if conditions are fine

        # Start penalizing only after a grace period

        if self.contact_penalty_steps > self.grace_period:
            return -self.prohibited_po_weight * (self.contact_penalty_steps - self.grace_period)

        return 0

    def _transition_reward(self):
        if self.transition_history > 1:
            return -self.transition_weight * self.transition_history
        elif self.transition_history == 1:  # not ideal
            return self.transition_weight
        else:
            return 0

    def _compute_delta_x(self):
        delta_x = self.delta_x
        self.delta_x = 0
        return self.delta_x_weight * delta_x

    def _check_rgc_violation(self):
        if self.robot_states.rcg_status == 0:
            return -self.const_violation_weight
        elif self.robot_states.rcg_status == -1:
            return -self.const_violation_weight * 1.25
        return 0

    def _check_body_ground_collision(self):
        if self.robot_states.b_pos[1, 0] < 0.50:
            return -self.body_weight * (0.50 - self.robot_states.b_pos[1, 0])
        return 0

    def _check_foot_high(self):
        # Calculate foot height
        toe_height = self.robot_states.toe_pos[1, 0]
        # toe_contact = self.robot_states.toe_cont[0, 0]

        heel_height = self.robot_states.heel_pos[1, 0]
        # hell_contact = self.robot_states.heel_cont[0, 0]

        if toe_height > self.jump_height_threshold and heel_height > self.jump_height_threshold:
            foot_meam_hight = (toe_height + heel_height) / 2
            return self.jump_height_weight * np.clip(
                foot_meam_hight - self.jump_height_threshold,
                a_min=self.jump_height_threshold,
                a_max=self.jump_height_threshold * 3,
            )
        else:
            return 0

    def _foot_inside_ground(self, reward):
        toe_height = self.robot_states.toe_pos[1, 0]
        heel_height = self.robot_states.heel_pos[1, 0]

        # If the foot goes too low, heavily penalize
        if (toe_height < -0.05) or (heel_height < -0.05):
            reward = -50 - abs(reward)

        return reward

    def _compute_air_time_reward(self):
        """
        Computes the air-time reward after ensuring a valid jump.
        Now also ensures the robot reaches a minimum height.
        """
        self._track_air_time()  # Ensure air-time is properly tracked

        # Get the highest foot height during the jump
        foot_max_height = min(self.robot_states.toe_pos[1, 0], self.robot_states.heel_pos[1, 0])

        # Reward only if jump duration exceeds a threshold and height is sufficient
        if self.delta_jump > self.jump_time_threshold and foot_max_height > self.jump_height_threshold:
            self.n_jumps += 1  # Count the number of jumps
            air_time_reward = self.delta_jump * self.jump_weight + self.n_jumps
            self.delta_x = abs(self.robot_states.b_pos[0, 0] - self.last_b_x)
        else:
            air_time_reward = 0  # Ignore micro-hops or low jumps

        self.delta_jump = 0  # Reset jump duration after using it
        return air_time_reward

    def _track_air_time(self):
        """
        Track the air time of the robot based on foot contact (foot_contact_state).
        """
        if (self.foot_contact_state[0, 0] == 3) and not self.first_landing:
            self.first_landing = True

        # Start jump timer when the foot leaves the ground
        if self.first_landing and (self.foot_contact_state[0, 0] == 0) and self.time_jump == 0:
            self.time_jump = time.time()
            self.last_b_x = self.robot_states.b_pos[0, 0]

        # End jump timer and calculate delta_jump when foot touches the ground again
        if self.first_landing and (self.foot_contact_state[0, 0] != 0) and self.time_jump != 0:
            self.delta_jump = time.time() - self.time_jump
            self.time_jump = 0
            self.first_landing = False  # Reset for next jump

    def _stagnation_penalty(self):
        """
        Penalize the agent if the state AND the control mode remain unchanged for too many steps.
        Also updates the stagnation metric used in observations.
        """
        current_states = np.hstack(
            (
                self.robot_states.r_pos.flatten(),
                self.robot_states.r_vel.flatten(),
                self.robot_states.th.flatten(),
                self.robot_states.dth.flatten(),
                self.robot_states.q.flatten(),
                self.robot_states.dq.flatten(),
            )
        )

        current_mode = self.robot_states.ag_act  # Get current action mode

        # If first iteration, initialize prev_states and prev_mode
        if self.prev_states is None:
            self.prev_states = current_states
            self.prev_mode = current_mode
            self.stagnation_metric = 0  # Initialize stagnation info
            return 0  # No penalty at first step

        # Compute state difference
        delta_states = np.abs(current_states - self.prev_states)
        mode_unchanged = current_mode == self.prev_mode  # Check if mode remains unchanged

        # Check if the state AND mode have changed

        if np.max(delta_states) < 0.009 and mode_unchanged:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0  # Reset if the state or mode changes

        self.prev_states = current_states  # Update previous state
        self.prev_mode = current_mode  # Update previous mode

        # Normalize stagnation metric for observation (range 0 to 1)
        self.stagnation_metric = min(self.stagnation_steps / self.stagnation_threshold, 1.0)

        # Apply penalty if stagnation lasts too long
        if self.stagnation_steps > self.stagnation_threshold:
            return -self.stagnation_penalty_weight * (self.stagnation_steps - self.stagnation_threshold)
        else:
            return 0
