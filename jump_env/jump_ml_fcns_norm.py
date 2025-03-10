import numpy as np
import time
from collections import deque
import norm_robot_states


class JumpMLFcns:
    def __init__(self, _robot_states):
        self.N = 15

        self.actions = deque([-1] * self.N, maxlen=self.N)
        self.transition_history = 0

        # Gym variables
        self.OBS_LOW_VALUE = 0
        self.OBS_HIGH_VALUE = 1
        self.NUM_OBS_STATES = 24
        self.NUM_ACTIONS = 8

        # Reward variables
        self.episode_reward = 0
        self.min_reward = -250

        self.max_inter = 1000
        self.inter = 0

        self.rewards = np.zeros((9, 1), dtype=np.double)

        # class

        # self.robot_states = _robot_states
        # State normalizer
        self.state_normalizer = norm_robot_states.NormRobotStates(_robot_states)

        # Store normalized states as an instance variable (initialized with zeros)
        self.normalized_states = self.state_normalizer.get_normalized_states()

        self.foot_contact_state = 0

        self.ac_total_changes = 0
        self.n_jumps = 0
        self.time_jump = 0
        self.delta_jump = 0
        self.first_landing = False
        self.jump_height_threshold = 0.1

        self.jump_weight = 2.15 / 100
        self.jump_height_weight = 2.15 / 100
        self.transition_weight = 1.5
        self.const_violation_weight = 3
        self.prohibited_po_weight = 3
        self.joint_error_weight = 0.75
        self.body_weight = 3.25
        self.delta_x_weight = 900

        self.last_b_x = 0

        self.first_interation = True

    def end_of_step(self):
        """Handles the end of step logic: updates observation, computes reward, checks termination."""
        obs = self._observation()
        reward = self._reward()
        terminated = self._done()
        self.inter += 1
        return obs, reward, terminated

    def observation(self):
        """Returns the current observation vector."""
        return self._observation()

    def _observation(self):
        """Updates and returns a normalized observation vector."""
        self.state_normalizer.normalize()

        # Update the stored normalized states
        self.normalized_states = self.state_normalizer.get_normalized_states()

        # Compute transition history
        self.transition_history = self._check_transition()

        # Compute foot contact state correctly
        self.foot_contact_state = self.normalized_states.toe_cont + self.normalized_states.heel_cont * 2

        # Stack states into a single observation vector
        states = np.vstack(
            (
                self.normalized_states.r_vel,
                self.normalized_states.r_pos,
                self.normalized_states.th,
                self.normalized_states.dth,
                self.normalized_states.q,
                self.normalized_states.dq,
                self.normalized_states.qr,
                self.normalized_states.qrh,
                self.normalized_states.tau,
                self.normalized_states.ag_act,
                np.array([[self.transition_history]]),  # Ensure it's a 2D array
                self.foot_contact_state / 3,
            )
        )

        # Ensure the shape matches the expected number of observation states
        return states.reshape((self.NUM_OBS_STATES,))

    def _reward(self):
        self.rewards[:] = 0
        # Joint error reward
        self.rewards[0] = self._compute_joint_error_reward()

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

        reward = self.rewards.sum()
        # # Check if the foot is inside the ground
        self.rewards[8] = self._foot_inside_ground(reward)
        reward = self.rewards[8]
        # # Check if the foot is inside the ground
        # self.rewards_action.append(self.rewards)

        self.episode_reward += reward

        return reward

    def _done(self):
        if self.episode_reward <= self.min_reward:
            return True

        if self.robot_states.toe_pos[1, 0] < -0.025:
            return True

        if self.robot_states.heel_pos[1, 0] < -0.025:
            return True

        # check if the base postion in Z direction is near the ground
        if self.robot_states.b_pos[1, 0] < 0.35:
            return True

        if (self.n_jumps + 1) * self.max_inter <= self.inter:
            return True

        # otherwise continue
        return False

    ############################

    def reset_vars(self):
        self.actions = deque([-1] * self.N, maxlen=self.N)
        self.episode_reward = 0
        self.n_jumps = 0
        self.ac_total_changes = 0
        self.rewards[:, :] = 0
        # print(self.inter)
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
    def _compute_joint_error_reward(self):
        # joint_errors = self.qr - self.q
        joint_errors = self.robot_states.qrh - self.robot_states.q
        joint_error_abs = np.abs(joint_errors)
        epsilon = 1e-6  # Avoid division issues
        return sum(np.clip(self.joint_error_weight / (joint_error_abs + epsilon), 0, 10))

    def _compute_contact_penalty(self):
        # Penalty if foot contact state and certain actions occur
        if (self.robot_states.toe_cont or self.robot_states.heel_cont) and (self.actions[0] in [5, 6]):
            return -self.prohibited_po_weight
        return 0

    def _transition_reward(self):
        if self.transition_history > 1:
            return -self.transition_weight * self.transition_history
        elif self.transition_history <= 1:
            return self.transition_weight
        else:
            return 0

    def _compute_delta_x(self):
        delta_x = abs(self.robot_states.b_pos[0, 0] - self.last_b_x)
        self.last_b_x = self.robot_states.b_pos[0, 0]
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

    def _compute_air_time_reward(self):
        # Track air time
        self._track_air_time()

        if self.delta_jump > 0.02:
            self.n_jumps += 1
            air_time_reward = self.delta_jump * self.jump_weight + self.n_jumps
            self.delta_jump = 0  # Reset delta_jump after reward
            return air_time_reward
        return 0

    def _check_foot_high(self):
        # Calculate foot height
        toe_height = self.robot_states.toe_pos[1, 0]
        toe_contact = self.robot_states.toe_cont[0, 0]

        heel_height = self.robot_states.heel_pos[1, 0]
        hell_contact = self.robot_states.heel_cont[0, 0]

        foot_meam_hight = (toe_height + heel_height) / 2

        if (toe_contact == 0) and (hell_contact == 0) and (foot_meam_hight > self.jump_height_threshold):
            # Reward is proportional to how close the foot height is to the desired height
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

    def _track_air_time(self):
        """
        Track the air time of the robot based on foot contact (foot_contact_state).
        """
        if self.foot_contact_state and not self.first_landing:
            self.first_landing = True

        # Start jump timer when the foot leaves the ground
        if self.first_landing and not self.foot_contact_state and self.time_jump == 0:
            self.time_jump = time.time()

        # End jump timer and calculate delta_jump when foot touches the ground again
        if self.first_landing and self.foot_contact_state and self.time_jump != 0:
            self.delta_jump = time.time() - self.time_jump
            self.time_jump = 0
            self.first_landing = False  # Reset for next jump
