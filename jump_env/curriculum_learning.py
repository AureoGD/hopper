import numpy as np
import time
from collections import deque


class JumpMLFcns:
    def __init__(self, _robot_states):
        self.N = 30

        self.actions = deque([-1] * self.N, maxlen=self.N)
        self.transition_history = 0

        # Gym variables
        self.OBS_LOW_VALUE = 0
        self.OBS_HIGH_VALUE = 1
        self.NUM_OBS_STATES = 22
        self.NUM_ACTIONS = 8

        self.dt = 0.01

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

        self.max_inter = 2000
        self.inter = 0

        self.rewards = np.zeros((12, 1), dtype=np.double)

        self.robot_states = _robot_states
        self.foot_contact_state = 0

        self.ac_total_changes = 0
        self.n_jumps = 0
        self.time_jump = 0
        self.delta_jump = 0
        self.first_landing = False
        self.jump_height_threshold = 0.1
        self.jump_time_threshold = 0.05
        self.jump_weight = 2.15 / 100
        self.jump_height_weight = 2.15 / 100
        self.const_violation_weight = 3
        self.joint_error_weight = 0.75
        self.body_weight = 3.25
        self.delta_x_weight = 100
        self.j_val_weight = 1
        self.last_b_x = 0
        self.delta_x = 0

        # Weight for stagnation penalty
        self.stagnation_metric = 0
        self.stagnation_penalty_weight = 3.5
        self.stagnation_steps = 0  # Count of how long the agent is "stuck"
        self.stagnation_threshold = 250  # How many steps before penalty
        self.prev_states = None  # Store previous states for comparison

        # Transition
        self.action_short_transition_weight = 1.25
        self.action_long_transition_weight = 0.65 * self.action_short_transition_weight
        self.max_transitions = self.N

        # minimal  mode duration
        self.min_mode_duration_weight = 1.2
        self.mode_duration = 0
        self.mode_min_duration = {0: 20, 1: 20, 2: 20, 3: 20, 4: 20, 5: 20, 6: 20, 7: 20}

        # Invalid PO for a robot configuration
        self.invalid_po_weight = 1.5
        self.invalid_po_steps = 0
        self.invalid_po_grace_period = 15
        self.prohibited_actions = [6, 7]

        # Survive reward
        self.survive_weight = 1.2

        # Body orientation reward
        self.body_orietation_weight = 2.5

        # Po violation (not solved or error to create a PO)
        self.po_violation_weight = 1.5

        # Curriculum learning variables
        # Stay up right
        self.curriculum_phase = 1
        self.phase1_success_steps = 0
        self.phase1_success_steps_threshold = 100

        # crouch
        self.standing_height = 0
        self.max_com_drop = 0.2
        self.d_crouch_depth = 0
        self.crouch_weight = 1.0

        self.first_interation = True

    def end_of_step(self):
        self.inter += 1
        obs = self._observation()
        reward = self._reward()
        terminated = self._done()

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

        # Reward for survive
        self.rewards[0] = self.survive_weight * self._survive_reward()

        # Penalize hight PO changes in short term
        self.rewards[1] = self.action_short_transition_weight * self._short_term_transition_penalty()

        # Penalize hight PO changes in long term
        self.rewards[2] = self.action_long_transition_weight * self._short_term_transition_penalty()

        # Penalize for invalid PO choice
        self.rewards[3] = self.invalid_po_weight * self._invalid_po_penalty()

        # Penalizes high body angles
        self.rewards[4] = self.body_orietation_weight * self._body_orientation_penalty()

        #  Penalizes stationary conditios for long time
        self.rewards[5] = self.stagnation_penalty_weight * self._stagnation_penalty()

        # Penalizes PO not solve
        self.rewards[6] = self.po_violation_weight * self._check_rgc_violation()

        # reward the stand phase
        self.rewards[7] = self._stand_stability_reward()

        # Reward crouch robot
        self.rewards[8] = self.crouch_weight * self._crouch_reward()

        # Penalizes chnages before a minimum duration
        self.rewards[9] = self.min_mode_duration_weight * self._minimum_mode_duration_penalty()

        # Reward for use a empirical best mode for a phase:
        self.rewards[10] = 0.5 * self._preferred_mode_bonus()

        reward = self.rewards.sum()

        curriclum_reward = self._curriculum_learning_check()

        reward += curriclum_reward

        self.episode_reward += reward

        return reward

    def _done(self):
        if self.episode_reward <= self.min_reward:
            return True

        if self.robot_states.toe_pos[1, 0] < -0.025:
            return True

        if self.robot_states.heel_pos[1, 0] < -0.025:
            return True

        # # check if the base postion in Z direction is near the ground
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
        self.invalid_po_steps = 0
        self.inter = 0
        self.first_interation = True
        self.phase1_success_steps = 0
        self.standing_height = 0
        self.curriculum_phase = 1
        self.prev_crouch_depth = 0
        self.d_crouch_depth = 0
        self.mode_duration = 0

    #############################
    def _check_transition(self):
        ac = self.robot_states.ag_act.reshape(1, 1)
        self.actions.appendleft(ac[0])

        if (self.actions[0] != self.actions[1]) and (self.actions[1] != -1):
            self.ac_total_changes += 1

        return sum(
            1
            for i in range(len(self.actions) - 1, 0, -1)
            if self.actions[i] != self.actions[i - 1] and self.actions[i] != -1
        )

    #############################

    def _survive_reward(self):
        return 1

    def _short_term_transition_penalty(self):
        """
        Penalizes recent excessive changes in PO (short-term jitteriness).
        Uses transition history over last N steps. Keeps in [-1, 1].
        """
        if self.transition_history > 1:
            return -min(self.transition_history / self.max_transitions, 1.0)
        elif self.transition_history == 1:
            return 1.0  # Single meaningful change
        return 0.0

    def _long_term_transition_penalty(self):
        """
        Penalizes high total number of action changes over time.
        Helps prevent instability over the episode. Returns [-1, 0].
        """

        change_ratio = self.ac_total_changes / self.inter
        return -min(change_ratio, 1.0)

    def _invalid_po_penalty(self):
        """
        Penalize staying in a prohibited PO (mode) while foot is in contact.
        The penalty decreases gradually and is capped at -1. Keeps reward in [-1, 0].
        """
        # TODO: reward logic can be added to give positive values for use the right PO if in contact

        if self.foot_contact_state != 0 and self.actions[0] in self.prohibited_actions:
            self.invalid_po_steps += 1
        else:
            self.invalid_po_steps = 0

        if self.invalid_po_steps > self.invalid_po_grace_period:
            max_steps = self.invalid_po_grace_period * 3
            over_steps = self.invalid_po_steps - self.invalid_po_grace_period
            normalized_penalty = -min(over_steps / (max_steps - self.invalid_po_grace_period), 1.0)
            return normalized_penalty

        return 0

    def _body_orientation_penalty(self):
        """
        Penalize the robot for exceeding the allowable body pitch orientation (in radians).

        The penalty starts when the absolute value of the body orientation angle (th)
        exceeds a defined threshold (0.8 rad). The penalty scales linearly between 0
        and -1 as the orientation approaches a maximum tolerated value (e.g., 1.5 rad).

        Returns:
            float: A negative reward (penalty) in the range [-1, 0], or 0 if within threshold.
        """
        th = abs(self.robot_states.th[0, 0])
        threshold = 0.8  # Angle (in rad) where penalty begins
        max_th = 1.2  # Maximum angle for full penalty (overflow capped)

        if th > threshold:
            overflow = th - threshold
            normalized_excess = min(overflow / (max_th - threshold), 1.0)
            return -normalized_excess  # Scaled penalty in [-1, 0]
        else:
            return 0

    def _stagnation_penalty(self):
        """
        Penalize the agent if both the robot state and control mode remain unchanged
        for too many steps. The penalty scales between 0 and -1, where -1 means the
        robot is fully stagnant beyond the threshold, and 0 means it's moving or
        switching modes frequently.

        Also updates a stagnation metric used for observation.
        """
        # Concatenate key robot state variables
        # TODO: maybe change for the base velocity
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

        current_mode = self.robot_states.ag_act

        # Initialize on the first step
        if self.prev_states is None:
            self.prev_states = current_states
            self.prev_mode = current_mode
            self.stagnation_metric = 0
            return 0

        # Calculate if change occurred
        delta_states = np.abs(current_states - self.prev_states)
        mode_unchanged = current_mode == self.prev_mode

        max_delta = np.max(delta_states)

        # Deadband logic
        if max_delta < 0.0001 and mode_unchanged:
            self.stagnation_steps += 1
        elif max_delta > 0.001 or not mode_unchanged:
            self.stagnation_steps = 0
        else:
            pass

        # if self.stagnation_steps == 1:
        #     print(f"[Stagnation detected] Step: {self.inter}")

        # Update previous state/mode
        self.prev_states = current_states
        self.prev_mode = current_mode

        # Normalize stagnation level (0 to 1)
        self.stagnation_metric = min(self.stagnation_steps / self.stagnation_threshold, 1.0)

        penalty = -1 / (1 + np.exp(-20 * (self.stagnation_metric - 0.2)))

        # Linearly scale the penalty to [-1, 0]
        # return -self.stagnation_metric
        return penalty

    def _check_rgc_violation(self):
        if self.robot_states.rcg_status == 0:
            return -1
        elif self.robot_states.rcg_status == -1:
            return -1.25
        return 0

    def _stand_stability_reward(self):
        """
        Curriculum Phase 1: Reward the agent for standing upright, with a valid PO,
        low linear and angular velocity, and a minimum CoM height.

        Reward scales from 0 to 1.
        """
        if self.curriculum_phase != 1:
            return 0
        r_z = self.robot_states.r_pos[1, 0]
        th = abs(self.robot_states.th[0, 0])
        r_vel = np.linalg.norm(self.robot_states.r_vel)
        dth = np.linalg.norm(self.robot_states.dth)
        valid_po = self.robot_states.ag_act not in self.prohibited_actions

        com_ok = r_z > 0.7
        upright = th < 0.15
        low_motion = r_vel < 0.003 and dth < 0.025  # can adjust this threshold

        if com_ok and upright and valid_po and low_motion:
            self.phase1_success_steps += 1
            return 1.0
        elif com_ok and upright and valid_po:
            return 0.5
        elif valid_po:
            return 0.2
        return 0

    def _crouch_reward(self):
        """
        Reward the robot for crouching (i.e., lowering CoM from the standing position),
        simulating the potential energy storage phase of a spring.

        Returns:
            float: Reward in range [0, 1], where 1 is max crouch.
        """
        # Make sure phase 1 was completed and reference height is stored
        if self.curriculum_phase != 2:
            return 0

        com_z = self.robot_states.r_pos[1, 0]
        crouch_depth = self.standing_com_height - com_z
        self.d_crouch_depth = abs(self.prev_crouch_depth - crouch_depth) / self.dt
        self.prev_crouch_depth = crouch_depth

        # Normalize the crouch reward [0, 1]
        if crouch_depth <= 0.001:
            return 0  # Too high (standing)
        else:
            reward = min(crouch_depth / self.max_com_drop, 1.0)
            if reward <= 0.001:
                return 0
            else:
                return reward

    def _curriculum_learning_check(self):
        if self.curriculum_phase == 1 and self.phase1_success_steps > self.phase1_success_steps_threshold:
            # print(f"End of phase 1: {self.inter}")
            self.curriculum_phase = 2
            self.standing_com_height = self.robot_states.r_pos[1, 0]
            return 100
        elif self.curriculum_phase == 2 and self.d_crouch_depth < 0.009 and self.robot_states.r_pos[1, 0] < 0.6:
            # print(f"End of phase 2: {self.inter}")
            # print(self.inter)
            self.curriculum_phase = 3
            return 100

        return 0

    def _minimum_mode_duration_penalty(self):
        """
        Penalizes if the current mode has not been maintained for at least `min_steps` steps.
        Optionally scales the penalty in [-1, 0] based on how close the agent is to meeting the requirement.

        Args:
            min_steps (int): Minimum number of steps required in the same mode before avoiding penalty.
            scaled (bool): Whether to scale penalty smoothly from -1 to 0.

        Returns:
            float: A penalty in [-1.0, 0.0]
        """
        if self.actions[0] == -1 or self.actions[1] == -1:
            return 0.0  # Ignore uninitialized states

        if self.actions[0] == self.actions[1]:
            self.mode_duration += 1
        else:
            self.mode_duration = 0

        min_steps = self.mode_min_duration.get(self.actions[0][0], 10)

        if self.mode_duration < min_steps:
            return -1.0 * (1.0 - self.mode_duration / min_steps)
        return 0

    def _preferred_mode_bonus(self):
        """
        During curriculum phase 1, give a small bonus for using empirically preferred modes.
        Helps encourage stable behaviors early in training.

        Returns:
            float: A small positive reward (e.g., 0.5) or 0 otherwise.
        """
        if self.curriculum_phase == 1 and self.actions[0] in [0, 1]:
            return 1.0
        if self.curriculum_phase == 2 and self.actions[0] in [2]:
            return 1.0
        return 0.0
