import numpy as np
import time

class JumpGAFcns:
    def __init__(self, _robot_states):
        self.robot_states = _robot_states

        # Reward tracking
        self.episode_reward = 0
        self.rewards = np.zeros((8, 1), dtype=np.double)

        # Jump tracking
        self.n_jumps = 0
        self.time_jump = 0
        self.delta_jump = 0
        self.first_landing = False
        self.last_b_x = 0
        self.delta_x = 0

        # Timers & flags
        self.inter = 0
        self.max_inter = 1500
        self.min_reward = -250

        # Thresholds
        self.jump_height_threshold = 0.1
        self.jump_time_threshold = 0.05

        # Reward weights
        self.jump_weight = 1
        self.jump_height_weight = 1
        self.prohibited_po_weight = 2.1
        self.body_weight = 3
        self.delta_x_weight = 2
        self.survive_weight = 1.1
        self.body_orietation_weight = 1
        self.stagnation_penalty_weight = 3

        self.max_stagnation_steps = 0

    def get_stats(self):
        return {
            "n_jumps": self.n_jumps,
            "reward": self.episode_reward,
            "max_stagnation": self.max_stagnation_steps
        }


    def end_of_step(self):
        reward = self._reward()
        terminated = self._done()
        self.inter += 1
        return reward, terminated

    def _reward(self):
        self.rewards[:] = 0

        self.rewards[0] = self._compute_air_time_reward()
        self.rewards[1] = self._stagnation_penalty()
        self.rewards[2] = self._check_body_ground_collision()
        self.rewards[3] = self._compute_delta_x()
        self.rewards[4] = self._check_foot_high()
        self.rewards[5] = self._survive_reward()
        self.rewards[6] = self._body_orietation_reward()

        reward = self.rewards.sum()

        # Heavy penalty if foot goes inside ground
        self.rewards[7] = self._foot_inside_ground(reward)
        reward = self.rewards[7]

        self.episode_reward += reward
        return reward

    def _done(self):
        if self.robot_states.toe_pos[1, 0] < -0.025:
            return True
        if self.robot_states.heel_pos[1, 0] < -0.025:
            return True
        if self.robot_states.b_pos[1, 0] < 0.35:
            return True
        if self.inter >= self.max_inter:
            return True
        return False

    def reset_vars(self):
        self.episode_reward = 0
        self.n_jumps = 0
        self.rewards[:, :] = 0
        self.inter = 0
        self.first_landing = False
        self.time_jump = 0
        self.delta_jump = 0
        self.delta_x = 0
        self.last_b_x = 0
        self.prev_states = None
        self.prev_mode = None
        self.stagnation_steps = 0
        self.max_stagnation_steps = 0
    
    ###########################
    def _compute_air_time_reward(self):
        self._track_air_time()
        foot_max_height = min(self.robot_states.toe_pos[1, 0], self.robot_states.heel_pos[1, 0])

        if self.delta_jump > self.jump_time_threshold and foot_max_height > self.jump_height_threshold:
            self.n_jumps += 1
            air_time_reward = self.delta_jump * self.jump_weight + self.n_jumps
            self.delta_x = abs(self.robot_states.b_pos[0, 0] - self.last_b_x)
        else:
            air_time_reward = 0

        self.delta_jump = 0
        return air_time_reward

    def _track_air_time(self):
        foot_contact_state = self.robot_states.toe_cont + self.robot_states.heel_cont * 2

        if (foot_contact_state[0, 0] == 3) and not self.first_landing:
            self.first_landing = True

        if self.first_landing and (foot_contact_state[0, 0] == 0) and self.time_jump == 0:
            self.time_jump = time.time()
            self.last_b_x = self.robot_states.b_pos[0, 0]

        if self.first_landing and (foot_contact_state[0, 0] != 0) and self.time_jump != 0:
            self.delta_jump = time.time() - self.time_jump
            self.time_jump = 0
            self.first_landing = False


    def _check_body_ground_collision(self):
        if self.robot_states.b_pos[1, 0] < 0.50:
            return -self.body_weight * (0.50 - self.robot_states.b_pos[1, 0])
        return 0

    def _compute_delta_x(self):
        dx = self.delta_x
        self.delta_x = 0
        return self.delta_x_weight * dx

    def _check_foot_high(self):
        toe_h = self.robot_states.toe_pos[1, 0]
        heel_h = self.robot_states.heel_pos[1, 0]

        if toe_h > self.jump_height_threshold and heel_h > self.jump_height_threshold:
            foot_avg = (toe_h + heel_h) / 2
            return self.jump_height_weight * np.clip(
                foot_avg - self.jump_height_threshold,
                a_min=self.jump_height_threshold,
                a_max=self.jump_height_threshold * 3
            )
        return 0

    def _survive_reward(self):
        return self.survive_weight

    def _body_orietation_reward(self):
        th = self.robot_states.th[0, 0]
        if th < -0.8 or th > 0.8:
            return -self.body_orietation_weight * 100 * (abs(th) - 0.8)
        return 0

    def _foot_inside_ground(self, reward):
        toe_h = self.robot_states.toe_pos[1, 0]
        heel_h = self.robot_states.heel_pos[1, 0]
        if toe_h < -0.05 or heel_h < -0.05:
            return -50 - abs(reward)
        return reward

    def _stagnation_penalty(self):
        current_states = np.hstack((
            self.robot_states.r_pos.flatten(),
            self.robot_states.r_vel.flatten(),
            self.robot_states.th.flatten(),
            self.robot_states.dth.flatten(),
            self.robot_states.q.flatten(),
            self.robot_states.dq.flatten(),
        ))
        current_mode = self.robot_states.ag_act

        if self.prev_states is None:
            self.prev_states = current_states
            self.prev_mode = current_mode
            self.stagnation_metric = 0
            self.stagnation_steps = 0
            self.max_stagnation_steps = 0  # For logging
            return 0

        delta_states = np.abs(current_states - self.prev_states)
        mode_unchanged = current_mode == self.prev_mode

        if np.max(delta_states) < 0.009 and mode_unchanged:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0

        self.max_stagnation_steps = max(self.max_stagnation_steps, self.stagnation_steps)

        self.prev_states = current_states
        self.prev_mode = current_mode

        return 0  # No reward penalty for now
