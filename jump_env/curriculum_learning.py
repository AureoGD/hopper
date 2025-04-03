from jump_env.jump_ml_fcns import JumpMLFcns


class CurriculumJumpFcns:
    def __init__(self, robot_states):
        self.base = JumpMLFcns(robot_states)
        self.curriculum_stage = 0
        self.stage_thresholds = [100, 5]  # [min reward, min jumps]

    def reset_vars(self):
        self.base.reset_vars()

    def observation(self):
        return self.base.observation()

    def end_of_step(self):
        obs, reward, done = self.base.end_of_step()
        reward += self._curriculum_bonus()
        return obs, reward, done

    def _curriculum_bonus(self):
        # Optionally give bonus for current stage or modify reward logic
        if self.curriculum_stage == 0:
            return self._balance_reward()
        elif self.curriculum_stage == 1:
            return self._hop_reward()
        else:
            return self._jump_reward()

    def _balance_reward(self):
        # Reward standing still or upright orientation
        return 5.0 if self.base.robot_states.r_pos[1, 0] > 1.0 else 0.0

    def _hop_reward(self):
        return self.base._compute_air_time_reward() * 0.5

    def _jump_reward(self):
        return self.base._compute_air_time_reward()

    def check_curriculum_progression(self):
        if self.curriculum_stage == 0 and self.base.episode_reward > self.stage_thresholds[0]:
            self.curriculum_stage = 1
            print("Advanced to Stage 1: Hop Training")
        elif self.curriculum_stage == 1 and self.base.n_jumps > self.stage_thresholds[1]:
            self.curriculum_stage = 2
            print("Advanced to Stage 2: Jump Training")
