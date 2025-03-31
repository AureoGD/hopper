from jumper_env_ga import JumperEnv

class JumpEnvEvaluator:
    def __init__(self, env, max_steps=1000, log_data=True):
        self.env = env
        self.max_steps = max_steps
        self.log_data = log_data
        self.logs = []


    def evaluate(self, genes, render=False):
        # Set render mode for the env if needed
        self.env._is_render = render

        # Reset the environment and apply gene controller
        self.env.reset()
        self.env.set_genes(genes)

        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < self.max_steps:
            reward, done, _, _ = self.env.step()
            total_reward += reward
            steps += 1

        # Optional logging
        if self.log_data:
            log = {
                "genes": genes.tolist(),
                "reward": total_reward,
                "jumps": self.env.robot_ga_fcns.n_jumps,
                "delta_x": self.env.robot_ga_fcns.delta_x,
                "steps": steps,
            }
            self.logs.append(log)

        return total_reward

# env = JumperEnv(render=True, max_interations=2500)
# evaluator = JumpEnvEvaluator(env)
# genes = [3, 6, 9, 0.25, 0.25, 0.25, 0.25]
# evaluator.evaluate(genes=genes)