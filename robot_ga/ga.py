import numpy as np
import matplotlib.pyplot as plt
from ga_logger import GALogger  # You will create this file as shown before
from multi_robot_env import MultiRobotEnv
import jump_ga_fcns

class GeneticAlgorithm:
    def __init__(self, population_size, generations, mutation_rate):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.gene_size = 7  # 3 T_switch + 4 Ts values
        self.logger = GALogger()

        self.T_SWITCH_MIN = 0.5
        self.T_SWITCH_MAX = 2

        self.TS_MIN = 0.05
        self.TS_MAX = 1.0

        # Init population: [T1, T2, T3, Ts1, Ts2, Ts3, Ts4]
        self.population = [self._generate_individual() for _ in range(self.population_size)]

    def _generate_individual(self):
        # T_switch = sorted(np.random.uniform(0.1, 0.6, size=3))
        # Ts = np.random.uniform(0.02, 0.2, size=4)
        # return np.concatenate([T_switch, Ts])
        # return np.array([2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5])

        T_switch = sorted(np.random.uniform(self.T_SWITCH_MIN, self.T_SWITCH_MAX, size=3))
        Ts = np.random.uniform(self.TS_MIN, self.TS_MAX, size=4)
        return np.concatenate([T_switch, Ts])

    def _mutate(self, gene):
        new_gene = gene.copy()
        for i in range(len(new_gene)):
            if np.random.rand() < self.mutation_rate:
                if i < 3:
                    new_gene[i] += np.random.normal(0.1, 3)  # T_switch
                else:
                    new_gene[i] += np.random.normal(0.1, 2)  # Ts
        new_gene[:3] = sorted(new_gene[:3])
        return np.clip(new_gene, 0.01, 1.0)

    def _crossover(self, parent1, parent2):
        point = np.random.randint(1, self.gene_size - 1)
        child = np.concatenate([parent1[:point], parent2[point:]])
        child[:3] = sorted(child[:3])
        return child

    def _select_parents(self, fitnesses):
        fitnesses = fitnesses.flatten()  # Ensure it's 1D
        top2 = np.argsort(fitnesses)[-2:]  # Indices of top 2
        i1, i2 = int(top2[0]), int(top2[1])
        return self.population[i1], self.population[i2]
    
    def run(self):
        plt.ion()
        env = MultiRobotEnv(render=True)

        for gen in range(self.generations):
            env.reset_generation(self.population)

            terminated = [False] * self.population_size
            while not all(terminated):
                env.step_all()
                for i, robot in enumerate(env.robots):
                    if not terminated[i]:
                        reward, done = robot["ga"].end_of_step()
                        robot["reward"] += reward
                        terminated[i] = done

            # Evaluate fitness
            fitnesses = np.array([r["reward"] for r in env.robots])
            best_idx = int(np.argmax(fitnesses))
            best_robot = env.robots[best_idx]

            self.logger.log_generation(
                gen_idx=gen,
                fitnesses=fitnesses,
                genes=self.population,
                best_idx=best_idx,
                best_stats={
                    "n_jumps": best_robot["ga"].n_jumps,
                    "reward": best_robot["ga"].episode_reward
                }
            )

            # Plot
            self._plot_fitness(self.logger.get_history())

            # Reproduce
            parent1, parent2 = self._select_parents(fitnesses)
            new_population = [self.population[best_idx]]  # Elitism
            while len(new_population) < self.population_size:
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            self.population = new_population

        print("\n=== Genetic Algorithm Complete ===")

    def _plot_fitness(self, history):
        gens = [entry["generation"] for entry in history]
        best = [entry["best_fitness"] for entry in history]
        mean = [entry["mean_fitness"] for entry in history]

        plt.clf()
        plt.plot(gens, best, label="Best Fitness")
        plt.plot(gens, mean, label="Mean Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("GA Fitness Progress")
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=10, generations=1000, mutation_rate=0.1)
    ga.run()
