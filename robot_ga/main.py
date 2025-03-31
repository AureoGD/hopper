from jumper_env_ga import JumperEnv
from ga_evaluator import JumpEnvEvaluator
from ga import GeneticAlgorithm
import json

def main():
    # Step 1: Create your simulation environment
    env = JumperEnv(render=False, render_every=False)

    # Step 2: Wrap it in an evaluator that can track fitness and logging
    evaluator = JumpEnvEvaluator(env, max_steps=1000, log_data=True)

    # Step 3: Create and configure the genetic algorithm
    ga = GeneticAlgorithm(
        evaluator=evaluator,
        gene_size=7,              # [T1, T2, T3, Ts1, Ts2, Ts3, Ts4]
        population_size=30,
        generations=1000,
        mutation_rate=0.1,
        elite_fraction=0.4
    )

    # Step 4: Run evolution
    best_gene, best_fitness = ga.run()

    # Step 5: Print the best solution
    print("\n=== Genetic Algorithm Complete ===")
    print(f"Best Gene: {best_gene}")
    print(f"Best Fitness: {float(best_fitness):.4f}")

    # Step 6: Save log data (optional)
    with open("ga_results.json", "w") as f:
        json.dump(evaluator.logs, f, indent=2)

if __name__ == "__main__":
    main()
