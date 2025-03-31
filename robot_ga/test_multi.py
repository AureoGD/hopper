import numpy as np
from multi_robot_env import MultiRobotEnv

def main():
    # 1. Create the multi-robot environment
    env = MultiRobotEnv(render=True)

    # 2. Create 6 individuals (random genes of length 7 for 4 postures)
    genes = [np.random.uniform(0.01, 0.15, size=(7,)) for _ in range(6)]

    # 3. Reset generation (spawn robots)
    env.reset_generation(genes)

    # 4. Run rollout
    fitnesses = env.run_generation_rollout()

    # 5. Show results
    print("=== Rollout Complete ===")
    for i, f in enumerate(fitnesses):
        print(f"Robot {i} → Fitness: {f.item():.4f}")

if __name__ == "__main__":
    main()
