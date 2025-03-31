import json
import os
import numpy as np
import matplotlib.pyplot as plt

class GALogger:
    def __init__(self, log_dir="robot_ga/logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "ga_log.json")
        self.history = []
        self.gene_history = []
        self._init_plot()

    def _init_plot(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(8, 1, figsize=(12, 16), sharex=True)

        # Fitness subplot
        self.fitness_ax = self.axes[-1]
        self.fitness_ax.set_title("Fitness Evolution")
        self.fitness_ax.set_ylabel("Fitness")
        self.fitness_ax.set_xlabel("Generation")
        self.fitness_ax.grid(True)
        self.line_best, = self.fitness_ax.plot([], [], label="Best Fitness", color="tab:blue")
        self.line_mean, = self.fitness_ax.plot([], [], label="Mean Fitness", color="tab:orange")
        self.fitness_ax.legend()

        # Gene subplots
        self.gene_axes = self.axes[:-1]
        self.gene_lines = []
        for i, ax in enumerate(self.gene_axes):
            ax.set_title(f"Gene {i+1}")
            ax.set_ylabel(f"G{i+1}")
            ax.grid(True)
            line, = ax.plot([], [], label=f"G{i+1}")
            self.gene_lines.append(line)

        self.fig.tight_layout()

    def _update_plot(self):
        gens = [entry["best_gene"] for entry in self.history]
        best = [entry["best_fitness"] for entry in self.history]
        mean = [entry["mean_fitness"] for entry in self.history]

        # Update fitness lines
        self.line_best.set_data(gens, best)
        self.line_mean.set_data(gens, mean)
        self.fitness_ax.relim()
        self.fitness_ax.autoscale_view()

        if not self.gene_history:
            return

        genes_array = np.array(self.gene_history)  # shape: (gens, 7)
        for i, line in enumerate(self.gene_lines):
            if i < genes_array.shape[1]:
                line.set_data(gens, genes_array[:, i])
                self.gene_axes[i].relim()
                self.gene_axes[i].autoscale_view()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def log_generation(self, gen_idx, fitnesses, genes, best_idx, best_stats):
        best_fitness = float(fitnesses[best_idx])
        mean_fitness = float(fitnesses.mean())
        best_gene = genes[best_idx].tolist() if isinstance(genes[best_idx], np.ndarray) else list(genes[best_idx])

        entry = {
            "generation": int(gen_idx),
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "best_gene": best_gene,
            "n_jumps": int(best_stats.get("n_jumps", 0)),
            "max_stagnation": int(best_stats.get("max_stagnation", 0)),
            "reward": float(best_stats.get("reward", 0.0))            
        }

        self.history.append(entry)
        self.gene_history.append(best_gene)

        print(f"\n=== Generation {gen_idx} ===")
        print(f"Best Fitness: {best_fitness:.4f}")
        print(f"Mean Fitness: {mean_fitness:.4f}")
        print(f"Best Gene: {best_gene}")
        print(f"Jumps: {entry['n_jumps']} | Reward: {entry['reward']:.2f}")

        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # self._update_plot()

    def get_history(self):
        return self.history
