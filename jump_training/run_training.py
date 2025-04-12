import sys
import os

# Optional: Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jump_training import generic_ppo

if __name__ == "__main__":
    sys.argv = [
        "generic_training.py",  # Fake script name (ignored)
        "--policy",
        "DNN",
        "--stack_size",
        "10",
        "--hidden_layers",
        "128",
        "128",
        "--total_timesteps",
        "200000",
        "--num_envs",
        "8",
        "--eval_freq",
        "10000",
        "--save_freq",
        "25000",
    ]

    # Run the training script
    exec(open(os.path.join("jump_training", "generic_training.py")).read())
