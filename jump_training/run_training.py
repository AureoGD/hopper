import sys
import os

# Optional: Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jump_training.generic_ppo import train

if __name__ == "__main__":
    from jump_training.generic_ppo import train

    # train(
    #     policy="DNN",
    #     stack_size=1,
    #     hidden_layers=[128, 128],
    #     total_interations=500,
    #     num_envs=16,
    #     eval_freq=5000,
    #     save_freq=1000,
    # )

    train(
        policy="CNN",
        stack_size=15,
        hidden_layers=[128, 128],
        total_interations=100,
        num_envs=16,
        eval_freq=5000,
        save_freq=1000,
    )
