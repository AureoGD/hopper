import numpy as np
import sympy as sp
from model_matrices import ModelMatrices

# Define symbolic variables
q0, q1, q2 = sp.symbols("q0 q1 q2")
dq0, dq1, dq2 = sp.symbols("dq0 dq1 dq2")

# Explicit Mass Matrix (M)
M_explicit = sp.Matrix(
    [
        [
            -0.065 * sp.sin(q2)
            - 0.0585 * sp.sin(q1 + q2)
            + 0.675 * sp.cos(q1)
            + 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)
            + 0.818583333333334,
            -0.065 * sp.sin(q2) - 0.02925 * sp.sin(q1 + q2) + 0.3375 * sp.cos(q1) + 0.316725,
            -0.0325 * sp.sin(q2) - 0.02925 * sp.sin(q1 + q2) + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.004225,
        ],
        [
            -0.065 * sp.sin(q2) - 0.02925 * sp.sin(q1 + q2) + 0.3375 * sp.cos(q1) + 0.316725,
            0.332525 - 0.065 * sp.sin(q2),
            0.004225 - 0.0325 * sp.sin(q2),
        ],
        [
            -0.0325 * sp.sin(q2) - 0.02925 * sp.sin(q1 + q2) + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.004225,
            0.004225 - 0.0325 * sp.sin(q2),
            2.16840434497101e-19 * sp.cos(-2 * q0 + 2 * q1 + 2 * q2)
            + 2.16840434497101e-19 * sp.cos(2 * q0 - 2 * q1 + 2 * q2)
            + 2.16840434497101e-19 * sp.cos(2 * q0 + 2 * q1 - 2 * q2)
            - 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)
            + 0.0504583333333333,
        ],
    ]
)
# Explicit Coriolis Matrix (C)
C_explicit = C = sp.Matrix(
    [
        [
            -0.006025 * dq0 * sp.sin(2 * q0 + 2 * q1 + 2 * q2)
            - dq1 * (0.3375 * sp.sin(q1) + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.02925 * sp.cos(q1 + q2))
            - dq2 * (0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.0325 * sp.cos(q2) + 0.02925 * sp.cos(q1 + q2)),
            -dq0 * (0.3375 * sp.sin(q1) + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.02925 * sp.cos(q1 + q2))
            - dq1 * (0.3375 * sp.sin(q1) + 0.02925 * sp.cos(q1 + q2))
            - dq2 * (0.0325 * sp.cos(q2) + 0.02925 * sp.cos(q1 + q2) - 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)),
            -dq0 * (0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.0325 * sp.cos(q2) + 0.02925 * sp.cos(q1 + q2))
            - dq1 * (0.0325 * sp.cos(q2) + 0.02925 * sp.cos(q1 + q2) - 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2))
            - dq2
            * (
                2.16840434497101e-19 * sp.sin(-2 * q0 + 2 * q1 + 2 * q2)
                - 2.16840434497101e-19 * sp.sin(2 * q0 - 2 * q1 + 2 * q2)
                - 2.16840434497101e-19 * sp.sin(2 * q0 + 2 * q1 - 2 * q2)
                + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2)
                + 0.0325 * sp.cos(q2)
                + 0.02925 * sp.cos(q1 + q2)
                - 0.01205 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)
            ),
        ],
        [
            dq0 * (0.3375 * sp.sin(q1) + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2) + 0.02925 * sp.cos(q1 + q2))
            - dq2 * (0.0325 * sp.cos(q2) + 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)),
            -0.0325 * dq2 * sp.cos(q2),
            -dq0 * (0.0325 * sp.cos(q2) + 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2))
            - 0.0325 * dq1 * sp.cos(q2)
            - dq2
            * (
                -2.16840434497101e-19 * sp.sin(-2 * q0 + 2 * q1 + 2 * q2)
                + 2.16840434497101e-19 * sp.sin(2 * q0 - 2 * q1 + 2 * q2)
                - 2.16840434497101e-19 * sp.sin(2 * q0 + 2 * q1 - 2 * q2)
                + 0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2)
                + 0.0325 * sp.cos(q2)
            ),
        ],
        [
            dq0
            * (
                0.006025 * sp.sin(2 * q0 + 2 * q1 + 2 * q2)
                + 0.0325 * sp.cos(q2)
                + 0.02925 * sp.cos(q1 + q2)
                + 0.01205 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)
            )
            + dq1 * (0.0325 * sp.cos(q2) + 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)),
            dq0 * (0.0325 * sp.cos(q2) + 0.006025 * sp.cos(2 * q0 + 2 * q1 + 2 * q2)) + 0.0325 * dq1 * sp.cos(q2),
            0,
        ],
    ]
)


def evaluate_symbolic_matrices(q_vals, dq_vals):
    """
    Evaluates the explicit symbolic Mass and Coriolis matrices using numerical values.
    """
    substitutions = {q0: q_vals[0], q1: q_vals[1], q2: q_vals[2], dq0: dq_vals[0], dq1: dq_vals[1], dq2: dq_vals[2]}

    # Compute numerical matrices
    M_sym_eval = np.array(M_explicit.subs(substitutions), dtype=np.float64)
    C_sym_eval = np.array(C_explicit.subs(substitutions), dtype=np.float64)

    return M_sym_eval, C_sym_eval


def evaluate_numerical_matrices(model, q_vals, dq_vals):
    """
    Uses ModelMatrices class to compute numerical Mass and Coriolis matrices.
    """
    model.update_robot_states(q_vals, dq_vals)
    model.update_kinematics()

    # Compute numerical Mass matrix
    M_num = np.zeros((3, 3))
    for i in range(3):
        J_i = getattr(model, f"J_com{i + 1}")
        M_num += model.masses[i + 1] * (J_i.T @ J_i)
        R_i = model.HT[i + 1][:3, :3]
        M_num += R_i @ model.Inertia[i + 1] @ R_i.T

    # Compute numerical Coriolis matrix
    C_num = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                C_num[i, j] += 0.5 * (M_num[i, j] + M_num[j, i] - M_num[k, i]) * dq_vals[k]

    return M_num, C_num


# Initialize the model
model = ModelMatrices()

# Test values for joint angles and velocities
q_test = np.array([0.1, 0, 0.7])  # Joint angles in radians
dq_test = np.array([0.1, -0.2, 0.05])  # Joint velocities in rad/s

# Compute matrices
M_sym, C_sym = evaluate_symbolic_matrices(q_test, dq_test)
M_num, C_num = evaluate_numerical_matrices(model, q_test, dq_test)

# Compare results
print("\n🔹 Mass Matrix Comparison")
print("Symbolic M:\n", M_sym)
print("Numerical M:\n", M_num)
print("Difference M:\n", np.abs(M_sym - M_num))

print("\n🔹 Coriolis Matrix Comparison")
print("Symbolic C:\n", C_sym)
print("Numerical C:\n", C_num)
print("Difference C:\n", np.abs(C_sym - C_num))

# Save results to file
with open("comparison_results.txt", "w") as f:
    f.write("🔹 Mass Matrix Comparison\n")
    f.write("Symbolic M:\n" + str(M_sym) + "\n")
    f.write("Numerical M:\n" + str(M_num) + "\n")
    f.write("Difference M:\n" + str(np.abs(M_sym - M_num)) + "\n\n")

    f.write("🔹 Coriolis Matrix Comparison\n")
    f.write("Symbolic C:\n" + str(C_sym) + "\n")
    f.write("Numerical C:\n" + str(C_num) + "\n")
    f.write("Difference C:\n" + str(np.abs(C_sym - C_num)) + "\n")

print("\n✅ Comparison complete! Results saved to `comparison_results.txt`")
