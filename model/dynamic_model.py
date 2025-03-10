import sympy as sp
import numpy as np


class ModelMatrices:
    def __init__(self):
        # Link lengths
        self.L0 = 0.4
        self.L1 = 0.45
        self.L2 = 0.5
        self.L3 = 0.39
        self.r = 0.05

        # **Numerical Masses** (Constants)
        self.m0 = 7.0
        self.m1 = 1.0
        self.m2 = 1.0
        self.m3 = 1.0
        self.masses = [self.m0, self.m1, self.m2, self.m3]  # List of masses

        # **Inertia Tensors (Numerical Values)**
        self.I0 = self._initi_inertia_tensor(self.m0, self.r, self.L0)
        self.I1 = self._initi_inertia_tensor(self.m1, self.r, self.L1)
        self.I2 = self._initi_inertia_tensor(self.m2, self.r, self.L2)
        self.I3 = self._initi_inertia_tensor(self.m3, self.r, self.L3)
        arr = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        self.I3 = arr @ self.I3 @ arr.transpose()

        self.Inertia = [self.I0, self.I1, self.I2, self.I3]

        # **Symbolic Joint Angles & Velocities**
        self.q = sp.Matrix(sp.symbols("q0 q1 q2"))  # Joint positions
        self.dq = sp.Matrix(sp.symbols("dq0 dq1 dq2"))  # Joint velocities

        # Compute sin/cos values
        self.sin_vals, self.cos_vals = self.compute_sin_cos(
            [self.q[0], self.q[0] + self.q[1], self.q[0] + self.q[1] + self.q[2]]
        )

        # **Homogeneous Transformations & Jacobians**
        self.HT_com = [sp.eye(4) for _ in range(4)]
        self.J_com = [sp.zeros(3, 3) for _ in range(3)]

        # **Update Transformations & Jacobians**
        self.update_homog_trans()
        self.update_jacobians()

    def _initi_inertia_tensor(self, m, r, h):
        Inertia_tensor = np.zeros((3, 3), dtype=np.float64)
        Inertia_tensor[0, 0] = ((m * h**2) / 12) + ((m * r**2) / 4)
        Inertia_tensor[1, 1] = Inertia_tensor[0, 0]
        Inertia_tensor[2, 2] = (m * r**2) / 2

        return Inertia_tensor

    def compute_sin_cos(self, angles):
        """Compute symbolic sin and cos values."""
        sin_vals = [sp.sin(angle) for angle in angles]
        cos_vals = [sp.cos(angle) for angle in angles]
        return sin_vals, cos_vals

    def update_homog_trans(self):
        """Compute homogeneous transforms symbolically (Corrected version)."""
        sin_vals, cos_vals = self.sin_vals, self.cos_vals

        # HT_com1 (First link transformation)
        self.HT_com[1][:, :] = sp.Matrix(
            [
                [sin_vals[0], cos_vals[0], 0, 0.225 * sin_vals[0]],
                [0, 0, -1, 0],
                [-cos_vals[0], sin_vals[0], 0, -0.225 * cos_vals[0]],
                [0, 0, 0, 1],
            ]
        )

        # HT_com2 (Second link transformation)
        self.HT_com[2][:, :] = sp.Matrix(
            [
                [sin_vals[1], cos_vals[1], 0, 0.45 * sin_vals[0] + 0.25 * sin_vals[1]],
                [0, 0, -1, 0],
                [-cos_vals[1], sin_vals[1], 0, -0.45 * cos_vals[0] - 0.25 * cos_vals[1]],
                [0, 0, 0, 1],
            ]
        )

        # HT_com3 (Third link transformation)
        self.HT_com[3][:, :] = sp.Matrix(
            [
                [sin_vals[2], cos_vals[2], 0, 0.45 * sin_vals[0] + 0.065 * cos_vals[2] + 0.5 * sin_vals[1]],
                [0, 0, -1, 0],
                [-cos_vals[2], sin_vals[2], 0, 0.065 * sin_vals[2] - 0.45 * cos_vals[0] - 0.5 * cos_vals[1]],
                [0, 0, 0, 1],
            ]
        )

    def update_jacobians(self):
        """Compute symbolic Jacobians."""
        sin_vals, cos_vals = self.sin_vals, self.cos_vals

        self.J_com[0][:, 0] = sp.Matrix([0.225 * cos_vals[0], 0, 0.225 * sin_vals[0]])

        self.J_com[1][:, :2] = sp.Matrix(
            [
                [0.45 * cos_vals[0] + 0.25 * cos_vals[1], 0.25 * cos_vals[1]],
                [0, 0],
                [0.45 * sin_vals[0] + 0.25 * sin_vals[1], 0.25 * sin_vals[1]],
            ]
        )

        self.J_com[2][:, :] = sp.Matrix(
            [
                [
                    0.45 * cos_vals[0] - 0.065 * sin_vals[2] + 0.5 * cos_vals[1],
                    0.5 * cos_vals[1] - 0.065 * sin_vals[2],
                    -0.065 * sin_vals[2],
                ],
                [0, 0, 0],
                [
                    0.45 * sin_vals[0] + 0.065 * cos_vals[2] + 0.5 * sin_vals[1],
                    0.5 * sin_vals[1] + 0.065 * cos_vals[2],
                    0.065 * cos_vals[2],
                ],
            ]
        )

    def compute_mass_matrix(self):
        """Compute the Mass Matrix (M) using numerical masses and inertia tensors."""
        M = sp.Matrix(3, 3, lambda i, j: 0)

        for i in range(3):
            J_i = self.J_com[i]
            M += self.masses[i + 1] * (J_i.T @ J_i)
            R_i = self.HT_com[i + 1][:3, :3]  # Extract rotation matrix
            M += R_i @ sp.Matrix(self.Inertia[i + 1]) @ R_i.T

        return sp.simplify(M)

    def compute_coriolis_matrix(self):
        """Compute the Coriolis Matrix (C) using Christoffel symbols."""
        M = self.compute_mass_matrix()
        C = sp.Matrix(3, 3, lambda i, j: 0)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    C[i, j] += (
                        (1 / 2)
                        * (M[i, j].diff(self.q[k]) + M[i, k].diff(self.q[j]) - M[j, k].diff(self.q[i]))
                        * self.dq[k]
                    )

        return sp.simplify(C)


# **Create Model Instance**
model = ModelMatrices()

# **Compute Mass & Coriolis Matrices**
M_sym = model.compute_mass_matrix()
C_sym = model.compute_coriolis_matrix()

# **Save Results**
with open("dynamics_output.txt", "w") as f:
    f.write("Mass Matrix (M):\n" + str(M_sym) + "\n\n")
    f.write("Coriolis Matrix (C):\n" + str(C_sym) + "\n\n")

print("Mass & Coriolis Matrices saved to `dynamics_output.txt`")
