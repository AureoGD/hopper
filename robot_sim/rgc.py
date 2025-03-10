import numpy as np
import osqp
import model_matrices
from math import sin, cos
from scipy import sparse

np.set_printoptions(edgeitems=30, linewidth=1000000)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


class rgc:
    def __init__(self):
        self.mdl = model_matrices.ModelMatrices()
        self.prob = osqp.OSQP()

        self.ts = 0.01
        self.N = 15
        self.M = 10

        self.q = np.zeros((3, 1))
        self.qr = np.zeros((3, 1))
        self.r = np.zeros((2, 1))
        self.dth = np.zeros((1, 1))
        self.th = np.zeros((1, 1))
        self.g = np.zeros((1, 1))
        self.th = np.zeros((2, 1))

        # x = [dr, dth, q, r, th, g] 11 sts
        self.nx = 10
        self.nu = 3
        self.ny = 3

        self.x = np.zeros((self.nx + self.nu, 1))

        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        self.C = np.zeros((self.ny, self.nx))

        self.Aa = np.zeros((self.nx + self.nu, self.nx + self.nu))
        self.Ba = np.zeros((self.nx + self.nu, self.nu))
        self.Ca = np.zeros((self.ny, self.nx + self.nu))

        self.Ca[:, 3:6] = np.identity(self.ny)

        self.ref = np.zeros((self.N * self.ny, 1))

        self.kp = 120
        self.kd = np.sqrt(self.kp)

        # self.H = sparse.csc_matrix(np.zeros((self.nu * self.M, self.nu * self.M)))
        # self.F = np.zeros((1, self.nu * self.M))

        self.Q = 0.0025 * np.identity(self.N * self.ny)
        self.R = 1.5 * np.identity(self.M * self.nu)
        self.init_flag = False

        self.dqr = np.zeros((3, 1))

    def solve_PO(self, q, qra, r, dr, th, dth, b, nqr):
        self.q = q
        self.qra = qra
        self.r = r
        self.dr = dr
        self.th = th
        self.dth = dth
        self.b = b

        for i in range(self.N):
            self.ref[i * self.ny : (i + 1) * self.ny, :] = nqr.reshape(3, 1)

        self.x[0:2, 0] = self.dr[:, 0]
        self.x[2, 0] = self.dth[:, 0]
        self.x[3:6, 0] = self.q[:, 0]
        self.x[6:8, 0] = self.r[:, 0]
        self.x[8, 0] = self.th[:, 0]
        self.x[9, 0] = -9.81
        self.x[10:, 0] = self.qra

        self.mdl_update()
        Phi, G = self.update_pred_mdl()

        H = 2 * sparse.csc_matrix((G.T @ self.Q @ G + self.R))
        F = 2 * (((Phi @ self.x) - self.ref).T) @ self.Q @ G

        if not self.init_flag:
            self.prob.setup(H, F.T, A=None, l=None, u=None, verbose=False)
            self.init_flag = True
        else:
            self.prob.update(q=F.T)
            self.prob.update(Px=sparse.triu(H).data)
        res = self.prob.solve()
        if res.info.status != "solved":
            raise ValueError("OSQP did not solve the problem!")
        else:
            self.dqr = res.x[0:3]

        return self.dqr

    def mdl_update(self):
        rot = np.array(
            [[cos(self.th[0, 0]), 0, sin(self.th[0, 0])], [0, 1, 0], [-sin(self.th[0, 0]), 0, cos(self.th[0, 0])]]
        )

        # kinematics
        self.mdl.update_robot_states(q=self.q, dq=[0])
        self.mdl.update_kinematics()
        Jc = rot @ self.mdl.J_ankle  # Jacobian of the ankle
        Jcom = rot @ self.mdl.com_jacobian()  # Jacobian of the leg CoM
        pc = rot @ self.mdl.HT_ankle[0:3, 3]  # Contact point
        r_ = rot @ self.mdl.update_com_pos()  # Center of mass position
        base = np.array([[self.b[0, 0]], [0], [self.b[1, 0]]])  # base position
        r_pc = (base + pc.reshape(3, 1)) - (base + r_)  #

        # Constraint Jacobian
        Jcs = np.array([[Jc[0, 0], Jc[0, 1], Jc[0, 2]], [Jc[2, 0], Jc[2, 1], Jc[2, 2]], [1, 1, 1]])

        # Contact Jaconian: [Jf 0; 0 Jb]
        Jc_invT = np.linalg.inv(Jcs.T)

        # Jacobian of leg CoM

        gamma = Jcom - Jc

        alpha = np.array(
            [
                [1, 0, r_pc[2, 0]],
                [0, 1, -r_pc[0, 0]],
                [0, 0, 1],
            ]
        )

        beta = np.array(
            [
                [gamma[0, 0], gamma[0, 1], gamma[0, 2]],
                [gamma[2, 0], gamma[2, 1], gamma[2, 2]],
                [1, 1, 1],
            ]
        )

        # skew symetric matrices of (pc-r), only return second line
        m1 = np.array([r_pc[2, 0], -r_pc[0, 0]]).reshape(1, 2)

        inertia = np.linalg.inv(self.mdl.update_inertia_tensor())

        sum_f = np.concatenate((np.identity(2), np.zeros((2, 1))), axis=1) / self.mdl.m
        sum_m = inertia[1, 1] * np.concatenate((m1, np.array([[-1]])), axis=1)
        T0 = np.linalg.inv(beta) @ alpha
        T1 = self.kd * Jc_invT @ T0

        self.A[0:2, 0:3] = -sum_f @ T1
        self.A[0:2, 3:6] = self.kp * sum_f @ Jc_invT
        self.A[1, 9] = 1

        self.A[2, 0:3] = -sum_m @ T1
        self.A[2, 3:6] = self.kp * sum_m @ Jc_invT

        self.A[3:6, 0:3] = T0

        self.A[6:8, 0:2] = np.identity(2)
        self.A[8, 2] = 1

        self.B[0:2, :] = -self.kp * sum_f @ Jc_invT
        self.B[2, :] = -self.kp * sum_m @ Jc_invT

        self.Aa[0:10, 0:10] = np.identity(self.nx) + self.ts * self.A
        self.Aa[0:10, 10:] = self.ts * self.B
        self.Aa[10:, 10:] = np.identity(self.nu)

        self.Ba[0:10, :] = self.ts * self.B
        self.Ba[10:, :] = np.identity(self.nu)

    def update_pred_mdl(self):
        G = np.zeros((self.ny * self.N, self.nu * self.M))
        Phi = np.zeros((self.ny * self.N, self.nx + self.nu))

        aux = np.zeros((self.ny, self.nu))
        aux[:, :] = self.Ca @ self.Ba

        Phi[0 : self.ny, :] = self.Ca @ self.Aa

        for i in range(self.N):
            j = 0
            if i != 0:
                Phi[i * self.ny : (i + 1) * self.ny, :] = Phi[(i - 1) * self.ny : i * self.ny, :] @ self.Aa
                aux[:, :] = Phi[(i - 1) * self.ny : i * self.ny, :] @ self.Ba

            while (j < self.M) and (i + j < self.N):
                G[(i + j) * self.ny : (i + j + 1) * self.ny, j * (self.nu) : (j + 1) * (self.nu)] = aux[:, :]
                j += 1

        return Phi, G
