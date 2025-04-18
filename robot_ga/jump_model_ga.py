import numpy as np
import os
from math import cos, sin, sqrt
import random
from collections import deque
import sys
from icecream import ic
from dataclasses import dataclass, field
import time

current_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append("./model")
import model_matrices


@dataclass
class RobotStates:
    b_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    b_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    r_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    r_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    th: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    dth: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))

    q: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    dq: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    qr: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    dqr: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    qrh: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))
    tau: np.ndarray = field(default_factory=lambda: np.zeros((3, 1), dtype=np.float64))

    toe_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    toe_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    toe_cont: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))

    heel_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    heel_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    heel_cont: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))

    ankle_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    ankle_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))

    knee_pos: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))
    knee_vel: np.ndarray = field(default_factory=lambda: np.zeros((2, 1), dtype=np.float64))

    j_val: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))
    ag_act: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))

    rcg_status: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float64))


class JumpModel:
    def __init__(self, _robot_states, N=10):
        # Paths and Constants
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/hopper.urdf")
        # print(self.model_path)
        self.AC_JOINT_LIST = (4, 6, 8)
        self.JOINT_ST_LIST = (0, 1, 2, 4, 6, 8)
        self.CONTACT_JOINTS = (10, 11)
        self.JOINT_MODEL_NUM = len(self.JOINT_ST_LIST)

        self.mdl = model_matrices.ModelMatrices()

        # Time steps
        self.sim_dt = 0.001  # Simulation time step
        self.qr_dt = 0.01  # Robot controller time step

        # Joint control variables
        self.kp = 150
        self.kd = 2 * sqrt(self.kp)
        self.Kp = self.kp * np.identity(len(self.AC_JOINT_LIST))
        self.Kd = self.kd * np.identity(len(self.AC_JOINT_LIST))
        self.MAX_TAU = 50
        self.V_MAX_TAU = self.MAX_TAU * np.ones((len(self.AC_JOINT_LIST), 1))


        # Robot states variables
        self.robot_states = _robot_states

        self.rw_ant = np.zeros((2, 1), dtype=np.float64)

        self.joint_p_max = [0, 1.15, 0, 0.75, 0.45, 1]
        self.joint_p_min = [0, 0.95, 0, -0.75, -2.0, -1]

        # Flag for first integration step
        self.flag_first_int = False
        self.ac_step = 0

        # GA variables
        self.alpha = []
        self.k = []
        self.T_switch = []
        self.mode = 0
        self.qr_ant = np.zeros((3, 1))
        self.qr_targets = [
            np.array([[-70], [110], [-40]])*np.pi/180,  # qr1
            np.array([[-32], [60], [-10]])*np.pi/180,  # qr2
            np.array([[-70], [120], [-50]])*np.pi/180,# qr3
            np.array([[-32.5], [60], [-25]])*np.pi/180   # qr4
        ]

    def set_genes(self, genes):
        self.T_switch = genes[0:3]

        Ts_list = genes[3:7]
        self.alpha = [np.exp(-0.01 * 0.35 * 2 * np.pi / Ts) for Ts in Ts_list]
        self.k = [1.0 - a for a in self.alpha]

    def update_qr(self, t):
        T1, T2, T3 = self.T_switch
        if t < T1:
            mode = 0
        elif t < T1 + T2:
            mode = 1
        elif t < T1 + T2 + T3:
            mode = 2
        else:
            mode = 3

        # Apply corresponding filter
        qr_target = self.qr_targets[mode]
        self.robot_states.qr = self.alpha[mode] * self.qr_ant + self.k[mode] * qr_target
        self.qr_ant = np.copy(self.robot_states.qr)

    def command_torque(self):
        tau_pd = self._torque_pd()
        tau_cg = self._torque_compG()
        self.robot_states.tau = np.clip(tau_pd - tau_cg, -self.V_MAX_TAU, self.V_MAX_TAU)
        return self.robot_states.tau

    def _torque_pd(self):
        qr = self.robot_states.qr
        q = self.robot_states.q
        dqr = self.robot_states.dqr
        dq = self.robot_states.dq

        return self.Kp @ (qr - q) + self.Kd @ (dqr - dq)

    def _torque_compG(self):
        rot = self._rotY()

        J_com1 = rot @ self.mdl.J_com1
        J_com2 = rot @ self.mdl.J_com2
        J_com3 = rot @ self.mdl.J_com3

        tau_g = (
            J_com1.transpose() * self.mdl.m1 + J_com2.transpose() * self.mdl.m2 + J_com3.transpose() * self.mdl.m3
        ) @ np.array([[0], [0], [-9.81]])

        return tau_g

    def update_robot_states(self, q_aux, dq_aux, aux_F_cont):
        self.robot_states.b_pos[0, 0] = q_aux[0, 0]
        self.robot_states.b_pos[1, 0] = q_aux[1, 0]

        self.robot_states.b_vel[0, 0] = dq_aux[0, 0]
        self.robot_states.b_vel[1, 0] = dq_aux[1, 0]

        self.robot_states.th[0, 0] = q_aux[2, 0]
        self.robot_states.dth[0, 0] = dq_aux[2, 0]

        # joints pos and vel
        self.robot_states.q[0, 0] = q_aux[3, 0]
        self.robot_states.q[1, 0] = q_aux[4, 0]
        self.robot_states.q[2, 0] = q_aux[5, 0]

        self.robot_states.dq[0, 0] = dq_aux[3, 0]
        self.robot_states.dq[1, 0] = dq_aux[4, 0]
        self.robot_states.dq[2, 0] = dq_aux[5, 0]

        self.robot_states.heel_cont[:] = aux_F_cont[1, 0]
        self.robot_states.toe_cont[:] = aux_F_cont[0, 0]

        self.mdl.update_robot_states(q=self.robot_states.q, dq=self.robot_states.dq)
        self.mdl.update_kinematics()

        r_vet = self.mdl.update_com_pos()

        self.robot_states.r_pos, self.robot_states.r_vel = self._write_wrt(r_vet, self.robot_states.r_pos)
        if not self.flag_first_int:
            self.flag_first_int = True
            self.robot_states.r_vel[:] = 0

    def _rotY(self):
        th = self.robot_states.th
        return np.array([[cos(th), 0, sin(th)], [0, 1, 0], [-sin(th), 0, cos(th)]])

    def randon_joint_pos(self):
        # Create a array for joint positions
        q = np.zeros((self.JOINT_MODEL_NUM, 1), dtype=np.float64)

        q[1, 0] = 0.9
        q[3, 0] =  np.pi * -32.5 / 180
        q[4, 0] =  np.pi * 60 / 180 
        q[5, 0] = np.pi * -32.5 / 180

        # ensure drw as zero in the first interation
        self.rw_ant[:] = 0
        self.flag_first_int = False

        return q

    def ml_states(self):
        self.robot_states.ankle_pos, self.robot_states.ankle_vel = self._write_wrt(
            self.mdl.HT_ankle[0:3, 3], self.robot_states.ankle_pos
        )

        self.robot_states.toe_pos, self.robot_states.toe_vel = self._write_wrt(
            self.mdl.HT_toe[0:3, 3], self.robot_states.toe_pos
        )

        self.robot_states.heel_pos, self.robot_states.heel_vel = self._write_wrt(
            self.mdl.HT_heel[0:3, 3], self.robot_states.heel_pos
        )

        self.robot_states.knee_pos, self.robot_states.knee_vel = self._write_wrt(
            self.mdl.HT_knee[0:3, 3], self.robot_states.knee_pos
        )

        if self.ac_step == 0:
            self.robot_states.ankle_vel[:] = 0
            self.robot_states.toe_vel[:] = 0
            self.robot_states.heel_vel[:] = 0
            self.robot_states.knee_vel[:] = 0

        self.ac_step += 1

    def _write_wrt(self, _pos, _prev_pos):
        pos_vet = self._rotY() @ _pos.reshape(3, 1)

        prev_pos = np.copy(_prev_pos)

        vet_ = np.zeros((2, 1), dtype=np.float64)
        dvet_ = np.zeros((2, 1), dtype=np.float64)

        vet_[0, 0] = self.robot_states.b_pos[0, 0] + pos_vet[0, 0]
        vet_[1, 0] = self.robot_states.b_pos[1, 0] + pos_vet[2, 0]

        dvet_ = (vet_ - prev_pos) / self.sim_dt

        return vet_, dvet_

    def reset_variables(self):
        self.ac_step = 0
        self.flag_first_int = False

    def init_qr(self, _q):
        self.robot_states.qr[0, 0] = _q[0, 0] - random.uniform(-0.1, 0.1)
        self.robot_states.qr[1, 0] = _q[1, 0] - random.uniform(-0.1, 0.1)
        self.robot_states.qr[2, 0] = _q[2, 0] - random.uniform(-0.1, 0.1)
