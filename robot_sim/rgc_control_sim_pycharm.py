import pybullet as p
import pybullet_data
import numpy as np
import os
from icecream import ic
import time
from math import sin, cos
import matplotlib.pyplot as plt
import sys

import model_matrices
import rgc

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "/../rgc_controller/build")


import pybind_opWrapper


rgc_controller_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../rgc_controller/config/config.yaml"))


mdl = model_matrices.ModelMatrices()
gvctrl = rgc.rgc()


SIM_TIME = 0.001

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/hopper.urdf")

JOINT_ST_LIST = (0, 1, 2, 4, 6, 8)
AC_JOINT_LIST = (4, 6, 8)

N_int = 15000
ch = 5

p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(SIM_TIME)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
p.changeDynamics(plane, -1, lateralFriction=1.0)

model = p.loadURDF(MODEL_PATH, [0, 0, 0], p.getQuaternionFromEuler([0, 0 * np.pi / 180, 0]))

number_joints = p.getNumJoints(model)

for idx in range(number_joints - 1):
    p.setJointMotorControl2(model, idx, p.VELOCITY_CONTROL, force=0)
    # q[1, 0] = 0.75
    # q[3, 0] = 0.92
    # q[4, 0] = -1.6
    # q[5, 0] = 0.6
# q0 = np.array([0, 0.75, 0, np.pi * 32.5 / 180, np.pi * -60 / 180, np.pi * 32.5 / 180])

# q0 = np.array([0, 0.95, 0, -0.56, 1.06, -0.56])
q0 = np.array([0, 0.95, 0, 0.56, -1.06, 0.56])


for idx in range(len(JOINT_ST_LIST)):
    p.resetJointState(model, JOINT_ST_LIST[idx], q0[idx])


for idx in range(len(AC_JOINT_LIST)):
    p.enableJointForceTorqueSensor(model, AC_JOINT_LIST[idx], enableSensor=True)


q_aux = np.zeros((len(JOINT_ST_LIST), 1), dtype=np.float64)
dq_aux = np.zeros((len(JOINT_ST_LIST), 1), dtype=np.float64)

q = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)
dq = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)
qr = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)
qrh = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)

b = np.zeros((2, 1), dtype=np.float64)
db = np.zeros((2, 1), dtype=np.float64)
rb = np.zeros((2, 1), dtype=np.float64)

dr = np.zeros((2, 1), dtype=np.float64)
rw = np.zeros((2, 1), dtype=np.float64)
rw_ant = np.zeros((2, 1), dtype=np.float64)

th = np.zeros((1, 1), dtype=np.float64)
dth = np.zeros((1, 1), dtype=np.float64)

t1 = np.zeros((4, 1), dtype=np.float64)
t2 = np.zeros((4, 1), dtype=np.float64)


# [t, rw, drw, b, db, th, dth, q, dq]
max_int = 20000
x_his = np.zeros((11, max_int))
q_his = np.zeros((10, max_int))

kp = 150
kd = 10
# kd = 10

KP_mtx = kp * np.identity(len(AC_JOINT_LIST))
KD_mtx = kd * np.identity(len(AC_JOINT_LIST))
V_MAX_TAU = 70 * np.ones((len(AC_JOINT_LIST), 1), dtype=np.float64)

qrh[:, 0] = q0[3:]
qr[:, 0] = q0[3:]

count = 0
flag = True
PO = 0

RGC = pybind_opWrapper.Op_Wrapper()
RGC.load_config(rgc_controller_path)
RGC.RGCConfig(0.01, kp, kd)


#  SIM
def get_joit_states():
    for idx in range(len(JOINT_ST_LIST)):
        q_aux[idx], dq_aux[idx], forces, tau_ = p.getJointState(model, JOINT_ST_LIST[idx])

    # Floating base joints
    b[0, 0] = q_aux[0, 0]
    b[1, 0] = q_aux[1, 0]

    db[0, 0] = dq_aux[0, 0]
    db[1, 0] = dq_aux[1, 0]

    th[0, 0] = q_aux[2, 0]
    dth[0, 0] = dq_aux[2, 0]

    # joints pos and vel
    q[0, 0] = q_aux[3, 0]
    q[1, 0] = q_aux[4, 0]
    q[2, 0] = q_aux[5, 0]

    dq[0, 0] = dq_aux[3, 0]
    dq[1, 0] = dq_aux[4, 0]
    dq[2, 0] = dq_aux[5, 0]

    mdl.update_robot_states(q=q, dq=[0])
    mdl.update_kinematics()

    r_vet = mdl.update_com_pos()
    rb[0, 0] = r_vet[0, 0]
    rb[1, 0] = r_vet[2, 0]

    r_vet = rotY() @ r_vet

    rw[0, 0] = b[0, 0] + r_vet[0, 0]
    rw[1, 0] = b[1, 0] + r_vet[2, 0]

    if count != 0:
        dr_vet = (rw - rw_ant) / SIM_TIME
        dr[0, 0] = dr_vet[0, 0]
        dr[1, 0] = dr_vet[1, 0]
    rw_ant[:, 0] = rw[:, 0]


def low_lvl_control():
    global qrh
    if count > 9999 and count % 10 == 0:
        """ C++ method
            Op_Wrapper::UpdateSt(Eigen::Matrix<double, 3, 1> *_q,
                                    Eigen::Matrix<double, 3, 1> *_qd,
                                    Eigen::Matrix<double, 3, 1> *_qr,
                                    Eigen::Matrix<double, 2, 1> *_dr,
                                    Eigen::Matrix<double, 2, 1> *_r,
                                    Eigen::Matrix<double, 2, 1> *_db,
                                    Eigen::Matrix<double, 2, 1> *_b,
                                    double _dth,
                                    double _th)
        """

        RGC.UpdateSt(q, dq, qr[:, 0], dr, rw, db, b, dth[0, 0], th[0, 0])
        rgc_solved = RGC.ChooseRGCPO(PO)
        # print(rgc_solved)
        qrh = RGC.delta_qhl
        qrh = qrh.reshape(3, 1)
        if rgc_solved == 1:
            dqr = RGC.delta_qr.reshape(3, 1)
            if np.any(np.isnan(dqr)):
                rgc_solved = 0
            else:
                # print(dqr)
                qr[:, 0] = qr[:, 0] + dqr[:, 0]

    tau1 = np.matmul(KP_mtx, (qr - q)) - np.matmul(KD_mtx, dq)
    tau2 = comp_tau()
    tau_star = np.clip(tau1 - tau2, -V_MAX_TAU, V_MAX_TAU)
    p.setJointMotorControlArray(model, AC_JOINT_LIST, p.TORQUE_CONTROL, forces=tau_star[:, 0])
    return tau1, tau2


def rotY():
    return np.array([[cos(th[0, 0]), 0, sin(th[0, 0])], [0, 1, 0], [-sin(th[0, 0]), 0, cos(th[0, 0])]])


def comp_tau():
    rot = rotY()

    # J_com1, J_com2, J_com3 = mdl.update_CoM_jacobians(qr)

    # J_com1 = rot @ J_com1
    # J_com2 = rot @ J_com2
    # J_com3 = rot @ J_com3

    J_com1 = rot @ mdl.J_com1
    J_com2 = rot @ mdl.J_com2
    J_com3 = rot @ mdl.J_com3

    tau_g = (J_com1.transpose() * mdl.m1 + J_com2.transpose() * mdl.m2 + J_com3.transpose() * mdl.m3) @ np.array(
        [[0], [0], [-9.81]]
    )

    return tau_g


def write_history():
    # [t, rw, drw, b, db, th, dth, q, dq]

    x_his[0:2, count] = dr[:, 0]
    x_his[2:4, count] = rw[:, 0]
    x_his[4, count] = dth[0, 0]
    x_his[5, count] = th[0, 0]

    x_his[6:9, count] = (t1).reshape(
        3,
    )

    x_his[10, count] = th[0, 0] - q[0, 0] - q[1, 0] - q[2, 0]

    q_his[0, count] = count * SIM_TIME

    q_his[1, count] = q[0, 0]
    q_his[2, count] = q[1, 0]
    q_his[3, count] = q[2, 0]

    q_his[4, count] = qr[0, 0]
    q_his[5, count] = qr[1, 0]
    q_his[6, count] = qr[2, 0]

    q_his[7, count] = qrh[0, 0]
    q_his[8, count] = qrh[1, 0]
    q_his[9, count] = qrh[2, 0]


# while count <= N_int * ch - 1:

while count <= max_int - 1:
    get_joit_states()
    t1, t2 = low_lvl_control()
    write_history()
    p.stepSimulation()
    count += 1
    if count > 9999 and count < 17000:
        PO = 5
    if count >= 17000:
        PO = 7
    # elif count > 15000 and count < 15250:
    #     PO = 4
    # elif count > 15250 and count < 15500:
    #     PO = 6
    # elif count > 15500 and count < 16100:
    #     PO = 5
    # elif count > 16100:
    #     PO = 2
    # if count % N_int == 0:
    # if flag:
    #     PO = 2
    #     flag = False
    # else:
    #     PO = 4
    #     flag = True
    # time.sleep(SIM_TIME)

p.disconnect()

t = q_his[0, 1000:]
data_series = q_his[1:, 1000:]

subplot_groups = [
    [1, 4],  # Subplot 7: series 11, 12, and 13
    [2, 5],  # Subplot 7: series 11, 12, and 13
    [3, 6],  # Subplot 7: series 11, 12, and 13
]

series_labels = [
    r"$q$",
    r"$q_{rgc}$",
    r"$q$",
    r"$q_{rgc}$",
    r"$q$",
    r"$q_{rgc}$",
]

subplot_titles = [r"$q_{1}$", r"$q_{2}$", r"$q_{3}$"]
subplot_ylabel = ["rad", "rad", "rad"]


fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
axes = axes.flatten()

label_idx = 0
# Plot each group in its respective subplot
for idx, (group, title, label) in enumerate(zip(subplot_groups, subplot_titles, subplot_ylabel)):
    ax = axes[idx]  # Current subplot
    for series in group:
        ax.plot(t, data_series[series - 1], linestyle="-", label=f"{series_labels[label_idx]}")
        label_idx += 1  # Move to the next label
    ax.set_ylabel(label)
    ax.set_title(title)  # Set title from the list
    ax.grid(True)
    ax.legend()

# Remove unused subplots
for ax in axes[len(subplot_groups) :]:
    fig.delaxes(ax)

# Add a shared X-axis label
plt.xlabel("Time")
ax.set_xlim(t[0], t[-1])
# Adjust layout
plt.tight_layout()


print(data_series[0, -1])
print(data_series[1, -1])
print(data_series[2, -1])

data_series = x_his[:, 1000:]

subplot_groups = [
    [1, 2],  # Subplot 7: series 11, 12, and 13
    [3, 4],  # Subplot 7: series 11, 12, and 13
    [5],  # Subplot 7: series 11, 12, and 13
    [6],  # Subplot 8: series 14, 15, and 16
    [11],
]

series_labels = [
    r"$\dot{r}_{x}$",
    r"$\dot{r}_{z}$",
    r"$r_{x}$",
    r"$r_{z}$",
    r"$\dot{\theta}$",
    r"$\theta$",
    r"$\theta$",
]

subplot_titles = [r"CoM lin. vel", r"CoM pos", r"Base ang. vel", r"Base ori", r"foot"]
subplot_ylabel = ["m/s", "m", "rad/s", "rad", "rad"]


fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
axes = axes.flatten()

label_idx = 0
# Plot each group in its respective subplot
for idx, (group, title, label) in enumerate(zip(subplot_groups, subplot_titles, subplot_ylabel)):
    ax = axes[idx]  # Current subplot
    for series in group:
        ax.plot(t, data_series[series - 1], linestyle="-", label=f"{series_labels[label_idx]}")
        label_idx += 1  # Move to the next label
    ax.set_ylabel(label)
    ax.set_title(title)  # Set title from the list
    ax.grid(True)
    ax.legend()

# Remove unused subplots
for ax in axes[len(subplot_groups) :]:
    fig.delaxes(ax)

# Add a shared X-axis label
plt.xlabel("Time")
ax.set_xlim(t[0], t[-1])
# Adjust layout
plt.tight_layout()


subplot_groups = [
    [7],
    [8],
    [9],
]

series_labels = [
    r"$\tau_{1}$",
    r"$\tau_{2}$",
    r"$\tau_{3}$",
]

subplot_titles = [r"Joint 1", r"Joint 2", r"Joint 3"]
subplot_ylabel = ["Nm", "Nm", "Nm"]


fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
axes = axes.flatten()

label_idx = 0
# Plot each group in its respective subplot
for idx, (group, title, label) in enumerate(zip(subplot_groups, subplot_titles, subplot_ylabel)):
    ax = axes[idx]  # Current subplot
    for series in group:
        ax.plot(t, data_series[series - 1], linestyle="-", label=f"{series_labels[label_idx]}")
        label_idx += 1  # Move to the next label
    ax.set_ylabel(label)
    ax.set_title(title)  # Set title from the list
    ax.grid(True)
    ax.legend()

# Remove unused subplots
for ax in axes[len(subplot_groups) :]:
    fig.delaxes(ax)

# Add a shared X-axis label
plt.xlabel("Time")
ax.set_xlim(t[0], t[-1])
# Adjust layout
plt.tight_layout()

plt.show()
