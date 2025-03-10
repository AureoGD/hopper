import pybullet as p
import pybullet_data
import numpy as np
import os
from icecream import ic
import time
from math import sin, cos
import matplotlib.pyplot as plt

import model_matrices

SIM_TIME = 0.001

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../model/hopper.urdf")

JOINT_ST_LIST = (0, 1, 2, 4, 6, 8)
AC_JOINT_LIST = (4, 6, 8)

valor = 0
mdl = model_matrices.ModelMatrices()
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
# p.setGravity(0, 0, 0)
p.setTimeStep(SIM_TIME)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
p.changeDynamics(plane, -1, lateralFriction=1.0)

model = p.loadURDF(MODEL_PATH, [0, 0, 1.5], [0, 0, 0, 1])
# model = p.loadURDF(MODEL_PATH, [0, 0, 1], [0, 0, 0, 1])
number_joints = p.getNumJoints(model)

for idx in range(number_joints - 1):
    p.setJointMotorControl2(model, idx, p.VELOCITY_CONTROL, force=0)

for idx in range(len(AC_JOINT_LIST)):
    p.setJointMotorControl2(model, AC_JOINT_LIST[idx], p.TORQUE_CONTROL, force=0)


q_aux = np.zeros((len(JOINT_ST_LIST), 1), dtype=np.float64)
dq_aux = np.zeros((len(JOINT_ST_LIST), 1), dtype=np.float64)

q = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)
dq = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)
qr = np.zeros((len(AC_JOINT_LIST), 1), dtype=np.float64)
b = np.zeros((2, 1), dtype=np.float64)
db = np.zeros((2, 1), dtype=np.float64)
rw = np.zeros((2, 1), dtype=np.float64)
th = np.zeros((1, 1), dtype=np.float64)
dth = np.zeros((1, 1), dtype=np.float64)
tau = np.zeros((3, 1), dtype=np.float64)
stau = np.zeros((6, 1), dtype=np.float64)

t1 = np.zeros((3, 1), dtype=np.float64)
t2 = np.zeros((3, 1), dtype=np.float64)


q0 = np.array([0, 0, 0, np.pi * 32.5 / 180, np.pi * -60 / 180, np.pi * 32.5 / 180])
for idx in range(len(JOINT_ST_LIST)):
    p.resetJointState(model, JOINT_ST_LIST[idx], q0[idx])

p.resetJointState(model, 1, 0.85)
# p.resetJointState(model, 1, 0)

p.enableJointForceTorqueSensor(model, 8, enableSensor=1)
p.enableJointForceTorqueSensor(model, 6, enableSensor=1)
p.enableJointForceTorqueSensor(model, 4, enableSensor=1)


qr[0, 0] = q0[3]
qr[1, 0] = q0[4]
qr[2, 0] = q0[5]

N_int = 10000
ch = 3
# [t, rw, drw, b, db, th, dth, q, dq]

st_his = np.zeros((20, N_int * ch))
dq_star = np.zeros((3, N_int * ch))
# ft_star = np.zeros((3, N_int * ch))

ft_star = np.zeros((9, N_int * ch))


print(mdl.m * 9.81)
print((mdl.m0 + mdl.m1 + mdl.m2) * 9.81)

count = 0
flag = True

kp = 200 * np.identity(3)
kd = 2 * np.sqrt(kp) * np.identity(3)
V_MAX_TAU = 50 * np.ones((3, 1), dtype=np.float64)


def get_joit_states():
    for idx in range(len(JOINT_ST_LIST)):
        q_aux[idx], dq_aux[idx], forces, tau_ = p.getJointState(model, JOINT_ST_LIST[idx])
        stau[idx, 0] = forces[4]

    # ax_data = p.getLinkState(model, 4, computeLinkVelocity=1)

    # base linear pos and vel

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

    mdl.update_robot_states(q=q, dq=dq)
    mdl.update_kinematics()
    rot = rotY()

    rb = rot @ mdl.update_com_pos()
    rw[0, 0] = b[0, 0] + rb[0, 0]
    rw[1, 0] = b[1, 0] + rb[2, 0]

    # rb = mdl.update_com_pos()
    # rw[0, 0] = rb[0, 0]
    # rw[1, 0] = rb[2, 0]


def estimate_dq():
    rot = rotY()
    J_com = rot @ mdl.com_jacobian()
    J_cont = rot @ mdl.J_ankle

    r = rot @ mdl.update_com_pos()
    pc = rot @ mdl.HT_ankle[0:3, 3].reshape((3, 1))
    # rr1 = np.array([[b[0, 0]], [0], [b[1, 0]]])
    # vet = (rr1 + pc) - (rr1 + r)
    vet = pc - r
    gamma = J_com - J_cont

    # estimation of the joints velocity using the Jacobains and CoM velocities
    alpha = np.array(
        [
            [1, 0, vet[2, 0]],
            [0, 1, -vet[0, 0]],
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

    dq_est = np.linalg.inv(beta) @ alpha @ np.array([[st_his[3, count]], [st_his[4, count]], [dth[0, 0]]])
    dq_star[0, count] = dq_est[0, 0]
    dq_star[1, count] = dq_est[1, 0]
    dq_star[2, count] = dq_est[2, 0]

    # external force estimation using the transpose Jacobian.

    J_heel = mdl.J_ankle
    m1 = np.array([[J_heel[0, 0], J_heel[0, 1], J_heel[0, 2]], [J_heel[2, 0], J_heel[2, 1], J_heel[2, 2]], [1, 1, 1]])
    # f1 = -np.linalg.inv(m1.transpose()) @ t1
    f1 = -m1.transpose() @ t1
    f11 = np.array([[rot[0, 0], rot[0, 2]], [rot[2, 0], rot[2, 2]]]) @ f1[0:2, 0]
    ft_star[0:2, count] = f1[0:2, 0]
    # ft_star[0:2, count] = f1[:, 0]
    #
    ft_star[2, count] = np.array([vet[2, 0], -vet[0, 0]]) @ f11
    ft_star[3, count] = f1[2, 0]
    ft_star[4, count] = ft_star[2, count] - ft_star[3, count]

    # ft_star[5, count] = np.array([vet[2, 0], -vet[0, 0]]) @ f1[0:2, 0]
    # ft_star[6:9, count] = t2[:, 0]

    inertia = mdl.update_inertia_tensor()
    inv_inertia = rot * np.linalg.inv(inertia)

    if count != 0:
        ft_star[5, count] = ft_star[5, count - 1] + 0.001 * inv_inertia[1, 1] * ft_star[4, count]
        ft_star[6, count] = ft_star[6, count - 1] + 0.001 * ft_star[5, count]

    # f2 = -np.linalg.inv(m1.transpose()) @ t2
    # ft_star[6, count] = np.array([vet[2, 0], -vet[0, 0]]) @ f2[0:2, 0]
    # result = -np.linalg.inv(m1.transpose()) @ (t1 + t2)[:, 0]
    # ft_star[6, count] = np.array([vet[2, 0], -vet[0, 0]]) @ result[0:2] - (t1 + 2 * t2)[2, 0]
    # ft_star[7, count] = (t1 + 2 * t2)[2, 0]
    # ft_star[8, count] = ft_star[6, count] + ft_star[7, count]


def write_history():
    # [t, rw, drw, b, db, th, dth, q, dq]
    st_his[0, count] = count * SIM_TIME
    st_his[1, count] = rw[0, 0]
    st_his[2, count] = rw[1, 0]

    if count == 0:
        st_his[3, count] = 0
        st_his[4, count] = 0
    else:
        st_his[3, count] = (st_his[1, count] - st_his[1, count - 1]) / SIM_TIME
        st_his[4, count] = (st_his[2, count] - st_his[2, count - 1]) / SIM_TIME

    st_his[5, count] = b[0, 0]
    st_his[6, count] = b[1, 0]
    st_his[7, count] = db[0, 0]
    st_his[8, count] = db[1, 0]
    st_his[9, count] = th[0, 0]
    st_his[10, count] = dth[0, 0]
    st_his[11, count] = q[0, 0]
    st_his[12, count] = q[1, 0]
    st_his[13, count] = q[2, 0]
    st_his[14, count] = qr[0, 0]
    st_his[15, count] = qr[1, 0]
    st_his[16, count] = qr[2, 0]
    st_his[17, count] = dq[0, 0]
    st_his[18, count] = dq[1, 0]
    st_his[19, count] = dq[2, 0]

    estimate_dq()


def low_lvl_control():
    tau1 = np.matmul(kp, (qr - q)) - np.matmul(kd, dq)
    tau2 = comp_tau()
    # porque o sinal deve ser invertido?
    tau_star = np.clip(tau1 - tau2, -V_MAX_TAU, V_MAX_TAU)

    for idx in range(len(AC_JOINT_LIST)):
        p.setJointMotorControl2(
            model,
            AC_JOINT_LIST[idx],
            p.TORQUE_CONTROL,
            force=tau_star[idx, 0],
        )

    return tau1, tau2


def rotY():
    return np.array([[cos(th[0, 0]), 0, sin(th[0, 0])], [0, 1, 0], [-sin(th[0, 0]), 0, cos(th[0, 0])]])


def comp_tau():
    rot = rotY()

    J_com1, J_com2, J_com3 = mdl.update_CoM_jacobians(qr)

    J_com1 = rot @ J_com1
    J_com2 = rot @ J_com2
    J_com3 = rot @ J_com3

    # J_com1 = rot @ mdl.J_com1
    # J_com2 = rot @ mdl.J_com2
    # J_com3 = rot @ mdl.J_com3

    tau_g = (J_com1.transpose() * mdl.m1 + J_com2.transpose() * mdl.m2 + J_com3.transpose() * mdl.m3) @ np.array(
        [[0], [0], [-9.81]]
    )

    return tau_g


while count <= N_int * ch - 1:
    get_joit_states()
    t1, t2 = low_lvl_control()
    print(t1 - t2)
    write_history()
    p.stepSimulation()
    count += 1
    # if count % N_int == 0:
    #     if flag:
    #         # qr[0, 0] = np.pi * 45 / 180
    #         # qr[1, 0] = np.pi * -105 / 180
    #         # qr[2, 0] = np.pi * 55 / 180
    #         qr[0, 0] = np.pi * 25 / 180
    #         qr[1, 0] = np.pi * -60 / 180
    #         qr[2, 0] = np.pi * 25 / 180

    #         flag = False
    #     else:
    #         # qr[0, 0] = np.pi * 50 / 180
    #         # qr[1, 0] = np.pi * -110 / 180
    #         # qr[2, 0] = np.pi * 60 / 180
    #         qr[0, 0] = np.pi * 30 / 180
    #         qr[1, 0] = np.pi * -50 / 180
    #         qr[2, 0] = np.pi * 20 / 180

    #         flag = True
    # time.sleep(SIM_TIME)

p.disconnect()

# Figures

t = st_his[0, 1000:]  # First row (time)
data_series = st_his[1:, 1000:]  # Subsequent rows (data series)

# [t, rw, drw, b, db, th, dth, q, qr, dq]

subplot_groups = [
    [5, 1],  # Subplot 1: x
    [3, 7],  # Subplot 2: dx
    [6, 2],  # Subplot 3: z
    [4, 8],  # Subplot 4: dz
    [9],  # Subplot 5: series 9
    [10],  # Subplot 6: series 10
]

series_labels = [
    r"$b$",
    r"$r$",
    r"$b$",
    r"$r$",
    r"$b$",
    r"$r$",
    r"$b$",
    r"$r$",
    r"$\theta$",
    r"$d\theta$",
]

subplot_titles = ["X pos", "X vel", "Z pos", "Z vel", r"$\theta$", r"$\dot{\theta}$"]
subplot_ylabel = ["m", "m/s", "m", "m/s", "rad", "rad/s"]


# Create subplots with 2 columns and 4 rows
fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

# Flatten the axes array for easier iteration
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


# Joint angles figures

subplot_groups2 = [
    [11, 14],  # Subplot 7: series 11, 12, and 13
    [12, 15],  # Subplot 7: series 11, 12, and 13
    [13, 16],  # Subplot 7: series 11, 12, and 13
    [17, 18, 19],  # Subplot 8: series 14, 15, and 16
]

series_labels2 = [
    r"$q_{1}$",
    r"$q_{r,1}$",
    r"$q_{2}$",
    r"$q_{r,2}$",
    r"$q_{3}$",
    r"$q_{r,3}$",
    r"$dq_{1}$",
    r"$dq_{2}$",
    r"$dq_{3}$",
]

subplot_titles2 = [r"$q_{1}$", r"$q_{2}$", r"$q_{3}$", r"$\dot{q}$"]
subplot_ylabel2 = ["rad", "rad", "rad", "rad/s"]


fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
axes = axes.flatten()

label_idx = 0
# Plot each group in its respective subplot
for idx, (group, title, label) in enumerate(zip(subplot_groups2, subplot_titles2, subplot_ylabel2)):
    ax = axes[idx]  # Current subplot
    for series in group:
        ax.plot(t, data_series[series - 1], linestyle="-", label=f"{series_labels2[label_idx]}")
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


data_series3 = np.concatenate((st_his[17:, 1000:], dq_star[:, 1000:]), axis=0)

subplot_groups3 = [
    [1, 4],  # dq1
    [2, 5],  # dq2
    [3, 6],  # dq3
]

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
axes = axes.flatten()

label_idx = 0
for idx, (group) in enumerate(subplot_groups3):
    ax = axes[idx]  # Current subplot
    for series in group:
        ax.plot(t[:], data_series3[series - 1], linestyle="-")
        label_idx += 1  # Move to the next label

    ax.grid(True)
ax.set_xlim(t[0], t[-1])


data_series4 = ft_star[0:9, 1000:]

subplot_groups4 = [
    [1, 2],  # forces
    [3, 4, 5],
    [6],
    [7, 8, 9],
]

fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
axes = axes.flatten()

label_idx = 0
for idx, (group) in enumerate(subplot_groups4):
    ax = axes[idx]  # Current subplot
    for series in group:
        ax.plot(t[:], data_series4[series - 1], linestyle="-")
        label_idx += 1  # Move to the next label

    ax.grid(True)
ax.set_xlim(t[0], t[-1])


# Show the plot
plt.show()
