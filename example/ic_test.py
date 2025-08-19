import mujoco 
import sys
import os

import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import robot_utils
from utils import visualizer
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
import matplotlib.pyplot as plt
from collections import deque

# pin_model = pin.buildModelFromUrdf("/home/qiushi/workspace/xarm_sim/ur_urdf/urdf/ur5e.urdf")
pin_model = pin.buildModelFromUrdf("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/urdf/iiwa.urdf")
pin_data = pin_model.createData()

mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/xml/iiwa.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002
mj_model.opt.gravity = np.zeros(3)
mj_model.opt.gravity = np.array([0,0, -9.81])*0
eeid_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "L7")
eeid_pin = pin_model.getFrameId("L7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)

nq = pin_model.nq

q_init, v_init = np.zeros(nq), np.zeros(nq)
q_init[1] = -np.pi/2
q_init[2] = np.pi/2
q_init[3] = np.pi/2
T_goal = robot_utils.forward_kinematics(pin_model,pin_data,eeid_pin, q_init)
# T_goal2  = robot_utils.forward_kinematics(pin_model,pin_data, eeid_pin2, q_init)
mj_data.qpos, mj_data.qvel = q_init.copy(), v_init.copy()
q, v = q_init, v_init
P = 500
D = 5
KP = np.diag([10000,10000,10000,300,300,300])
KD = np.diag([400,400,400, 2,2,2.5])


T_test = T_goal 


dt = 0.002

x_desired_i = T_goal[:3,3]+ np.array([0.2, 0, 0.03])
v_desired = np.zeros(6)
Ree_des = T_goal[:3,:3]
print("REE_desired", Ree_des)



sim_time = 0 

T_init = T_test 
T_init[:3,:3] = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
q_init = robot_utils.compute_IK_full(pin_model,pin_data, q_init, T_init, eeid_pin)
mj_data.qpos = q_init
q, v = q_init, v_init
P = 500
D = 5
KP = np.diag([10000,10000,10000,300,300,300])
KD = np.diag([400,400,400, 2,2,2.5])


# KP = np.diag([900, 2500, 2500, 150, 150, 188])  # Stiffnes
# KD = np.diag([60, 100, 100, 0.6, 0.6, 0.8])  # Damping


dt = 0.002

x_desired_i = T_goal[:3,3]
# x_desired2 = T_goal2[:3,3]+np.array([-0.2,0, 0])*0
v_desired = np.zeros(6)
Ree_des = T_goal[:3,:3]
print("REE_desired", Ree_des)


# ===================== GMO SETUP (before the while loop) =====================
import matplotlib.pyplot as plt

# Observer gain (diag matrix L). Start modest; raise for faster response.
L_vals = np.array([50.0]*nq)          # tune per joint if needed
L_mat  = np.diag(L_vals)

# Integral state for ∫(tau_bar + r)ds and previous r
z_int  = np.zeros(nq)
r_prev = np.zeros(nq)

# Live plot
plt.ion()
fig, ax = plt.subplots(figsize=(9,4))
lines = []
for i in range(nq):
    (ln,) = ax.plot([], [], lw=1.4, label=f'τ_ext[{i}]')
    lines.append(ln)
ax.set_xlabel('time [s]'); ax.set_ylabel('external torque [Nm]')
ax.set_title('GMO external torque estimate r(t)')
ax.grid(True); ax.legend(ncol=2, fontsize=8)
t_hist = []
r_hist = [[] for _ in range(nq)]



print("current gravity value", mj_model.opt.gravity)
while sim_time < 10:
   
    t = mj_data.time

    q, v = mj_data.qpos, mj_data.qvel 
    g = pin.computeGeneralizedGravity(pin_model, pin_data, q)
    
    T_ee = robot_utils.forward_kinematics(pin_model,pin_data,eeid_pin, q)
    x_ee = T_ee[:3,3]
    R_ee = T_ee[:3,:3]
    T_ee2 = robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, q)
    x_ee2 = T_ee2[:3,3]
    theta = np.pi/6*np.sin(2*np.pi*t)*0 
    Ree_des_rot = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])
    # Ree_des_rot = np.array([
    # [1.0,           0.0,            0.0],
    # [0.0,  np.cos(theta), -np.sin(theta)],
    # [0.0,  np.sin(theta),  np.cos(theta)]
    # ])
    # Ree_des_rot = np.array([
    # [ np.cos(theta), 0.0, np.sin(theta)],
    # [ 0.0,           1.0,          0.0],
    # [-np.sin(theta), 0.0, np.cos(theta)]
    # ]
    Ree_des_new = Ree_des_rot@Ree_des
    
    theta = 2*np.pi*t
    
    x_des = x_desired_i+np.array([0.05,0,0])*np.sin(theta)

    J_ee = robot_utils.compute_jacobian(pin_model, pin_data, eeid_pin, q) 
    # J_ee2 = pin_utils.compute_jacobian(pin_model, pin_data, eeid_pin2, q)
    V_ee = J_ee@v
    x_err = np.concatenate([x_des - x_ee, pin.rpy.matrixToRpy(Ree_des_new @ R_ee.T)])
    # x_err = 
    # print(x_err)
    v_des = np.array([-0.5*0.4*np.pi*np.sin(2*np.pi*0.2*t), 0.5*0.4*np.pi*np.cos(2*np.pi*0.2*t), 0,0,0,0])*0
    v_err = v_desired - V_ee 

    tau = J_ee.T @ (KP@(x_err) +KD@(v_err))
    
    mj_data.ctrl = tau+g*0
    T_des = np.eye(4)
    T_des[:3,:3] = Ree_des_new
    T_des[:3,3] = x_des

        # ===================== GMO UPDATE (inside your while loop) =====================
    # 0) Make sure MuJoCo derived forces for this step are up-to-date
    mujoco.mj_forward(mj_model, mj_data)   # updates qfrc_actuator, etc.

    # 1) Read state and motor torque (Nm) from MuJoCo
    q = mj_data.qpos.copy()
    v = mj_data.qvel.copy()
    tau_motor = mj_data.qfrc_actuator.copy()   # joint torques from actuators (Nm)

    # 2) Get model terms from Pinocchio: M(q), C(q,v), g(q)
    #    - Pinocchio's C satisfies the standard property Mdot - 2C skew-symmetric,
    #      matching the observer's requirement; we need C^T @ v below.
    M = pin.crba(pin_model, pin_data, q)                  # inertia
    C = pin.computeCoriolisMatrix(pin_model, pin_data, q, v)  # Coriolis matrix
    g = pin.computeGeneralizedGravity(pin_model, pin_data, q)  # gravity
    # Frictionless sim: tau_f = 0
    tau_f = np.zeros_like(g)

    # 3) Compute bar{tau} = C^T v - g - tau_f + tau_motor  (paper’s definition)
    tau_bar = C.T @ v - g - tau_f + tau_motor

    # 4) Discrete-time integral state and residual
    #    z_k = z_{k-1} + dt * (tau_bar_{k-1} + r_{k-1})
    #    r_k = L * ( M(q_k) v_k - z_k )
    z_int = z_int + dt * (tau_bar + r_prev)   # semi-implicit works well at small dt
    p = M @ v                                 # generalized momentum
    r = L_mat @ (p - z_int)                   # <-- τ_external estimate at this step
    r_prev = r

    # (OPTIONAL sanity check against MuJoCo’s contact/constraint torques)
    # tau_truth = mj_data.qfrc_constraint.copy()
    # print("||r - truth|| =", np.linalg.norm(r - tau_truth))

    # 5) live plot (last 10 s window)
    t_hist.append(mj_data.time)
    for i in range(nq):
        r_hist[i].append(r[i])
        lines[i].set_data(t_hist, r_hist[i])
    t0 = max(0.0, mj_data.time - 10.0)
    ax.set_xlim(t0, t0 + 10.0)
    # autoscale y softly
    window_idx = [k for k,tk in enumerate(t_hist) if tk >= t0]
    if window_idx:
        i0 = window_idx[0]
        seg = np.array([h[i0:] for h in r_hist])
        yabs = float(np.max(np.abs(seg))) if seg.size else 1.0
        ax.set_ylim(-max(5.0, 1.2*yabs), max(5.0, 1.2*yabs))
    fig.canvas.draw(); fig.canvas.flush_events()

    # expose it to the rest of your code if you need it
    tau_external = r


    

    

  
    visualizer.visualize_frame(viewer, T_ee)
    visualizer.visualize_frame(viewer, T_des )
    mujoco.mj_step(mj_model, mj_data)
    
    viewer.render()
    
