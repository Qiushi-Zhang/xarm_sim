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
import numpy as np 
# pin_model = pin.buildModelFromUrdf("/home/qiushi/workspace/xarm_sim/ur_urdf/urdf/ur5e.urdf")
pin_model = pin.buildModelFromUrdf("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/urdf/iiwa.urdf")
pin_data = pin_model.createData()


mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/xml/iiwa.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002
mj_model.opt.gravity = np.zeros(3)
mj_model.opt.gravity = np.array([0,0, -9.81])
# eeid_mj = 8 
eeid_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "L7")
eeid_pin = pin_model.getFrameId("L7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)

nq = pin_model.nq
# eeid_pin2 = pin_model.getFrameId("L4")

# mj_data.qpos, mj_data.qvel = np.zeros(7), np.zeros(7)
# mj_data.qpos[3] = np.pi/2 
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
# T_test[:3,3] = np.array([-0.2, -0.3, 0.3])

# KP = np.diag([900, 2500, 2500, 150, 150, 188])  # Stiffnes
# KD = np.diag([60, 100, 100, 0.6, 0.6, 0.8])  # Damping


dt = 0.002

x_desired_i = T_goal[:3,3]+ np.array([0.2, 0, 0.03])
# x_desired2 = T_goal2[:3,3]+np.array([-0.2,0, 0])*0
v_desired = np.zeros(6)
Ree_des = T_goal[:3,:3]
print("REE_desired", Ree_des)



sim_time = 0 

T_init = T_test 
T_init[:3,:3] = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
q_init = robot_utils.compute_IK_full(pin_model,pin_data, q_init, T_init, eeid_pin)
mj_data.qpos = q_init
theta_init = np.arctan2(T_init[1,3],T_init[0,3])
# print(theta)

R = 0.1
x, y, z = T_init[0,3], T_init[1,3], T_init[2,3]

P = 200
D = 5

qs = []
# while mj_data.time<2:
#     t = mj_data.time
#     theta = 2*np.pi*t
#     q, v = mj_data.qpos, mj_data.qvel 
#     T_delt = np.eye(4)
#     T_init[:3,3] = np.array([x+R*np.cos(theta), y+R*np.sin(theta), z])
#     T_des = T_init
#     q_des = robot_utils.compute_IK_full(pin_model, pin_data, q, T_des, eeid_pin)
#     qs.append(q_des.copy())
#     v_des = (q_des-q)/dt
#     # tau = P*(q_des - q) + D*(v_des - v)
#     # mj_data.ctrl = tau 
#     mj_data.qpos = q_des 
#     # qs.append(q_des.copy())
#     mujoco.mj_step(mj_model,mj_data)
    
#     viewer.render()
#     T_ee = robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, q)
#     visualizer.visualize_frame(viewer, T_ee, 0.1)

# qs_array = np.vstack(qs)                     # shape (N_steps, nq)
# vel_array = np.diff(qs_array, axis=0) / dt  

# time = np.arange(qs_array.shape[0]) * dt

# # plt.figure()
# # for j in range(qs_array.shape[1]):
# #     plt.plot(time, qs_array[:, j], label=f"Joint {j+1}")
# # plt.xlabel("Time [s]")
# # plt.ylabel("Joint Angle [rad]")
# # plt.title("Joint Trajectories (qs)")
# # plt.legend()
# # plt.grid(True)
# # plt.show()

idx = 0 
while mj_data.time<4:

        t= mj_data.time
    
        q = mj_data.qpos
        v = mj_data.qvel
        # q_des = qs_array[idx]
        # v_des = vel_array[idx]*0
        q_des = q_init+np.array([0,0,0.5*np.sin(2*np.pi*t), 0.5*np.sin(2*np.pi*t), 0.5*np.sin(2*np.pi*t), 0,0])
        v_des = np.array([0,0,0.5*2*np.pi*np.cos(2*np.pi*t),0.5*2*np.pi*np.cos(2*np.pi*t),0.5*2*np.pi*np.cos(2*np.pi*t),0,0])
        # tau = P.dot(q_des - q) + D.dot(v_des - v)
        a = P*(q_des - q) + D*(v_des - v)
        mj_data.ctrl = a 
        print(a)
        # mj_data.qpos = q_des 
        mujoco.mj_step(mj_model,mj_data)
        
        viewer.render()
        T_ee = robot_utils.forward_kinematics(pin_model, pin_data, eeid_pin, q)
        visualizer.visualize_frame(viewer, T_ee, 0.1)
        idx+=1
        



    


# robot_utils.compute_IK_full(pin_model, pin_data, q_init, T_test, eeid_pin)

# q = q_init 
# for i in range(10):
#     T_delt = np.eye(4)
#     T_delt[:3,3] = np.array([0.01*i, 0, 0.01*i])
#     T_des = T_test@T_delt 
   
#     # print("value of q", q) 
#     q = robot_utils.compute_IK_full(pin_model, pin_data, q, T_des, eeid_pin)
#     # print("value of T_des", T_des, "value of T_actual", robot_utils.forward_kinematics(pin_model,pin_data,eeid_pin, q))
    
    


