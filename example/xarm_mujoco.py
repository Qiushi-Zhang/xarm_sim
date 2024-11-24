import mujoco 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import pin_utils
from utils import visualizer

pin_model = pin.buildModelFromUrdf("../xarm_description/robots/xarm7.urdf")
pin_data = pin_model.createData()

mj_model = mujoco.MjModel.from_xml_path("../xarm_description/xarm_mj/xarm7_nohand.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002
eeid_mj = 8 
eeid_pin = pin_model.getFrameId("link7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)
nq = pin_model.nq

mode = "inverse_dynamics_PD_torque"
# mode = "hard_set_q"

# run a PD controller with torque as action 
if mode == "PD_test" : 
    P = 1000
    D = 50 
    while mj_data.time<10:
        t = mj_data.time
        q_des, dq_des, da_des = np.zeros(nq), np.zeros(nq), np.zeros(nq) 
        q_des[3], dq_des[3], da_des[3] = np.pi/2, 0, 0 
        q_des[6], dq_des[6], da_des[6] = np.pi/2*np.sin(2*np.pi*t), np.pi/2*np.cos(2*np.pi*t)*2*np.pi, -np.pi/2*np.sin(2*np.pi*t)*2*np.pi*2*np.pi 

        q ,dq= mj_data.qpos, mj_data.qvel 
        tau = P*(q_des-q)+D*(dq_des-dq) 

        mj_data.ctrl = tau 
        visualizer.visualize_ee(mj_data, viewer, eeid_mj )
        T = pin_utils.forward_kinematics(pin_model,pin_data, eeid_pin, q)
        visualizer.visualize_frame(viewer,T)
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()

# run a simulation where q is hardset to q_des 
elif mode == "hard_set_q":
     while mj_data.time<10:
        t = mj_data.time
        q_des, dq_des, da_des = np.zeros(nq), np.zeros(nq), np.zeros(nq) 
        q_des[3], dq_des[3], da_des[3] = np.pi/2, 0, 0 
        q_des[6], dq_des[6], da_des[6] = np.pi/2*np.sin(2*np.pi*t), np.pi/2*np.cos(2*np.pi*t)*2*np.pi, -np.pi/2*np.sin(2*np.pi*t)*2*np.pi*2*np.pi 

        mj_data.qpos = q_des 
        q = mj_data.qpos 
        visualizer.visualize_ee(mj_data, viewer, eeid_mj )
        T = pin_utils.forward_kinematics(pin_model,pin_data, eeid_pin, q)
        visualizer.visualize_frame(viewer,T)
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()

# run a inverse dynamics based PD controller with acceleration as action 
elif mode == "inverse_dynamics_PD_acc":
    P = 1000
    D = 50
    while mj_data.time<10:
        t = mj_data.time
        q_des, dq_des, da_des = np.zeros(nq), np.zeros(nq), np.zeros(nq) 
        q_des[3], dq_des[3], da_des[3] = np.pi/2, 0, 0 
        q_des[6], dq_des[6], da_des[6] = np.pi/2*np.sin(2*np.pi*t), np.pi/2*np.cos(2*np.pi*t)*2*np.pi, -np.pi/2*np.sin(2*np.pi*t)*2*np.pi*2*np.pi 
        
        # mj_data.qpos = np.zeros(pin_model.nq)
        # mj_data.qpos[3] = np.pi/2
        # mj_data.qpos[6] = np.pi/2*np.sin(2*np.pi*t)
        q ,dq= mj_data.qpos, mj_data.qvel 
        a_des = da_des+ P*(q_des-q)+D*(dq_des-dq) 
        tau = pin.rnea(pin_model,pin_data, q, dq, a_des) 
        mj_data.ctrl = tau 



        visualizer.visualize_ee(mj_data, viewer, eeid_mj )
        T = pin_utils.forward_kinematics(pin_model,pin_data, eeid_pin, q)
        visualizer.visualize_frame(viewer,T)
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()
# run a inverse dynamics based PD controller with acceleration as action 
elif mode == "inverse_dynamics_PD_torque":
    P = 1000
    D = 50
    while mj_data.time<10:
        t = mj_data.time
        q_des, dq_des, da_des = np.zeros(nq), np.zeros(nq), np.zeros(nq) 
        q_des[3], dq_des[3], da_des[3] = np.pi/2, 0, 0 
        q_des[6], dq_des[6], da_des[6] = np.pi/2*np.sin(2*np.pi*t), np.pi/2*np.cos(2*np.pi*t)*2*np.pi, -np.pi/2*np.sin(2*np.pi*t)*2*np.pi*2*np.pi 
        
        # mj_data.qpos = np.zeros(pin_model.nq)
        # mj_data.qpos[3] = np.pi/2
        # mj_data.qpos[6] = np.pi/2*np.sin(2*np.pi*t)
        q ,dq= mj_data.qpos, mj_data.qvel 
        
        tau_des = pin.rnea(pin_model,pin_data, q, dq, da_des) 
        tau = tau_des + P*(q_des-q)+D*(dq_des-dq) 
        mj_data.ctrl = tau 



        visualizer.visualize_ee(mj_data, viewer, eeid_mj )
        T = pin_utils.forward_kinematics(pin_model,pin_data, eeid_pin, q)
        visualizer.visualize_frame(viewer,T)
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()

# elif mode == "inverse_kinematics"





