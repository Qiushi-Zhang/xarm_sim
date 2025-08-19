import mujoco 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
# from utils import pin_utils
from utils import visualizer

# build pinocchio model for robot kinematics and dynamics computations 
pin_model = pin.buildModelFromUrdf("../xarm_description/robots/xarm7.urdf")
pin_data = pin_model.createData()

# build mujoco model for simulation 
mj_model = mujoco.MjModel.from_xml_path("../xarm_description/xarm_mj/xarm7_nohand.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002

eeid_pin = pin_model.getFrameId("link7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)
print("successfully build viewer")
# nq = pin_model.nq
nq = mj_model.nq
print(nq)

mode = "PD_test"
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
        # T = pin_utils.forward_kinematics(pin_model,pin_data, eeid_pin, q)
        # visualizer.visualize_frame(viewer,T)
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()

if mode = "impedance "

