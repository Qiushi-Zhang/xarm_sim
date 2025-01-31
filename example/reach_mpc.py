import mujoco 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import pin_utils
from utils import visualizer 
from mpc.solver.reach_ddp import solver_ddp_reach
import glfw




pin_model = pin.buildModelFromUrdf("../xarm_description/robots/xarm7.urdf")
pin_data = pin_model.createData()
urdf_path = "../xarm_description/robots/xarm7.urdf"

mj_model = mujoco.MjModel.from_xml_path("../xarm_description/xarm_mj/xarm7_nohand.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002
eeid_mj = 8 
eeid_pin = pin_model.getFrameId("link7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)
nq = pin_model.nq

mode = "mpc_test"
# mode = "traj_opt_test"

T = 50
DT = 0.01
H = 3
K = 5 


P = 1000
D = 50
i = 0

xee = np.zeros((3,T+1))



print(mj_data.mocap_pos)
if mode == "mpc_test":
    while mj_data.time < 20:
        
        t = mj_data.time
        q, v = mj_data.qpos, mj_data.qvel 
        x_goal = np.array([0.4,-0.4,0.5])
        x_goal = mj_data.mocap_pos[0]
        x = np.concatenate([q,v])
        if i%(H*K) == 0:
            solver = solver_ddp_reach(urdf_path, T, DT)
            xs, us = solver.generate_ddp(x,x_goal)
            i = 0 
            for i in range(T+1):
                qs = xs[i][:nq]
                xee[:,i] = pin_utils.forward_kinematics(pin_model,pin_data, eeid_pin,qs)[:3,3]

            
            visualizer.visualize_trajectory(viewer, xee, (0,0,1,1))
        if i%(K) == 0:
            j = int(i/K)
            xf = xs[j]
        q_des, dq_des = xf[:nq], xf[nq:]
        tau = P*(q_des-q)+D*(dq_des-v)+pin.rnea(pin_model,pin_data,q, np.zeros(nq), np.zeros(nq))
        i+=1

        mj_data.ctrl = tau 
        mujoco.mj_step(mj_model,mj_data)
        viewer.render()




# for i in range(30):
#     q, v = np.random.rand(7), np.random.rand(7)
#     x_goal = np.array([0.4, -0.4, 0.4])
#     x = np.concatenate([q,v])
#     solver = solver_ddp_reach(urdf_path,T,DT)
#     xs, us = solver.generate_ddp(x,x_goal)
#     n_col = len(us)
#     xf = xs[-1]
#     qf = xf[:nq]
#     x_f = pin_utils.forward_kinematics(pin_model, pin_data, eeid_pin, qf)[:3,3]
#     print("EE error", np.linalg.norm(x_f-x_goal), "m")


if mode == "traj_opt_test":
    import time 
    q, v = mj_data.qpos, mj_data.qvel
    x_goal = np.array([0.4, -0.4, 0.4])
    x = np.concatenate([q,v])
    solver = solver_ddp_reach(urdf_path,T,DT)
    et = time.time()
    xs, us = solver.generate_ddp(x,x_goal)
    st = time.time()
    print("time required to solve", st-et)
    n_col = len(us)

    xf = xs[-1]
    qf = xf[:nq]

    print("reached EE position", pin_utils.forward_kinematics(pin_model, pin_data, eeid_pin, qf)[:3,3])
    for i in range(n_col):
        xi = xs[i]
        q = xi[:7]
        dq = xi[:-7]
        u = us[i]
        mj_data.qpos, mj_data.qvel, mj_data.ctrl = q,dq,u
        mujoco.mj_step(mj_model,mj_data)

        viewer.render()


