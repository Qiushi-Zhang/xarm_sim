import mujoco 
import sys
import os

import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import pin_utils
from utils import visualizer
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
# pin_model = pin.buildModelFromUrdf("../xarm_description/robots/xarm7.urdf")
pin_model = pin.buildModelFromUrdf("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/urdf/iiwa.urdf")
pin_data = pin_model.createData()

# robot = example_robot_data.load("ur5")
# viz = MeshcatVisualizer()
# robot.setVisualizer(viz)
# robot.initViewer(open = True)
# robot.loadViewerModel()
# pin_model = robot.model
# pin_data = robot.data

# mj_model = mujoco.MjModel.from_xml_path("../xarm_description/xarm_mj/xarm7_nohand.xml")
mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/xml/iiwa.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002
# eeid_mj = 8 
eeid_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "L7")
eeid_pin = pin_model.getFrameId("L7")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)

nq = pin_model.nq


# mj_data.qpos, mj_data.qvel = np.zeros(7), np.zeros(7)
# mj_data.qpos[3] = np.pi/2 
q_init, v_init = np.zeros(nq), np.zeros(nq)
q_init[3] = np.pi/2
T_goal = pin_utils.forward_kinematics(pin_model,pin_data,eeid_pin, q_init)
mj_data.qpos, mj_data.qvel = q_init.copy(), v_init.copy()
q, v = q_init, v_init
P = 2000
D = 10
dt = 0.002

x_desired = T_goal[:3,3]+np.array([0.1,0,0.2])
v_desired = np.zeros(6)
Ree_des = T_goal[:3,:3]
print("REE_desired", Ree_des)



sim_time = 0 

while sim_time < 10:
    # # mj_data.qpos = q.copy()
    # sim_time += dt
    # T_ee = pin_utils.forward_kinematics(pin_model, pin_data, eeid_pin, q)
    # pin.computeAllTerms(pin_model, pin_data, q, v)
    # J = pin.computeFrameJacobian(pin_model, pin_data, q, eeid_pin, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    # pin.updateFramePlacement(pin_model, pin_data,  eeid_pin)
    # x = pin_data.oMf[eeid_pin].translation
    # V = J@ v
    # Ree = pin_data.oMf[eeid_pin].rotation
    # x_err = np.concatenate([x_desired-x, pin.rpy.matrixToRpy(Ree_des@Ree.T)])
    # x_err = 
    # x_err = pin_utils.compute_frame_err(T_ee, T_goal)
    # v_err = v_desired - V 
    # F = P*(x_err) + D*(v_err)
    # tau = J.T@F 
    # g = pin.computeGeneralizedGravity(pin_model, pin_data, q)

    # a = pin.aba(pin_model, pin_data, q, v, tau+g)
    # v = v+dt*a
    # q = pin.integrate(pin_model,q, dt*v)

    # viz.display(q)

    q, v = mj_data.qpos, mj_data.qvel 
    g = pin.computeGeneralizedGravity(pin_model, pin_data, q)
    
#     # mj_data.qpos = q_init
#     # q, v = mj_data.qpos, mj_data.qvel 
#     # pin.computeAllTerms(pin_model,pin_data, q, v)
#     # pin.updateFramePlacement(pin_model,pin_data, eeid_pin)
    T_ee = pin_utils.forward_kinematics(pin_model,pin_data,eeid_pin, q)
    x_ee = T_ee[:3,3]
    R_ee = T_ee[:3,:3]

    J_ee = pin_utils.compute_jacobian(pin_model, pin_data, eeid_pin, q) 

    V_ee = J_ee@v
    x_err = np.concatenate([x_desired - x_ee, pin.rpy.matrixToRpy(Ree_des @ R_ee.T)])
    v_err = v_desired - V_ee 
    err = pin_utils.compute_frame_err(T_ee, T_goal)

    print(np.linalg.norm(err[3:]- x_err[:3]))
    
#     # F = P * pin_utils.compute_frame_err(T_ee,T_goal) + D*(np.zeros(6) - V_ee)
#     # x = T_ee [:3,3]
#     # x_des = T_goal[:3,3]
#     # F = P* (x_des-x)
#     # # print(pin_utils.compute_frame_err(T_ee,T_goal)[3:])
#     # s
#     # tau = np.zeros(7)
#     # mj_data.ctrl = tau  
    F = P*(x_err) + D*(v_err)
    tau = J_ee[3:,:].T @ F[3:] + g
    mj_data.ctrl = tau 
    visualizer.visualize_frame(viewer, T_ee)
    mujoco.mj_step(mj_model, mj_data)
    
    viewer.render()
    


