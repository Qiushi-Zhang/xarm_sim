import mujoco 
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import robot_utils
from utils import visualizer
from pinocchio.visualize import MeshcatVisualizer
import example_robot_data
# pin_model = pin.buildModelFromUrdf("../xarm_description/robots/xarm7.urdf")
pin_model = pin.buildModelFromUrdf("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/trifinger/nyu_finger_triple0.urdf")
pin_data = pin_model.createData()

mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mim_robots/python/mim_robots/robots/trifinger/nyu_finger_triple0.xml")
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = 0.002

# eeid_mj = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "L7")
eeid_pin = pin_model.getFrameId("finger0_tip_link")
viewer = mujoco_viewer.MujocoViewer(mj_model,mj_data)

nq = pin_model.nq
q_init, v_init = np.zeros(nq), np.zeros(nq)
q_init[2] = np.pi/2
T_goal = robot_utils.forward_kinematics(pin_model,pin_data,eeid_pin, q_init)
mj_data.qpos, mj_data.qvel = q_init.copy(), v_init.copy()
q, v = q_init, v_init
P = 200
D = 0.1
dt = 0.002

x_desired_i = T_goal[:3,3]
v_desired = np.zeros(6)
Ree_des = T_goal[:3,:3]
print("REE_desired", Ree_des)



sim_time = 0 

while mj_data.time < 10:
    t = mj_data.time

   
    
    viewer.render()
    visualizer.visualize_frame(viewer, T_ee)
    visualizer.visualize_frame(viewer, T_des)
    mujoco.mj_step(mj_model, mj_data)
    
    viewer.render()
    
    


