import mujoco 
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

# import mujoco.msh2obj_test
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import visualizer




# mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mujoco_menagerie/shadow_hand/right_hand.xml")
mj_model = mujoco.MjModel.from_xml_path("/home/qiushi/workspace/mujoco_menagerie/shadow_hand/left_hand.xml")
mj_data = mujoco.MjData(mj_model)
viewer = mujoco_viewer.MujocoViewer(mj_model, mj_data) 



mj_model.opt.timestep = 0.002
eeid = mj_data.body("lh_ffdistal").id
# print(eeid.type)


def compute_point_jacobian(model, data, body_id, local_pos):
    """
    Compute the Jacobian for a point specified in local (body) coordinates.
    
    Args:
        model (mujoco.MjModel): Loaded MuJoCo model.
        data (mujoco.MjData): MuJoCo data associated with the model.
        body_name (str): Name of the body on which the point lies.
        local_pos (np.ndarray): 3D coordinates of the point in the body's local frame.
        
    Returns:
        (J_pos, J_rot):
            J_pos: (3 x nv) translational Jacobian (maps joint velocities to the point's linear velocity).
            J_rot: (3 x nv) rotational Jacobian (maps joint velocities to the point's angular velocity).
    """
    # Get body id from name

    
    # Allocate space for Jacobians (3 x nv flattened)
    jacp = np.zeros((3, model.nv)) # translational part
    jacr = np.zeros((3, model.nv))  # rotational part
    
    # Compute the translational and rotational Jacobians for the point
    mujoco.mj_jac(model, data, jacp, jacr, local_pos,body_id)
    
    # Reshape from 1D [3*nv] to 2D [3, nv]
    J_pos = jacp.reshape((3, model.nv))
    J_rot = jacr.reshape((3, model.nv))
    
    return J_pos, J_rot

while mj_data.time < 20:


    R = mj_data.xmat[eeid].reshape(3, 3)    # 3×3, row-major
    p = mj_data.xpos[eeid]  
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = p

    T_trans = np.eye(4)
    T_trans[:3,3] = np.array([0,0,0.005])
    T_ee = T@T_trans
    
    J_pos, J_rot = compute_point_jacobian(mj_model, mj_data, eeid, T_ee[:3,3])


    # print(J_pos)
    F = np.array([0,0,5])
    tau = J_pos.T@F
    index_joint_names = ["lh_FFJ4",   # MCP
                     "lh_FFJ3",   # PIP
                     "lh_FFJ2",
                     "lh_FFJ1"]   # DIP

    index_joint_ids = [
        mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in index_joint_names
    ]

    # sanity-check: make sure nothing came back as -1
    if any(jid < 0 for jid in index_joint_ids):
        missing = [n for n, jid in zip(index_joint_names, index_joint_ids) if jid < 0]
        raise ValueError(f"Joint(s) not found: {missing}")

    print("Index-finger joint IDs:", index_joint_ids)
    print(tau[index_joint_ids])
    print(tau)
            

    mj_data.qpos[2:6] = np.array([np.pi/6, np.pi/6,np.pi/6,np.pi/6])


    # mj_data.qpos[1] = np.pi/3
        

    # mj_data.qvel= mj_data.qvel*0

    mujoco.mj_step(mj_model, mj_data)
    

    # visualizer.add_marker(viewer, x_des0, 0.01)
    # print(T_disk[:3,3])
    visualizer.visualize_frame(viewer, T_ee)
    # visualizer.visualize_frame(viewer, T_world, 0.1)
    # print(mj_data.qpos[-1]/np.pi*180)
    viewer.render()