import pinocchio as pin
import numpy as np

def forward_kinematics(model,data,frame_id,q):
    pin.forwardKinematics(model,data,q)
    pin.updateFramePlacements(model,data)
    T = data.oMf[frame_id].homogeneous
    return T 

def compute_frame_err(T1,T2):
    T1 = pin.SE3(T1)
    T2 = pin.SE3(T2)
    err = pin.log(T1.actInv(T2)).vector

    
    return err

def compute_jacobian(model,data,frame_id,q):
    J = pin.computeFrameJacobian(model,data, q,frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J 