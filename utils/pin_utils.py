import pinocchio as pin
import numpy as np

def forward_kinematics(model,data,frame_id,q):
    pin.forwardKinematics(model,data,q)
    pin.updateFramePlacements(model,data)
    T = data.oMf[frame_id].homogeneous
    return T 
