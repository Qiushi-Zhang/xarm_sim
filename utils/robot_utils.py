import pinocchio as pin
import numpy as np
import time 

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
    #Computes the Jacobian for a frame placed at a specified origin and aligned with the world frame.
    J = pin.computeFrameJacobian(model,data, q,frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    return J 

def compute_impedance_control(pin_model, pin_data, eeid, q, v, kp, kd, T_des, V_des):
    T_ee = forward_kinematics(pin_model, pin_data, eeid, q)
    J_ee = compute_jacobian(pin_model, pin_data, eeid, q)
    V_ee = J_ee@v
    x_ee = T_ee[:3,3]
    R_ee = T_ee[:3,:3]

    x_des = T_des[:3,3]
    Ree_des = T_des[:3,:3]
    x_err = np.concatenate([x_des - x_ee, pin.rpy.matrixToRpy(Ree_des @ R_ee.T)])
    print(x_err)
    v_err = V_des - V_ee 

    tau = J_ee.T @ (kp @(x_err) + kd @(v_err))
    return tau 

def compute_IK_full(pin_model, pin_data, q0, T_des, eeid):
   t1 = time.time()
   max_iterations = 1000
   q = q0 

   T_des = pin.SE3(T_des[:3,:3], T_des[:3,3])
   for i in range(max_iterations):
       T_SF = forward_kinematics(pin_model, pin_data, eeid, q) 
       T_SF = pin.SE3(T_SF[:3,:3], T_SF[:3,3])
       J = pin.computeFrameJacobian(pin_model, pin_data, q, eeid, pin.ReferenceFrame.LOCAL)
       gain = 1 
       alpha = 0.01 
       err = gain*pin.log(T_SF.actInv(T_des)).vector 

       err_norm  = np.linalg.norm(err)
       if err_norm < 0.001: 
            print(f'done in {i} steps')
            print(f'we found joint positions:\n {q}')
            print(f'this gives an end-effector position of \n{T_SF}')
            print(f'the desired end-effector position was \n{T_des}')
            break
       if i==max_iterations-1:
            print('no good solution found')
            print(f'current guess is:\n {q}')
            print(f'this gives an end-effector position of \n{T_SF}')
            print(f'the desired end-effector position was \n{T_des}')
            break 
       epsilon = 10e-4
       pinv = J.T @ np.linalg.inv(J @ J.T + epsilon*np.eye(6))
       dq = pinv @ err

       q = q+alpha*dq 
    
   t2 = time.time()
   dt = t2 -t1
   print("time used to compute is", dt)
   return q 
       
       
       
           

    

        
