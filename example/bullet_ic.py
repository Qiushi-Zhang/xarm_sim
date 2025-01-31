import pybullet as p 
import time 
import numpy as np
import matplotlib as mp 
import matplotlib.pyplot as plt 









p.connect(p.GUI)

# we set gravity
p.setGravity(0, 0, -9.81)
# we set the integration step to 1ms (each time we step the simulation it will advance by 1ms)
p.setPhysicsEngineParameter(fixedTimeStep=1.0/1000.0, numSubSteps=1)
# Disable the gui controller as we don't use them.
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

p.resetDebugVisualizerCamera(2, 50, -35, (0., 0., 0.))

# we set the initial position and orientation of the robot
robotStartPosition = [0.,0,.0]
robotStartOrientation = p.getQuaternionFromEuler([0,0,0])

# we load the robot - the robot should be attached to the ground
# so we set useFixedBase to True
robotId = p.loadURDF('/home/qiushi/workspace/mim_robots/python/mim_robots/robots/kuka/urdf/iiwa.urdf', robotStartPosition,
                robotStartOrientation, flags=p.URDF_USE_INERTIA_FROM_FILE,
                useFixedBase=True)

# you should now see our NYU finger in the pybullet window
# you should however not be able to do much because the simulation is not running yet
nj = p.getNumJoints(robotId)
print('the robot has: ' + str(nj) + ' joints\n')
print('the joint names are:')
for i in range(nj):
    print(p.getJointInfo(robotId, i)[1].decode('UTF-8'))


joint_names = [
            'finger_base_to_upper_joint',
            'finger_upper_to_middle_joint',
            'finger_middle_to_lower_joint',
        ]

# a map from names to ids
bullet_joint_map = {}
for ji in range(p.getNumJoints(robotId)):
    bullet_joint_map[p.getJointInfo(robotId, ji)[1].decode('UTF-8')] = ji

# a list of ids we are interested in
bullet_joint_ids = np.array([bullet_joint_map[name] for name in joint_names])
num_joints = bullet_joint_ids.size


p.setJointMotorControlArray(robotId, self.bullet_joint_ids,
                                    p.VELOCITY_CONTROL, forces=np.zeros(self.nj))