import mujoco 
import sys
import os
import crocoddyl
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pinocchio as pin 
import numpy as np 
import mujoco_viewer
from utils import pin_utils
from utils import visualizer
from mpc.utils.action import DAM_fwd_example, DAM_fwd_exampleT


class solver_ddp_reach():
    def __init__ (self, urdf_path, T,DT):
        self.robot_model = pin.buildModelFromUrdf(urdf_path)
        self.state = crocoddyl.StateMultibody(self.robot_model)
        self.T = T
        self.DT = DT

    def generate_ddp(self, x0, x_ref):
        frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state,
                                            self.robot_model.getFrameId("link7"),
                                            x_ref)
        goalTrackingCost = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
        xRegCost = crocoddyl.CostModelResidual(self.state, crocoddyl.ResidualModelState(self.state))
        uRegCost = crocoddyl.CostModelResidual(self.state, crocoddyl.ResidualModelControl(self.state))

        # Create cost model per each action model

        runningCostModel = crocoddyl.CostModelSum(self.state)
        terminalCostModel = crocoddyl.CostModelSum(self.state)

        runningCostModel.addCost("gripperPose", goalTrackingCost, 1e4)
        runningCostModel.addCost("stateReg", xRegCost, 1e-1)
        runningCostModel.addCost("ctrlReg", uRegCost, 1e-5)
        terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e7)
        terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
        terminalCostModel.addCost("ctrlReg", uRegCost, 1e-5)
        runningModel = crocoddyl.IntegratedActionModelEuler(
        DAM_fwd_example(self.state,  runningCostModel), self.DT)
        terminalModel = crocoddyl.IntegratedActionModelEuler(
        DAM_fwd_exampleT(self.state, terminalCostModel), 0.)
        problem = crocoddyl.ShootingProblem(x0, [runningModel]*self.T, terminalModel)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve()
        xs = ddp.xs
        us = ddp.us
        return xs, us 
    

