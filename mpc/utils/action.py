

import numpy as np
import crocoddyl
import mujoco
import pinocchio 
import os 



class DAM_fwd_example(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, state.nv, costModel.nr
        )
        self.costs = costModel
        self.enable_force = True
        self.armature = np.zeros(0)

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[: self.state.nq], x[-self.state.nv :]
        # Computing the dynamics using ABA or manually for armature case
        if self.enable_force:
            data.xout = pinocchio.aba(self.state.pinocchio, data.pinocchio, q, v, u)
        else:
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            data.M = data.pinocchio.M
            if self.armature.size == self.state.nv:
                data.M[range(self.state.nv), range(self.state.nv)] += self.armature
            data.Minv = np.linalg.inv(data.M)
            data.xout = data.Minv * (u - data.pinocchio.nle)
        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        if u is None:
            u = self.unone
        if True:
            self.calc(data, x, u)
        # Computing the dynamics derivatives
        if self.enable_force:
            pinocchio.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, u
            )
            data.Fx = np.hstack([data.pinocchio.ddq_dq, data.pinocchio.ddq_dv])
            data.Fu = data.pinocchio.Minv
        else:
            pinocchio.computeRNEADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, data.xout
            )
            data.Fx = -np.hstack(
                [data.Minv * data.pinocchio.dtau_dq, data.Minv * data.pinocchio.dtau_dv]
            )
            data.Fu = data.Minv
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print("The armature dimension is wrong, we cannot set it.")
        else:
            self.enable_force = False
            self.armature = armature.T

    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pinocchio.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        )  # this allows us to share the memory of cost-terms of action model
        return data

class DAM_fwd_exampleT(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, state.nv, costModel.nr
        )
        self.costs = costModel
        self.enable_force = True
        self.armature = np.zeros(0)
        # self.nu = 7
        # print(self.nu, "self.nu")
     
    def calc(self, data, x, u=None):
        # print("implementing calc T")
        if u is None:
            # print("u is none")
            u = np.zeros(self.nu)
        q, v = x[: self.state.nq], x[-self.state.nv :]
        # Computing the dynamics using ABA or manually for armature case
        if self.enable_force:
            data.xout = pinocchio.aba(self.state.pinocchio, data.pinocchio, q, v, u)
        else:
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            data.M = data.pinocchio.M
            if self.armature.size == self.state.nv:
                data.M[range(self.state.nv), range(self.state.nv)] += self.armature
            data.Minv = np.linalg.inv(data.M)
            data.xout = data.Minv * (u - data.pinocchio.nle)
        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        # print("implementing calcdiff")

        q, v = x[: self.state.nq], x[-self.state.nv :]
        if u is None:
            u = np.zeros(self.nu)
        if True:
            self.calc(data, x, u)
        # Computing the dynamics derivatives
        if self.enable_force:
            pinocchio.computeABADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, u
            )
            data.Fx = np.hstack([data.pinocchio.ddq_dq, data.pinocchio.ddq_dv])
            data.Fu = data.pinocchio.Minv
        else:
            pinocchio.computeRNEADerivatives(
                self.state.pinocchio, data.pinocchio, q, v, data.xout
            )
            data.Fx = -np.hstack(
                [data.Minv * data.pinocchio.dtau_dq, data.Minv * data.pinocchio.dtau_dv]
            )
            data.Fu = data.Minv
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print("The armature dimension is wrong, we cannot set it.")
        else:
            self.enable_force = False
            self.armature = armature.T

    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pinocchio.Data(self.state.pinocchio)
        data.multibody = crocoddyl.DataCollectorMultibody(data.pinocchio)
        data.costs = self.costs.createData(data.multibody)
        data.costs.shareMemory(
            data
        )  # this allows us to share the memory of cost-terms of action model
        return data
    