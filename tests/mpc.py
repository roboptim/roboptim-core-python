#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unittest
from collections import namedtuple

import numpy as np

import roboptim.core


ProblemParams = namedtuple('ProblemParams', ['N', 'dt', 'a0', 'v0', 'p0', 'x_target'])
"""Problem parameters"""


def integrate(params, J):
    """Integrate jerks to compute acceleration, velocity and position"""
    A = np.zeros(params.N)
    V = np.zeros(params.N)
    P = np.zeros(params.N)

    dt = params.dt

    A[0] = J[0]*dt + params.a0
    V[0] = A[0]*dt + params.v0
    P[0] = V[0]*dt + params.p0

    for k in range(1, params.N):
        A[k] = J[k]*dt + A[k-1]
        V[k] = A[k]*dt + V[k-1]
        P[k] = V[k]*dt + P[k-1]

    return A, V, P


def compute_D_mat(params):
    """Compute integration matrix"""
    return np.mat(np.tril(np.ones((params.N, params.N))*params.dt))


def compute_jacs(params):
    """Compute Acceleration, velocity and position jacobian"""
    D = compute_D_mat(params)
    AJ = D
    VJ = D*AJ
    PJ = D*VJ
    return AJ, VJ, PJ


# Objective


class JerkNorm(roboptim.core.PyDifferentiableFunction):
    """Minimize jerk norm"""
    def __init__ (self, params):
        super().__init__(params.N, 1, '')
        self.params = params

    def impl_compute (self, result, x):
        result[0] = np.sum(x**2)

    def impl_gradient(self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian(self, result, x):
        result[0,:] = 2.*x


class PosTarget(roboptim.core.PyDifferentiableFunction):
    """Target a final position"""
    def __init__ (self, params):
        super().__init__(params.N, 1, '')
        self.params = params

    def impl_compute (self, result, x):
        A, V, P = integrate(self.params, x)
        result[0] = (P[-1] - self.params.x_target)**2

    def impl_gradient(self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian(self, result, x):
        A, V, P = integrate(self.params, x)
        coef = 2.*(P[-1] - self.params.x_target)
        AJ, VJ, PJ = compute_jacs(self.params)
        result[0,:] = coef*PJ[-1,:]


class Cost(roboptim.core.PyDifferentiableFunction):
    """Sum cost function"""
    def __init__ (self, params, functions):
        super().__init__(params.N, 1, 'cost')
        self.params = params
        self.functions = functions

    def impl_compute (self, result, x):
        r = 0.
        for w, f in self.functions:
            r += w*f(x)
        result[0] = r

    def impl_gradient(self, result, x, functionId):
        jac = self.jacobian(x)
        result[:] = jac[0,:]

    def impl_jacobian(self, result, x):
        J = np.mat(np.zeros((1, self.params.N)))
        for w, f in self.functions:
            J += w*f.jacobian(x)
            result[0,:] = np.array(J[0,:]).reshape(result.shape)


# Constraints


class LastVelocity(roboptim.core.PyDifferentiableFunction):
    """Last Velocity"""
    def __init__ (self, params):
        super().__init__(params.N, 1, 'LastVelocity')
        self.params = params

    def impl_compute (self, result, x):
        A, V, P = integrate(self.params, x)
        result[0] = V[-1]

    def impl_gradient(self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian(self, result, x):
        AJ, VJ, PJ = compute_jacs(self.params)
        result[0,:] = VJ[-1,:]


class Velocity(roboptim.core.PyDifferentiableFunction):
    """Velocity"""
    def __init__ (self, params):
        super().__init__(params.N, params.N, 'Velocity')
        self.params = params

    def impl_compute (self, result, x):
        A, V, P = integrate(self.params, x)
        for i in range(self.params.N):
            result[i] = V[i]

    def impl_gradient(self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian(self, result, x):
        AJ, VJ, PJ = compute_jacs(self.params)
        result[:,:] = VJ


class Position(roboptim.core.PyDifferentiableFunction):
    """Position"""
    def __init__ (self, params):
        super().__init__(params.N, params.N, 'Position')
        self.params = params

    def impl_compute (self, result, x):
        A, V, P = integrate(self.params, x)
        for i in range(self.params.N):
            result[i] = P[i]

    def impl_gradient(self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian(self, result, x):
        AJ, VJ, PJ = compute_jacs(self.params)
        result[:,:] = PJ


class TestMPCPy(unittest.TestCase):

    def test_mpc(self):
        """
        Generate speed profile by integrating jerk
        This test provide big constraint jacobian matrix
        that make him fail if numpy matrix and roboptim internal matrix stride mismatch
        """
        #                      N,  dt,  a0, v0, p0, x_target
        params = ProblemParams(30, 0.2, 0., 0., 0., 5.)
        max_jerk = 0.6

        # J vector used by finite difference test
        TJ = np.random.rand(params.N)


        # Cost
        jerk_norm = JerkNorm(params)
        jerk_norm_finite = roboptim.core.PyFiniteDifference(jerk_norm)

        pos_target = PosTarget(params)
        pos_target_finite = roboptim.core.PyFiniteDifference(pos_target)

        cost = Cost(params, [(0.1, jerk_norm), (1., pos_target)])
        cost_finite = roboptim.core.PyFiniteDifference(cost)


        # Constraints
        vel = Velocity(params)
        vel_finite = roboptim.core.PyFiniteDifference(vel)

        last_vel = LastVelocity(params)
        last_vel_finite = roboptim.core.PyFiniteDifference(last_vel)

        pos = Position(params)
        pos_finite = roboptim.core.PyFiniteDifference(pos)

        # Validate jacobian
        np.testing.assert_almost_equal(jerk_norm.jacobian(TJ), jerk_norm_finite.jacobian(TJ), 1e-4)
        np.testing.assert_almost_equal(pos_target.jacobian(TJ), pos_target_finite.jacobian(TJ), 1e-4)
        np.testing.assert_almost_equal(cost.jacobian(TJ), cost_finite.jacobian(TJ), 1e-4)
        np.testing.assert_almost_equal(vel.jacobian(TJ), vel_finite.jacobian(TJ), 1e-4)
        np.testing.assert_almost_equal(last_vel.jacobian(TJ), last_vel_finite.jacobian(TJ), 1e-4)
        np.testing.assert_almost_equal(pos.jacobian(TJ), pos_finite.jacobian(TJ), 1e-4)

        # Create problem
        problem = roboptim.core.PyProblem(cost)
        problem.startingPoint = np.array([0.]*params.N)
        problem.argumentBounds = np.array([(-max_jerk, max_jerk)]*params.N)
        problem.addConstraint(pos, np.array([(0., params.x_target)]*params.N))
        problem.addConstraint(last_vel, np.array([(3., 3.)]))
        problem.addConstraint(vel, np.array([(0., 10.)]*params.N))


        solver = roboptim.core.PySolver("ipopt", problem)
        solver.setParameter("ipopt.output_file", "mpc.log")
        solver.setParameter("ipopt.hessian_approximation", "limited-memory")
        # We can check with the ipopt derivative checker that previously checked jacobian
        # are not well forwarded to ipopt
        solver.setParameter("ipopt.derivative_test", 'first-order')

        solver.solve()
        r = solver.minimum()

        self.assertTrue(isinstance(r, roboptim.core.PyResult))

if __name__ == '__main__':
    unittest.main ()
