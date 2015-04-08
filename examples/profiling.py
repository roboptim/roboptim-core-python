#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import os
import roboptim.core
import numpy, numpy.testing
import math
import argparse


class Engine (roboptim.core.PyDifferentiableFunction):
    def __init__ (self, n):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2*n, 1, "Dummy engine")
        self.n = n
        self.res = 0
        self.jac = numpy.zeros ((1, 2*n))

    def impl_compute (self, result, x):
        self.computeData(x)

    def impl_gradient (self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian (self, result, x):
        self.computeJacobian (x)

    def jacobian (self, x):
        self.computeJacobian (x)

    def computeData (self, x):
        self.res = x

    def computeJacobian (self, x):
        self.jac.fill (1.)


class Problem_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self, n):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2*n, 1, "100 (xi₁ - xi₀²)² + (1 - xi₀)²")
        self.n = n

    def impl_compute (self, result, x):
        result[0] = sum([100. * (xi[1] - xi[0]**2)**2 + (1. - xi[0])**2
                        for xi in x.reshape((self.n, 2))])

    def impl_gradient (self, result, x, functionId):
        for i in range(self.n):
            x0 = x[2*i]
            x1 = x[2*i+1]
            result[2*i]   = -400. * x0 * (x1 - x0 ** 2) - 2. * (1. - x0)
            result[2*i+1] = 200. * (x1 - x0**2)

class Problem_Constraint (roboptim.core.PyDifferentiableFunction):
    def __init__ (self, engine):
        self.n = engine.n
        self.engine = engine
        # Note: this can be handled directly as argument bounds,
        # but we add this as constraint to increase the size of
        # the Jacobian to help track possible leaks.
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2*n, 2*n, "xi₀ >= 0, xi₁ >= 0")

    def impl_compute (self, result, x):
        result = self.engine.res

    def impl_gradient (self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian (self, result, x):
        numpy.copyto(result, self.engine.jac)


"""
Schittkowski problem #1 duplicated n times.
"""
parser = argparse.ArgumentParser(description='Run a benchmark scaling with n.')
parser.add_argument('n', metavar='n', type=int,
                    help='Number of variables/constraints = 2n')
args = parser.parse_args()
n = args.n
cost = Problem_Cost (n)

starting_point = numpy.array([-2., 1.] * n).flatten()
bounds = numpy.array([[float("-inf"), float("inf")],
                      [-1.5, float("inf")]] * n)

problem = roboptim.core.PyProblem (cost)
problem.startingPoint = starting_point
problem.argumentBounds = bounds

engine = Engine (n)
cstr = Problem_Constraint (engine)
functions = [cstr]
pool = roboptim.core.PyFunctionPool (engine, functions, "Dummy FD pool")
fd_rule = roboptim.core.FiniteDifferenceRule.SIMPLE
fd_pool = roboptim.core.PyFiniteDifference (pool, rule = fd_rule)

problem.addConstraint (fd_pool, numpy.tile ([[0, float("inf")]], (2*n,1)))

# Check starting value
numpy.testing.assert_almost_equal (cost (problem.startingPoint), 909 * n)

solver = roboptim.core.PySolver ("ipopt", problem)
solver.setParameter("ipopt.mu_strategy", "adaptive")
solver.setParameter("ipopt.tol", 1e-6)
solver.setParameter("ipopt.output_file", "ipopt.log")
solver.setParameter("ipopt.print_user_options", "yes")

print (solver)
solver.solve ()
r = solver.minimum ()
print (r)

numpy.testing.assert_almost_equal (r.value, [0.], 5)
numpy.testing.assert_almost_equal (r.x, numpy.array([1., 1.] * n).flatten(), 5)
