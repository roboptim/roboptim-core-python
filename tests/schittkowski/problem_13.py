#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy, numpy.testing
import math

class Problem13_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "(x₀ - 2)² + x₁²")

    def impl_compute (self, result, x):
        result[0] = (x[0] - 2.)**2 + x[1]**2

    def impl_gradient (self, result, x, functionId):
        result[0] = 2. * x[0] - 4.
        result[1] = 2. * x[1]


class Problem13_G1 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "(1 - x₀)³ - x₁")

    def impl_compute (self, result, x):
        result[0] = (1. - x[0])**3 - x[1]

    def impl_gradient (self, result, x, functionId):
        result[0] = -3. * (1. - x[0])**2
        result[1] = -1.

class TestFunctionPy(unittest.TestCase):

    def test_problem_13(self):
        """
        Schittkowski problem #13
        """
        cost = Problem13_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([-2, -2., ])
        problem.argumentBounds = numpy.array([[0., float("inf")],
                                              [0., float("inf")], ])

        g1 = Problem13_G1 ()
        problem.addConstraint (g1, [0., float("inf"),])

        # Check starting value
        numpy.testing.assert_almost_equal (cost (problem.startingPoint)[0], 20.)

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem, log_dir="/tmp/roboptim-core-python/problem-13")
            print (solver)
            solver.solve ()
            r = solver.minimum ()
            print (r)
            numpy.testing.assert_almost_equal (r.value, [1.])
            numpy.testing.assert_almost_equal (r.x, [1., 0.])
        except Exception as e:
            print ("Error: " + str(e))

if __name__ == '__main__':
    unittest.main ()
