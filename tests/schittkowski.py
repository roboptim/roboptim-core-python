#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy, numpy.testing
import math


class Problem1_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "100 (x₁ - x₀²)² + (1 - x₀)²")

    def impl_compute (self, result, x):
        result[0] = 100. * (x[1] - x[0]**2)**2 + (1. - x[0])**2

    def impl_gradient (self, result, x, functionId):
        result[0] = -400. * x[0] * (x[1] - x[0] ** 2) - 2. * (1. - x[0])
        result[1] = 200. * (x[1] - x[0]**2)


class Problem6_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "(1 - x₀)²")

    def impl_compute (self, result, x):
        result[0] = (1 - x[0])**2

    def impl_gradient (self, result, x, functionId):
        result[0] = -2. * (1 - x[0])
        result[1] = 0.


class Problem6_G1 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "10 (x₁ - x₀²)")

    def impl_compute (self, result, x):
        result[0] = 10. * (x[1] - x[0]**2)

    def impl_gradient (self, result, x, functionId):
        result[0] = -20. * x[0]
        result[1] = 10.


class TestFunctionPy(unittest.TestCase):

    def test_problem_1(self):
        """
        Schittkowski problem #1
        """
        cost = Problem1_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([-2., 1., ])
        problem.argumentBounds = numpy.array([[float("-inf"), float("inf")],
                                              [-1.5, float("inf")], ])

        # Check starting value
        self.assertEqual (cost (problem.startingPoint), 909)

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem)
            print (solver)
            solver.solve ()
            r = solver.minimum ()
            print (r)
            numpy.testing.assert_almost_equal (r.value, [0.])
            numpy.testing.assert_almost_equal (r.x, [1., 1.])
        except Exception as e:
            print ("Error: " + str(e))

    def test_problem_2(self):
        """
        Schittkowski problem #2
        """
        cost = Problem1_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([-2., 1., ])
        problem.argumentBounds = numpy.array([[float("-inf"), float("inf")],
                                              [1.5, float("inf")], ])

        # Check starting value
        self.assertEqual (cost (problem.startingPoint), 909)

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem)
            print (solver)
            solver.solve ()
            r = solver.minimum ()
            print (r)
            a = math.pow (598./1200., 0.5)
            b = 400 * a**3
            final_x = [2.*a*math.cos (1./3. * math.acos (1./b)), 1.5]
            numpy.testing.assert_almost_equal (r.value, [0.0504261879])
            numpy.testing.assert_almost_equal (r.x, final_x)
        except Exception as e:
            print ("Error: " + str(e))

    def test_problem_6(self):
        """
        Schittkowski problem #6
        """
        cost = Problem6_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([-1.2, 1., ])
        problem.argumentBounds = numpy.array([[float("-inf"), float("inf")],
                                              [float("-inf"), float("inf")], ])

        g1 = Problem6_G1 ()
        problem.addConstraint (g1, [0., 0.,])

        # Check starting value
        numpy.testing.assert_almost_equal (cost (problem.startingPoint)[0], 4.84)

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem)
            print (solver)
            solver.solve ()
            r = solver.minimum ()
            print (r)
            numpy.testing.assert_almost_equal (r.value, [0.])
            numpy.testing.assert_almost_equal (r.x, [1., 1.])
        except Exception as e:
            print ("Error: " + str(e))

if __name__ == '__main__':
    unittest.main ()
