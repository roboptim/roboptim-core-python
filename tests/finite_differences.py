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
        assert False


class TestFunctionPy(unittest.TestCase):

    def test_problem_1(self):
        """
        Schittkowski problem #1
        """
        cost = roboptim.core.PyFiniteDifference (Problem1_Cost (),
                rule = roboptim.core.FiniteDifferenceRule.FIVE_POINTS)
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
            numpy.testing.assert_almost_equal (r.value, [0.], 5)
            numpy.testing.assert_almost_equal (r.x, [1., 1.], 5)
        except Exception as e:
            print ("Error: %s" % e)

if __name__ == '__main__':
    unittest.main()
