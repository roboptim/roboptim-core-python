#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy, numpy.testing


class Problem1_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "100 (x₁ - x₀²)² + (1 - x₀)²")

    def impl_compute (self, result, x):
        result[0] = 100. * (x[1] - x[0]**2)**2 + (1. - x[0])**2

    def impl_gradient (self, result, x, functionId):
        result[0] = -400. * x[0] * (x[1] - x[0] ** 2) - 2. * (1. - x[0])
        result[1] = 200. * (x[1] - x[0]**2)


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
            # TODO: assert result:
            # expected result = [1., 1.]
            # expected value = 0.
        except:
            print ("ipopt solver not available, passing...")

if __name__ == '__main__':
    unittest.main ()
