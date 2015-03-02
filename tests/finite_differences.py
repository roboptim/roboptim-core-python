#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy, numpy.testing
import math

class Square (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 1, 1, "square function")
        self.compute_counter = 0

    def reset (self):
        self.compute_counter = 0

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]
        self.compute_counter += 1

    def impl_gradient (self, result, x, f_id):
        raise NotImplementedError

class Problem1_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "100 (x₁ - x₀²)² + (1 - x₀)²")

    def impl_compute (self, result, x):
        result[0] = 100. * (x[1] - x[0]**2)**2 + (1. - x[0])**2

    def impl_gradient (self, result, x, functionId):
        raise NotImplementedError


class TestFiniteDifferences(unittest.TestCase):

    def test_jacobian_simple(self):

        fd_rule = roboptim.core.FiniteDifferenceRule.SIMPLE
        square = Square()
        f = roboptim.core.PyFiniteDifference (square, rule = fd_rule)

        x = numpy.array ([4.])
        res = f (x)
        numpy.testing.assert_almost_equal (res, [x[0] * x[0]], 5)
        assert square.compute_counter == 1
        square.reset ()

        grad = f.gradient (x, 0)
        numpy.testing.assert_almost_equal (grad, [2. * x[0]], 5)
        assert square.compute_counter == 2
        square.reset ()

        jac = f.jacobian (x)
        numpy.testing.assert_almost_equal (jac, [[2. * x[0]]], 5)
        assert square.compute_counter == 2

    def test_jacobian_fivepoints(self):

        fd_rule = roboptim.core.FiniteDifferenceRule.FIVE_POINTS
        square = Square()
        f = roboptim.core.PyFiniteDifference (square, rule = fd_rule)

        x = numpy.array ([4.])
        res = f (x)
        numpy.testing.assert_almost_equal (res, [x[0] * x[0]], 5)
        assert square.compute_counter == 1
        square.reset ()

        grad = f.gradient (x, 0)
        numpy.testing.assert_almost_equal (grad, [2. * x[0]], 5)
        assert square.compute_counter == 4
        square.reset ()

        jac = f.jacobian (x)
        numpy.testing.assert_almost_equal (jac, [[2. * x[0]]], 5)
        assert square.compute_counter == 4

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

        solver = roboptim.core.PySolver ("ipopt", problem)
        print (solver)
        solver.solve ()
        r = solver.minimum ()
        print (r)
        numpy.testing.assert_almost_equal (r.value, [0.], 5)
        numpy.testing.assert_almost_equal (r.x, [1., 1.], 5)

if __name__ == '__main__':
    unittest.main()
