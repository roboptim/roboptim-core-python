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


class Problem8_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "-1")

    def impl_compute (self, result, x):
        result[0] = -1.

    def impl_gradient (self, result, x, functionId):
        result[0] = 0.
        result[1] = 0.


class Problem8_G1 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "x₀² + x₁² - 25")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0] + x[1] * x[1] - 25.

    def impl_gradient (self, result, x, functionId):
        result[0] = 2. * x[0]
        result[1] = 2. * x[1]


class Problem8_G2 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "x₀ x₁ - 9")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[1] - 9.

    def impl_gradient (self, result, x, functionId):
        result[0] = x[1]
        result[1] = x[0]


class Problem48_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 5, 1, "(x₀ - 1)² + (x₁ - x₂)² + (x₃ - x₄)²")

    def impl_compute (self, result, x):
        result[0] = (x[0] - 1.)**2 + (x[1] - x[2])**2 + (x[3] - x[4])**2

    def impl_gradient (self, result, x, functionId):
        result[0] = 2. * ( x[0] - 1.);
        result[1] = 2. * ( x[1] - x[2]);
        result[2] = 2. * (-x[1] + x[2]);
        result[3] = 2. * ( x[3] - x[4]);
        result[4] = 2. * (-x[3] + x[4]);


class Problem48_G1 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 5, 2, "x₀ + x₁ + x₂ + x₃ + x₄, x₂ - 2(x₃ + x₄)")

    def impl_compute (self, result, x):
        result[0] = x[0] + x[1] + x[2] + x[3] + x[4];
        result[1] = x[2] - 2. * (x[3] + x[4]);

    def impl_gradient (self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian (self, result, x):
        result[0,0] = 1.;
        result[0,1] = 1.;
        result[0,2] = 1.;
        result[0,3] = 1.;
        result[0,4] = 1.;

        result[1,2] =  1.;
        result[1,3] = -2.;
        result[1,4] = -2.;

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

        solver = roboptim.core.PySolver ("ipopt", problem)
        solver.setParameter("ipopt.output_file", "problem_1.log")
        print (solver)
        solver.solve ()
        r = solver.minimum ()
        print (r)
        numpy.testing.assert_almost_equal (r.value, [0.])
        numpy.testing.assert_almost_equal (r.x, [1., 1.])

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

        solver = roboptim.core.PySolver ("ipopt", problem)
        solver.setParameter("ipopt.output_file", "problem_2.log")
        print (solver)
        solver.solve ()
        r = solver.minimum ()
        print (r)
        a = math.pow (598./1200., 0.5)
        b = 400 * a**3
        final_x = [2.*a*math.cos (1./3. * math.acos (1./b)), 1.5]
        numpy.testing.assert_almost_equal (r.value, [0.0504261879])
        numpy.testing.assert_almost_equal (r.x, final_x)

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

        solver = roboptim.core.PySolver ("ipopt", problem)
        solver.setParameter("ipopt.output_file", "problem_6.log")
        print (solver)
        solver.solve ()
        r = solver.minimum ()
        print (r)
        numpy.testing.assert_almost_equal (r.value, [0.])
        numpy.testing.assert_almost_equal (r.x, [1., 1.])

    def test_problem_8(self):
        """
        Schittkowski problem #8
        """
        cost = Problem8_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([2., 1., ])
        problem.argumentBounds = numpy.array([[float("-inf"), float("inf")],
                                              [float("-inf"), float("inf")], ])

        g1 = Problem8_G1 ()
        problem.addConstraint (g1, [0., 0.,])
        g2 = Problem8_G2 ()
        problem.addConstraint (g2, [0., 0.,])

        # Check starting value
        numpy.testing.assert_almost_equal (cost (problem.startingPoint)[0], -1.)

        solver = roboptim.core.PySolver ("ipopt", problem)
        solver.setParameter("ipopt.output_file", "problem_8.log")
        print (solver)
        solver.solve ()
        r = solver.minimum ()
        print (r)
        a = math.sqrt ((25. + math.sqrt (301.)) / 2.)
        numpy.testing.assert_almost_equal (r.value, [-1.])
        numpy.testing.assert_almost_equal (numpy.sort(r.x), numpy.sort([a, 9./a]))

    def test_problem_48(self):
        """
        Schittkowski problem #48
        """
        cost = Problem48_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([3., 5., -3., 2., -2.])

        g1 = Problem48_G1 ()
        problem.addConstraint (g1, numpy.array ([[5., 5.],[-3., -3.]]))

        # Check starting value
        numpy.testing.assert_almost_equal (cost (problem.startingPoint)[0], 84.)

        solver = roboptim.core.PySolver ("ipopt", problem)
        solver.setParameter("ipopt.output_file", "problem_48.log")
        print (solver)
        solver.solve ()
        r = solver.minimum ()
        print (r)
        numpy.testing.assert_almost_equal (r.value, [0.])
        numpy.testing.assert_almost_equal (r.x, [1., 1., 1., 1., 1.])

if __name__ == '__main__':
    unittest.main ()
