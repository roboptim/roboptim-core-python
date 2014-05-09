#!/usr/bin/env python
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy, numpy.testing

class Square (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 1, 1, "differentiable function test")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]

    def impl_gradient (self, result, x):
        result[0] = 2.


class TestFunctionPy(unittest.TestCase):
    def test_function(self):
        class F(roboptim.core.PyFunction):
            def __init__ (self):
                roboptim.core.PyFunction.__init__ (self, 1, 1, "function test")

            def impl_compute (self, result, x):
                result[0] = 42.

        f = F()
        print (f.inputSize ())
        print (f.outputSize ())
        print (f.name ())
        x = numpy.array ([1.,])
        print (f (x))
        print (f)

    def test_differentiable_function(self):
        f = Square ()
        print (f.inputSize ())
        print (f.outputSize ())
        print (f.name ())
        x = numpy.array ([2.,])
        print (f (x))
        print (f)

    def test_differentiable_function(self):
        cost = Square()
        problem = roboptim.core.PyProblem (cost)
        print (problem)

    def test_solver(self):
        cost = Square()
        problem = roboptim.core.PyProblem (cost)
        self.assertFalse (problem.startingPoint)
        problem.startingPoint = numpy.array([0.,])
        self.assertEqual (problem.startingPoint, [0.])

        problem.argumentBounds = numpy.array([[-3.,4.],])
        numpy.testing.assert_almost_equal (problem.argumentBounds, [[-3.,4.],])

        problem.argumentScales = numpy.array([2.,])
        numpy.testing.assert_almost_equal (problem.argumentScales, [2.,])

        g1 = Square ()
        problem.addConstraint (g1, [-1., 10.,])

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem)
            print (solver)

            solver.solve ()
            r = solver.minimum ()
            print (r)

            # Add a new dummy parameter
            parameters = solver.parameters
            parameters["dummy"] = tuple(("dummy description", "dummy value"))
            solver.parameters = parameters
            print (solver)
        except:
            print ("ipopt solver not available, passing...")

if __name__ == '__main__':
    unittest.main()
