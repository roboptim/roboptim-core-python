#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy, numpy.testing

class Square (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 1, 1, "square function")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]

    def impl_gradient (self, result, x, f_id):
        result[0] = 2. * x[0]


class SquareJacobian (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 1, 1, "square function")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]

    def impl_gradient (self, result, x, f_id):
        raise NotImplementedError

    def impl_jacobian (self, result, x):
        result[0,0] = 2. * x[0]


class DoubleSquare (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 1, 2, "double square function")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]
        result[1] = x[0] * x[0]

    def impl_gradient (self, result, x, f_id):
        result[0] = 2. * x[0]


class TestFunctionPy(unittest.TestCase):
    def test_function(self):
        class F(roboptim.core.PyFunction):
            def __init__ (self):
                roboptim.core.PyFunction.__init__ (self, 1, 1, "dummy function")

            def impl_compute (self, result, x):
                result[0] = 42.

        f = F()
        print (f.inputSize ())
        print (f.outputSize ())
        print (f.name ())
        x = numpy.array ([1.,])
        print ("f: %s" % f)
        print ("x = %s" % x)
        print ("f(x) = %s" % f (x))
        self.assertEqual (f (x), [42.])
        self.assertEqual ("dummy function (not differentiable)", "%s" % f)

    def test_differentiable_function(self):
        f = Square ()
        print (f.inputSize ())
        print (f.outputSize ())
        print (f.name ())
        x = numpy.array ([6.,])
        print ("f: %s" % f)
        print ("x = %s" % x)
        print ("f(x) = %s" % f (x))
        self.assertEqual (f (x), x[0] * x[0])
        print ("df(x) = %s" % f.gradient (x, 0))
        self.assertEqual (f.gradient (x, 0), 2. * x[0])
        self.assertEqual ("square function (differentiable function)", "%s" % f)

    def test_differentiable_function_jacobian(self):
        f = SquareJacobian ()
        print (f.inputSize ())
        print (f.outputSize ())
        print (f.name ())
        x = numpy.array ([6.,])
        print ("f: %s" % f)
        print ("x = %s" % x)
        print ("f(x) = %s" % f (x))
        self.assertEqual (f (x), x[0] * x[0])
        self.assertRaises(NotImplementedError, lambda: f.gradient (x, 0))
        print ("Jac(f)(x) = %s" % f.jacobian (x))
        self.assertEqual (f.jacobian (x), 2. * x[0])
        self.assertEqual ("square function (differentiable function)", "%s" % f)

    def test_problem(self):
        cost = Square()
        self.assertEqual ("square function (differentiable function)", "%s" % cost)
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

        g2 = DoubleSquare ()
        problem.addConstraint (g2, numpy.array ([[-1., 10.],[2., 3.]]))

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem)
            print (solver)

            solver.solve ()
            r = solver.minimum ()
            print (r)

            # Add a new dummy parameter
            parameters = dict()
            parameters["dummy"] = tuple(("dummy description",
                                         "dummy value"))
            solver.parameters = parameters

            print (solver)

            test_parameters = list()
            test_parameters.append (("foo_int", 42, "an integer"))
            test_parameters.append (("foo_double", 12., "a scalar"))
            test_parameters.append (("foo_str", "foo", "a string"))

            for p in test_parameters:
                solver.setParameter (p[0], p[1], p[2])

            parameters = solver.parameters
            print(parameters)
            assert "dummy" in parameters \
                    and parameters["dummy"][0] == "dummy description" \
                    and parameters["dummy"][1] == "dummy value"

            for p in test_parameters:
                assert p[0] in parameters \
                        and parameters[p[0]][0] == p[2] \
                        and parameters[p[0]][1] == p[1]

            print (solver)
        except Exception as e:
            print ("Error: %s" % e)

if __name__ == '__main__':
    unittest.main()
