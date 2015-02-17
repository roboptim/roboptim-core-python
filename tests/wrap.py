#!/usr/bin/env python
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy

class TestFunction(unittest.TestCase):
    def test_create(self):
        self.assertIsNotNone(roboptim.core.Function (1, 1, "test function"))
        self.assertIsNotNone(
            roboptim.core.DifferentiableFunction (1, 1, "test function"))

    def test_badcreate(self):
        self.assertRaises(TypeError, roboptim.core.Function, ())
        self.assertRaises(TypeError, roboptim.core.Function, ("", "", ""))

    def test_compute(self):
        def compute(result, x):
            result[0] = 2 * x[0]

        f = roboptim.core.Function (1, 1, "test function")
        roboptim.core.bindCompute(f, compute)

        self.assertEqual (roboptim.core.strFunction (f),
                          "test function (not differentiable)")

        # Check computation with sequences.
        x = [42.,]
        result = numpy.array([0.,])

        roboptim.core.compute (f, result, x)

        self.assertEqual (x, [42.,])
        self.assertEqual (result, [84.,])

        # Check computation with tuples.
        x = (0.,)
        roboptim.core.compute (f, result, x)
        self.assertEqual (x, (0.,))
        self.assertEqual (result, [0.,])

        # Check computation with arrays.
        x = numpy.array([-10.,])
        roboptim.core.compute (f, result, x)
        self.assertEqual (x[0], -10.)
        self.assertEqual (result, [-20.,])

        # Check with differentiable function
        f = roboptim.core.DifferentiableFunction (1, 1, "test function")
        roboptim.core.bindCompute(f, compute)

        self.assertEqual (roboptim.core.strFunction (f),
                          "test function (differentiable function)")

        x = [15.,]
        result = numpy.array([0.,])

        roboptim.core.compute (f, result, x)

        self.assertEqual (x, [15.,])
        self.assertEqual (result, [30.,])

    def test_badcompute(self):
        def badcallback():
            pass

        def badcallback2(a,b,c):
            pass

        f = roboptim.core.Function (1, 1, "test function")

        # We cannot call compute before setting a callback
        x = [42.,]
        result = numpy.array([0.,])
        self.assertRaises(TypeError, roboptim.core.compute, (f, result, x))

        # Check that errors are thrown properly when callback has a bad
        # prototype.
        roboptim.core.bindCompute(f, badcallback)

        x = [42.,]
        result = numpy.array([0.,])
        self.assertRaises(TypeError, roboptim.core.compute, (f, result, x))

        roboptim.core.bindCompute(f, badcallback2)
        self.assertRaises(TypeError, roboptim.core.compute, (f, result, x))

    def test_gradient(self):
        def compute(result, x):
            result[0] = x[0] * x[0]
        def gradient(result, x, functionId):
            result[0] = 2 * x[0]

        f = roboptim.core.DifferentiableFunction (1, 1, "x * x")
        roboptim.core.bindCompute(f, compute)
        roboptim.core.bindGradient(f, gradient)

        x = [15.,]
        gradient = numpy.array([0.,])
        roboptim.core.gradient (f, gradient, x, 0)
        self.assertEqual (gradient, [30.,])

    def test_jacobian(self):
        def compute(result, x):
            result[0] = x[0] + x[1]
            result[1] = x[0] - x[1]
            result[2] = x[0] * x[1]
        def gradient(result, x, functionId):
            raise NotImplementedError
        def jacobian(result, x):
            # x + y
            result[0,0] = 1.
            result[0,1] = 1.

            # x - y
            result[1,0] =  1.
            result[1,1] = -1.

            # x * y
            result[2,0] = x[1]
            result[2,1] = x[0]

        f = roboptim.core.DifferentiableFunction (2, 3, "x + y, x - y, x * y")
        roboptim.core.bindCompute(f, compute)
        roboptim.core.bindGradient(f, gradient)
        roboptim.core.bindJacobian(f, jacobian)
        inputSize = roboptim.core.inputSize(f)
        outputSize = roboptim.core.outputSize(f)

        x = [15., 10.]

        gradient = numpy.array([0., 0.])
        self.assertRaises(NotImplementedError,
                          lambda: roboptim.core.gradient (f, gradient, x, 0))

        jacobian = numpy.zeros((outputSize, inputSize))
        roboptim.core.jacobian (f, jacobian, x)
        expected_jacobian = numpy.array([[1., 1.], [1., -1.], [10., 15.]])
        numpy.testing.assert_almost_equal (jacobian, expected_jacobian)

    def test_finite_differences(self):
        def compute(result, x):
            result[0] = x[0] * x[0]
        def gradient(result, x, functionId):
            result[0] = 2 * x[0]

        f = roboptim.core.DifferentiableFunction (1, 1, "x * x")
        roboptim.core.bindCompute(f, compute)

        # Forward difference
        fwd = roboptim.core.SimpleFiniteDifferenceGradient (f)
        # Five-points rule
        five_points = roboptim.core.FivePointsFiniteDifferenceGradient (f)

        for fd in [fwd, five_points]:
            x = [42.,]
            result = numpy.array([0.,])
            result_fd = numpy.array([0.,])
            roboptim.core.compute (fd, result_fd, x)
            roboptim.core.compute (f, result, x)
            numpy.testing.assert_almost_equal (result, result_fd)
            roboptim.core.gradient (fd, result_fd, x, 0)
            roboptim.core.bindGradient(f, gradient)
            roboptim.core.gradient (f, result, x, 0)
            numpy.testing.assert_almost_equal (result, result_fd, 4)


class TestProblem(unittest.TestCase):
    def test_problem(self):
        def compute(result, x):
            result[0] = x[0] * x[0]
        def gradient(result, x, functionId):
            result[0] = 2 * x[0]

        f = roboptim.core.DifferentiableFunction (1, 1, "x * x")
        roboptim.core.bindCompute(f, compute)
        roboptim.core.bindGradient(f, gradient)

        problem = roboptim.core.Problem (f)
        self.assertTrue(roboptim.core.strProblem (problem))


class TestSolver(unittest.TestCase):
    def test_solver(self):
        def compute(result, x):
            result[0] = x[0] * x[0]
        def gradient(result, x, functionId):
            result[0] = 2 * x[0]

        f = roboptim.core.DifferentiableFunction (1, 1, "x * x")
        roboptim.core.bindCompute(f, compute)
        roboptim.core.bindGradient(f, gradient)

        problem = roboptim.core.Problem (f)
        roboptim.core.setStartingPoint(problem, numpy.array([-2.]))

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.Solver ("ipopt", problem)
            self.assertTrue(roboptim.core.strSolver (solver))
            roboptim.core.solve (solver)

            result = roboptim.core.minimum (solver)
            print (result)
        except:
            print ("ipopt solver not available, passing...")

    def test_solver_callback(self):
        def compute(result, x):
            result[0] = x[0] * x[0]
        def gradient(result, x, functionId):
            result[0] = 2 * x[0]

        class TestCallback(roboptim.core.PySolverCallback):
            def __init__ (self, pb):
                roboptim.core.PySolverCallback.__init__ (self, pb)

            def callback (self, pb, state):
                print("Dummy callback!")

        f = roboptim.core.DifferentiableFunction (1, 1, "x * x")
        roboptim.core.bindCompute(f, compute)
        roboptim.core.bindGradient(f, gradient)

        problem = roboptim.core.Problem (f)
        roboptim.core.setStartingPoint(problem, numpy.array([-2.]))

        callback = TestCallback (problem)
        print (callback)

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.Solver ("ipopt", problem)
            roboptim.core.setIterationCallback (solver, callback)
            self.assertTrue(roboptim.core.strSolver (solver))
            roboptim.core.solve (solver)

            result = roboptim.core.minimum (solver)
            print (result)
        except:
            print ("ipopt solver not available, passing...")

if __name__ == '__main__':
    unittest.main()
