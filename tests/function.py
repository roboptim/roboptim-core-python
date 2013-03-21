#!/usr/bin/env python
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

        # Check computation with sequences.
        x = [42.,]
        result = numpy.array([0.,])

        roboptim.core.compute (f, x, result)

        self.assertEqual (x, [42.,])
        self.assertEqual (result, [84.,])

        # Check computation with tuples.
        x = (0.,)
        roboptim.core.compute (f, x, result)
        self.assertEqual (x, (0.,))
        self.assertEqual (result, [0.,])

        # Check computation with arrays.
        x = numpy.array([-10.,])
        roboptim.core.compute (f, x, result)
        self.assertEqual (x[0], -10.)
        self.assertEqual (result, [-20.,])

        # Check with differentiable function
        f = roboptim.core.DifferentiableFunction (1, 1, "test function")
        roboptim.core.bindCompute(f, compute)

        x = [15.,]
        result = numpy.array([0.,])

        roboptim.core.compute (f, x, result)

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
        self.assertRaises(TypeError, roboptim.core.compute, (f, x, result))

        # Check that error are throw properly when callback has a bad
        # prototype.
        roboptim.core.bindCompute(f, badcallback)

        x = [42.,]
        result = numpy.array([0.,])
        self.assertRaises(TypeError, roboptim.core.compute, (f, x, result))

        roboptim.core.bindCompute(f, badcallback2)
        self.assertRaises(TypeError, roboptim.core.compute, (f, x, result))

    def test_gradient(self):
        def compute(result, x):
            result[0] = x[0] * x[0]
        def gradient(result, x):
            result[0] = 2

        f = roboptim.core.DifferentiableFunction (1, 1, "x * x")
        roboptim.core.bindCompute(f, compute)
        #roboptim.core.bindGradient(f, gradient)


if __name__ == '__main__':
    unittest.main()
