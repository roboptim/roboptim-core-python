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
        self.gradient_counter = 0
        self.jacobian_counter = 0

    def reset (self):
        self.compute_counter = 0
        self.gradient_counter = 0
        self.jacobian_counter = 0

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]
        self.compute_counter += 1

    def impl_gradient (self, result, x, f_id):
        result[0] = 2. * x[0]
        self.gradient_counter += 1

    def impl_jacobian (self, result, x):
        result[0,0] = 2. * x[0]
        self.jacobian_counter += 1

class TestFiniteDifferences(unittest.TestCase):

    def test_counters(self):

        square = Square()
        f = roboptim.core.PyCachedFunction (square, 10)

        x = numpy.array ([4.])
        res = f (x)
        numpy.testing.assert_almost_equal (res, [x[0] * x[0]], 5)
        assert square.compute_counter == 1
        res2 = f (x)
        numpy.testing.assert_almost_equal (res2, [x[0] * x[0]], 5)
        assert square.compute_counter == 1

        grad = f.gradient (x, 0)
        numpy.testing.assert_almost_equal (grad, [2. * x[0]], 5)
        assert square.gradient_counter == 1
        grad2 = f.gradient (x, 0)
        numpy.testing.assert_almost_equal (grad2, [2. * x[0]], 5)
        assert square.gradient_counter == 1

        jac = f.jacobian (x)
        numpy.testing.assert_almost_equal (jac, [[2. * x[0]]], 5)
        assert square.jacobian_counter == 1
        jac2 = f.jacobian (x)
        numpy.testing.assert_almost_equal (jac2, [[2. * x[0]]], 5)
        assert square.jacobian_counter == 1


if __name__ == '__main__':
    unittest.main()
