#!/usr/bin/env python
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy

class Square (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 1, 1, "test")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[0]

    def impl_gradient (self, result, x):
        result[0] = 2.


class TestFunctionPy(unittest.TestCase):
    def test_function(self):
        class F(roboptim.core.PyFunction):
            def __init__ (self):
                roboptim.core.PyFunction.__init__ (self, 1, 1, "test")

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
        solver = roboptim.core.PySolver ("ipopt", problem)
        print (solver)
        solver.solve ()


if __name__ == '__main__':
    unittest.main()
