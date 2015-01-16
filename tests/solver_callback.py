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
            (self, 2, 1, "x² + y²")

    def impl_compute (self, result, x):
        result[0] = x[0]**2 + x[1]**2

    def impl_gradient (self, result, x):
        result[0] = 2. * x[0]
        result[1] = 2. * x[1]

class IterCallback (roboptim.core.PySolverCallback):
    def __init__ (self, pb):
        roboptim.core.PySolverCallback.__init__ (self, pb)
        self.iter = 0

    def callback (self, pb, state):
        parameters = state.parameters
        if self.iter > 0:
            if 'ipopt.stop' in parameters:
                stop = list(parameters['ipopt.stop'])
                stop[1] = True
                parameters['ipopt.stop'] = tuple(stop)
            state.parameters = parameters
        print("[iter %i]\n%s\n" % (self.iter, state))
        self.iter += 1

class TestSolverCallbackPy(unittest.TestCase):

    def test_callback(self):
        cost = roboptim.core.PyFiniteDifference (Square ())
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([-2., 2.])
        problem.argumentBounds = numpy.array([[float("-inf"), float("inf")],
                                              [float("-inf"), float("inf")]])

        callback = IterCallback (problem)

        # Let the test fail if the solver does not exist.
        try:
            solver = roboptim.core.PySolver ("ipopt", problem)
            solver.setIterationCallback (callback)
            print (solver)
            solver.solve ()
            r = solver.minimum ()
            print (r)
        except Exception as e:
            print ("Error: %s" % e)

if __name__ == '__main__':
    unittest.main()
