#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import unittest
import roboptim.core
import numpy as np


class Engine (roboptim.core.PyDifferentiableFunction):
    def __init__ (self, n):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2*n, n, "Compute xᵢ² + yᵢ² for each function")
        self.n = n
        self.data = np.zeros (n)
        self.jac = np.zeros ((n, 2*n))

    def impl_compute (self, result, x):
        print("Engine: impl_compute")
        self.computeData(x)

    def impl_gradient (self, result, x):
        return NotImplementedError

    def impl_jacobian (self, result, x):
        print("Engine: impl_jacobian")
        self.computeJacobian (x)

    def jacobian (self, x):
        print("Engine: jacobian")
        self.computeJacobian (x)

    def computeData (self, x):
        print("Engine: computing data")
        for i in range(self.n):
            self.data[i] = x[2*i]**2 + x[2*i+1]**2

    def computeJacobian (self, x):
        print("Engine: computing Jacobian")
        self.jac.fill (0.)
        # For each square function
        for i in range(self.n):
            # For the 2 variables influencing the current square function
            for j in range(2*i,2*(i+1)):
                self.jac[i,j] = 2*x[j]

    def getData (self, idx):
        return self.data[idx]

    def getJac (self, idx, var_idx):
        return self.jac[idx, var_idx]

class Square (roboptim.core.PyDifferentiableFunction):
    def __init__ (self, engine, idx):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, engine.inputSize (), 1, "x² + y²")
        self.engine = engine
        self.idx = idx

    def impl_compute (self, result, x):
        print("Function %i: copying data" % self.idx)
        result[0] = self.engine.getData (self.idx)

    def impl_gradient (self, result, x, functionId):
        print("Function %i: copying gradient" % self.idx)
        for i in range(2):
            result[2*self.idx + i] = self.engine.getJac (self.idx, 2*self.idx + i)

class TestFunctionPoolPy(unittest.TestCase):

    def test_engine(self):
        engine = Engine (3)
        x = np.array([10., -5., 1., 2., -1., 1.])

        np.testing.assert_almost_equal (engine.data, np.zeros (engine.n))
        engine (x)
        np.testing.assert_almost_equal (engine.data,
                [xi**2 + yi**2 for xi,yi in x.reshape(engine.n, 2) ])

        np.testing.assert_almost_equal (engine.jac, np.zeros ((engine.n, 2*engine.n)))
        engine.jacobian (x)
        jac = np.zeros ((engine.n, 2*engine.n))
        for i in range(engine.n):
            for j in range(2):
                jac[i,2*i+j] = 2. * x[2*i+j]
        np.testing.assert_almost_equal (engine.jac, jac)

    def test_pool(self):
        engine = Engine (3)
        np.testing.assert_almost_equal (engine.data, np.zeros (engine.n))
        functions = [Square (engine, 0.), Square (engine, 1.), Square (engine, 2.)]
        print(engine)
        pool = roboptim.core.PyFunctionPool (engine, functions, "Dummy pool")
        print(pool)
        x = np.array([10., -5., 1., 2., -1., 1.])
        res = pool (x)
        np.testing.assert_almost_equal (engine.data,
                [xi**2 + yi**2 for xi,yi in x.reshape(engine.n, 2) ])
        pool.jacobian (x)
        jac = np.zeros ((engine.n, 2*engine.n))
        for i in range(engine.n):
            for j in range(2):
                jac[i,2*i+j] = 2. * x[2*i+j]
        np.testing.assert_almost_equal (engine.jac, jac)

if __name__ == '__main__':
    unittest.main()
