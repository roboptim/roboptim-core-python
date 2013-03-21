from __future__ import \
    print_function, unicode_literals, absolute_import, division

import abc
import inspect
import numpy

from .wrap import *

class PyFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__ (self, inSize, outSize, name):
        self._function = Function (inSize, outSize, name)
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))

    def inputSize (self):
        return inputSize (self._function)
    def outputSize (self):
        return outputSize (self._function)
    def name (self):
        return getName (self._function)

    @abc.abstractmethod
    def impl_compute (self, result, x):
        return

    def __call__(self, x):
        result = numpy.array ([0.,])
        compute (self._function, result, x)
        return result

    def __str__ (self):
        return strFunction (self._function)

    @classmethod
    def __subclasshook__ (cls, C):
        if cls is PyFunction:
            # FIXME: check that __call__ has the correct number
            # of arguments.
            if any("impl_compute" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class PyDifferentiableFunction(PyFunction):
    __metaclass__ = abc.ABCMeta

    def __init__ (self, inSize, outSize, name):
        self._function = DifferentiableFunction (inSize, outSize, name)
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))
        gradientCb = lambda result, x, fid: self.impl_gradient (result, x, fid)
        bindGradient (self._function, gradientCb)

    @abc.abstractmethod
    def impl_gradient (self, result, x):
        return

    def gradient (self, x, functionId):
        g = numpy.array ([0.,])
        gradient (self._function, g, x, functionId)
        return g

    @classmethod
    def __subclasshook__ (cls, C):
        if cls is PyDifferentiableFunction:
            # FIXME: check that __call__ has the correct number
            # of arguments.
            if any("impl_gradient" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented

class PyProblem(object):
    def __init__(self, cost):
        self.cost = cost
        self._problem = Problem (cost._function)

    def __str__ (self):
        return strProblem (self._problem)

    def addConstraint (self):
        pass #FIXME:

class PySolver(object):
    def __init__(self, solverName, problem):
        self._solver = Solver (solverName, problem._problem)

    def __str__ (self):
        return strSolver (self._solver)

    def solve (self):
        solve (self._solver)
        return None #FIXME:
