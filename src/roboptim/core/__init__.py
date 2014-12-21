from __future__ import \
    print_function, unicode_literals, absolute_import, division

import abc
import inspect
import numpy

from .wrap import *


class PyFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__ (self, inSize, outSize, name):
        self._function = Function (inSize, outSize,
                                   self._formatName(name))
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))

    def inputSize (self):
        return inputSize (self._function)

    def outputSize (self):
        return outputSize (self._function)

    def name (self):
        return getName (self._function)

    def _formatName(self,name):
        """
        This method is used to accept UTF-8 function names for both Python 2
        and Python 3.
        """
        if not isinstance (name, (str)):
            return name.encode ('utf-8')
        else:
            return name

    def _decodeName(self, name):
        """
        This method is used to decode UTF-8 function names for both Python 2
        and Python 3.
        """
        if not isinstance (name, (str)):
            return name.decode ('utf-8')
        else:
            return name

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
        self._function = DifferentiableFunction (inSize, outSize,
                                                 self._formatName(name))
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))
        gradientCb = lambda result, x, fid: self.impl_gradient (result, x, fid)
        bindGradient (self._function, gradientCb)

    @abc.abstractmethod
    def impl_gradient (self, result, x, functionId):
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


class FiniteDifferenceRule:
    SIMPLE = 1
    FIVE_POINTS = 2


class PyFiniteDifference(PyDifferentiableFunction):
    def __init__ (self, f, epsilon = 1e-8, rule = FiniteDifferenceRule.SIMPLE):
        PyDifferentiableFunction.__init__ \
            (self, f.inputSize (), f.outputSize (), \
             self._decodeName (f.name ()))
        if rule == FiniteDifferenceRule.SIMPLE:
            self._fd = SimpleFiniteDifferenceGradient (f._function, epsilon)
        elif rule == FiniteDifferenceRule.FIVE_POINTS:
            self._fd = FivePointsFiniteDifferenceGradient (f._function, epsilon)
        else:
            raise ValueError("Unknown finite-difference rule.")


    def impl_compute (self, result, x):
        compute (self._fd, result, x)

    def impl_gradient (self, result, x, functionId):
        gradient (self._fd, result, x, functionId)


class PyProblem(object):
    def __init__(self, cost):
        self.cost = cost
        self._problem = Problem (cost._function)

    def __str__ (self):
        return strProblem (self._problem)

    @property
    def startingPoint(self):
        return getStartingPoint (self._problem)

    @startingPoint.setter
    def startingPoint(self, value):
        setStartingPoint (self._problem, value)

    @property
    def argumentBounds(self):
        return getArgumentBounds (self._problem)

    @argumentBounds.setter
    def argumentBounds(self, value):
        setArgumentBounds (self._problem, value)

    @property
    def argumentScales(self):
        return getArgumentScales (self._problem)

    @argumentScales.setter
    def argumentScales(self, value):
        setArgumentScales (self._problem, value)

    def addConstraint (self, constraint, bounds):
        addConstraint (self._problem, constraint._function, bounds)


class PySolver(object):
    def __init__(self, solverName, problem):
        self._solver = Solver (solverName, problem._problem)

    def __str__ (self):
        return strSolver (self._solver)

    def solve (self):
        solve (self._solver)

    def minimum (self):
        (objType, obj) = minimum (self._solver)
        if objType == "roboptim_core_result":
            return PyResult (obj)
        elif objType == "roboptim_core_result_with_warnings":
            return PyResultWithWarnings (obj)
        elif objType == "roboptim_core_solver_error":
            return PySolverError (obj)
        else:
            raise TypeError ("unhandled case")

    @property
    def parameters(self):
        return getParameters (self._solver)

    @parameters.setter
    def parameters(self, value):
        setParameters (self._solver, value)

class PyResult(object):
    def __init__(self, _result):
        self._result = _result
        self._dict = resultToDict (_result)

    def __str__ (self):
        return strResult (self._result)

    @property
    def inputSize(self):
        return int(self._dict["inputSize"])

    @property
    def outputSize(self):
        return int(self._dict["outputSize"])

    @property
    def x(self):
        return self._dict["x"]

    @property
    def value(self):
        return self._dict["value"]

    @property
    def constraints(self):
        return self._dict["constraints"]

    @property
    def lagrange(self):
        return self._dict["lambda"]


class PyResultWithWarnings(PyResult):
    def __init__(self, _result):
        self._result = _result
        self._dict = resultWithWarningsToDict (_result)

    def __str__ (self):
        return strResultWithWarnings (self._result)

    @property
    def warnings(self):
        return self._dict["warnings"]


class PySolverError(object):
    def __init__(self, _error):
        self._error = _error
        self._dict = solverErrorToDict (_error)

    def __str__ (self):
        return strSolverError (self._error)

    @property
    def error(self):
        return self._dict["error"]

    @property
    def lastState(self):
        if self._dict["lastState"]:
            return self._dict["lastState"]
