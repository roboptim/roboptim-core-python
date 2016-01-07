from __future__ import \
    print_function, unicode_literals, absolute_import, division

import abc
import inspect
import os
import numpy

from concurrent.futures import ProcessPoolExecutor, as_completed

# Here, we use RTLD_GLOBAL to link with roboptim-core since the Python module
# is a plugin, itself calling RobOptim solver plugins. Without this, the Python
# plugin cannot access local symbols of roboptim-core. This is not ideal, but
# at least plugins do not need to link with roboptim-core themselves. A better
# solution may be implemented later on.
# As for the Boost dependencies, we need them as well since we use an
# OptimizationLogger (header only) which depends on extra Boost libraries.
# TODO: avoid these Boost dependencies...
from ctypes import CDLL, RTLD_GLOBAL
CDLL("libboost_date_time.so", RTLD_GLOBAL)
CDLL("libboost_system.so", RTLD_GLOBAL)
CDLL("libboost_filesystem.so", RTLD_GLOBAL)
CDLL("libroboptim-core.so", RTLD_GLOBAL)

from .wrap import *

def parallel_pool_jac_eval(data):
    f = data[0]
    x = data[1]
    i = data[2]
    return i, f.jacobian (x)

class PyFunction(object):
    __metaclass__ = abc.ABCMeta

    def __init__ (self, inSize, outSize, name):
        self._function = Function (inSize, outSize,
                                   self._formatName(name))
        self._setCallbacks()

    def _setCallbacks (self):
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))

    def inputSize (self):
        return inputSize (self._function)

    def outputSize (self):
        return outputSize (self._function)

    def name (self):
        return getName (self._function)

    def order (self):
        return getStorageOrder ()

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
        result = numpy.zeros (self.outputSize (), order=self.order())
        compute (self._function, result, x)
        return result

    def __str__ (self):
        return strFunction (self._function)

    def _getStateImpl(self, odict):
        # Remove PyCapsule object
        del odict["_function"]

    def __getstate__(self):
        odict = self.__dict__.copy()
        self._getStateImpl(odict)
        odict["inSize"] = self.inputSize ()
        odict["outSize"] = self.outputSize ()
        odict["name"] = self.name ()
        return odict

    def _setStateImpl(self, idict):
        self._function = Function (idict["inSize"], idict["outSize"],
                                   self._formatName(idict["name"]))

    def __setstate__(self, idict):
        self._setStateImpl (idict)
        del idict["inSize"]
        del idict["outSize"]
        del idict["name"]
        self.__dict__.update(idict)
        self._setCallbacks ()

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
        self._setCallbacks()

    def _setCallbacks (self):
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))
        gradientCb = lambda result, x, fid: self.impl_gradient (result, x, fid)
        bindGradient (self._function, gradientCb)

        # If the user reimplemented impl_jacobian
        if self._impl_jacobian_overriden():
            jacobianCb = lambda result, x: self.impl_jacobian (result, x)
            bindJacobian (self._function, jacobianCb)
        # Else we will rely on RobOptim's default C++ implementation

    def _impl_jacobian_overriden(self):
        return id(PyDifferentiableFunction.__dict__['impl_jacobian']) \
               != id(self.impl_jacobian.__func__)

    @abc.abstractmethod
    def impl_gradient (self, result, x, functionId):
        return

    def gradient (self, x, functionId):
        g = numpy.zeros (self.inputSize (), order=self.order())
        gradient (self._function, g, x, functionId)
        return g

    def impl_jacobian (self, result, x):
        return NotImplementedError

    def jacobian (self, x):
        jac = numpy.zeros ((self.outputSize (), self.inputSize ()), order=self.order())
        jacobian (self._function, jac, x)
        return jac

    def _setStateImpl(self, idict):
        self._function = DifferentiableFunction (idict["inSize"], idict["outSize"],
                                                 self._formatName(idict["name"]))

    @classmethod
    def __subclasshook__ (cls, C):
        if cls is PyDifferentiableFunction:
            # FIXME: check that __call__ has the correct number
            # of arguments.
            if any("impl_gradient" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented


class PyFunctionPool(PyDifferentiableFunction):
    def __init__ (self, callback, functions, name = "", n_proc = 0):
        self._callback = callback
        self._functions = functions
        inSize = callback.inputSize ()
        outSize = sum([f.outputSize() for f in functions])
        self._function = DifferentiableFunction (inSize, outSize,
                                                 self._formatName(name))
        bindCompute (self._function,
                     lambda result, x: self.impl_compute (result, x))

        bindJacobian (self._function,
                      lambda result, x: self.impl_jacobian (result, x))

        # Can be used as a row index for parallel Jacobian filling process
        self._ranges = list()
        pos = 0
        for f in self._functions:
            size = f.outputSize ()
            self._ranges.append((pos, pos + size))
            pos += size

        # Multiprocessing support
        self._n_proc = n_proc

    def impl_compute (self, result, x):
        # Run callback
        self._callback (x)

        # Fill result
        for i,f in enumerate(self._functions):
            start = self._ranges[i][0]
            end = self._ranges[i][1]
            result[start:end] = f(x)

    def impl_gradient (self, result, x, functionId):
        raise NotImplementedError

    def impl_jacobian (self, result, x):
        # FIXME: find why this fails (callback not called)
        #        In the meantime, we implement this in Python
        #jacobian (self._function, result, x)

        # Run callback
        self._callback.jacobian (x)

        def parallel_pool_jac_fill(future):
            idx, value = future.result()
            result[self._ranges[idx][0]:self._ranges[idx][1]] = value

        # Fill Jacobian
        # Parallel implementation
        if self._n_proc > 1:
            with ProcessPoolExecutor(max_workers=self._n_proc) as executor:
                jobs = [executor.submit(parallel_pool_jac_eval, (f, x, i)) \
                        .add_done_callback(parallel_pool_jac_fill)
                        for i,f in enumerate(self._functions)]
        # Serial implementation
        else:
            for i,f in enumerate(self._functions):
                start = self._ranges[i][0]
                end = self._ranges[i][1]
                result[start:end,:] = f.jacobian (x)

    def jacobian (self, x):
        jac = numpy.zeros ((self.outputSize (), self.inputSize ()), order=self.order())
        self.impl_jacobian (jac, x)
        return jac


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

    def impl_jacobian (self, result, x):
        jacobian (self._fd, result, x)


class PyCachedFunction(PyDifferentiableFunction):
    def __init__ (self, f, size):
        PyDifferentiableFunction.__init__ \
            (self, f.inputSize (), f.outputSize (), \
             self._decodeName (f.name ()))
        self._cachedFunction = CachedFunction (f._function, size)

    def impl_compute (self, result, x):
        compute (self._cachedFunction, result, x)

    def impl_gradient (self, result, x, functionId):
        gradient (self._cachedFunction, result, x, functionId)

    def impl_jacobian (self, result, x):
        jacobian (self._cachedFunction, result, x)


class PyProblem(object):
    def __init__(self, cost):
        self.cost = cost
        self._problem = Problem (cost._function)
        self._constraints = list()

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
    def argumentScaling(self):
        return getArgumentScaling (self._problem)

    @argumentScaling.setter
    def argumentScaling(self, value):
        setArgumentScaling (self._problem, value)

    def addConstraint (self, constraint, bounds, scaling = None):
        addConstraint (self._problem, constraint._function, bounds, scaling)
        self._constraints.append (constraint)

    @property
    def constraints(self):
        return self._constraints


class PySolver(object):
    def __init__(self, solverName, problem, log_dir = None):
        self._solver = Solver (solverName, problem._problem)
        self._callbacks = list()
        self._multiplexer = Multiplexer (self._solver)
        self._logDir = log_dir

    def __str__ (self):
        return strSolver (self._solver)

    def solve (self):
        """
        Solve the RobOptim problem. If a log directory was provided, the
        optimization logger callback will be added to the callback multiplexer.
        """
        logger = None
        if self._logDir is not None \
           and os.access(os.path.dirname(self._logDir), os.W_OK):
            logger = addOptimizationLogger (self._solver, self._multiplexer, self._logDir)
        solve (self._solver)

        # Force deletion of logger to end logging
        del logger

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

    def addIterationCallback (self, callback):
        """
        Add an iteration callback to the callback multiplexer.
        """
        self._callbacks.append (callback)
        addIterationCallback (self._multiplexer, callback._callback)

    def removeIterationCallback (self, index):
        """
        Remove an iteration callback from the callback multiplexer.
        """
        removeIterationCallback (self._multiplexer, index)
        self._callbacks.pop (index)

    def setParameter (self, key, value, description=""):
        """
        Set a single solver parameter.
        """
        setSolverParameter (self._solver, key, value, description)

    @property
    def parameters(self):
        return getSolverParameters (self._solver)

    @parameters.setter
    def parameters(self, value):
        setSolverParameters (self._solver, value)


class PySolverState(object):
    def __init__(self, state):
        self._solverState = state

    def __str__ (self):
        return strSolverState (self._solverState)

    @property
    def x(self):
        return getSolverStateX (self._solverState)

    @x.setter
    def x(self, value):
        setSolverStateX (self._solverState, numpy.array(value))

    @property
    def cost(self):
        return getSolverStateCost (self._solverState)

    @cost.setter
    def cost(self, value):
        setSolverStateCost (self._solverState, value)

    @property
    def constraintViolation(self):
        return getSolverStateConstraintViolation (self._solverState)

    @constraintViolation.setter
    def constraintViolation(self, value):
        setSolverStateConstraintViolation (self._solverState, value)

    @property
    def parameters(self):
        return getSolverStateParameters (self._solverState)

    @parameters.setter
    def parameters(self, value):
        setSolverStateParameters (self._solverState, value)

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


class PySolverCallback(object):
    __metaclass__ = abc.ABCMeta

    def __init__ (self, pb):
        self._callback = SolverCallback (pb)
        bindSolverCallback (self._callback,
                            lambda pb, state:
                              self.callback (pb, PySolverState(state)))

    @abc.abstractmethod
    def callback (self, pb, state):
        return
