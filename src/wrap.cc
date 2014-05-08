#include <stdexcept>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <boost/variant/static_visitor.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <roboptim/core/differentiable-function.hh>
#include <roboptim/core/function.hh>
#include <roboptim/core/io.hh>
#include <roboptim/core/linear-function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-factory.hh>
#include <roboptim/core/solver.hh>
#include <roboptim/core/twice-differentiable-function.hh>

#define FORWARD_TYPEDEFS(X)				\
  typedef X parent_t;					\
  typedef typename parent_t::result_t result_t;		\
  typedef typename parent_t::size_type size_type;	\
  typedef typename parent_t::argument_t argument_t;	\
  typedef typename parent_t::gradient_t gradient_t;	\
  typedef typename parent_t::jacobian_t jacobian_t


namespace roboptim
{
  namespace core
  {
    namespace python
    {
      class Function : public roboptim::Function
      {
      public:
	explicit Function (size_type inputSize,
			   size_type outputSize,
			   const std::string& name)
	  : roboptim::Function (inputSize, outputSize, name),
	    computeCallback_ (0)
	{
	}

	virtual ~Function ()
	{
	  if (computeCallback_)
	    {
	      Py_DECREF (computeCallback_);
	      computeCallback_ = 0;
	    }
	}

	virtual void
	impl_compute (result_t& result, const argument_t& argument)
	  const
	{
	  if (!computeCallback_)
	    {
	      PyErr_SetString
		(PyExc_TypeError,
		 "compute callback not set");
	      return;
	    }
	  if (!PyFunction_Check (computeCallback_))
	    {
	      PyErr_SetString
		(PyExc_TypeError,
		 "compute callback is not a function");
	      return;
	    }

	  npy_intp inputSize = static_cast<npy_intp> (this->inputSize ());
	  npy_intp outputSize = static_cast<npy_intp> (this->outputSize ());

	  PyObject* resultNumpy =
	    PyArray_SimpleNewFromData (1, &outputSize, NPY_DOUBLE, &result[0]);
	  if (!resultNumpy)
	    {
	      PyErr_SetString (PyExc_TypeError, "cannot convert result");
	      return;
	    }

	  PyObject* argNumpy =
	    PyArray_SimpleNewFromData
	    (1, &inputSize, NPY_DOUBLE, const_cast<double*> (&argument[0]));
	  if (!argNumpy)
	    {
	      PyErr_SetString (PyExc_TypeError, "cannot convert argument");
	      return;
	    }

	  PyObject* arglist = Py_BuildValue ("(OO)", resultNumpy, argNumpy);
	  if (!arglist)
	    {
	      Py_DECREF (arglist);
	      PyErr_SetString
		(PyExc_TypeError, "failed to build argument list");
	      return;
	    }

	  PyObject* resultPy = PyEval_CallObject (computeCallback_, arglist);
	  Py_DECREF (arglist);
	  Py_XDECREF(resultPy);
	}

	void
	setComputeCallback (PyObject* callback)
	{
	  if (callback == computeCallback_)
	    return;

	  if (computeCallback_)
	    {
	      Py_DECREF (computeCallback_);
	      computeCallback_ = 0;
	    }

	  Py_XINCREF (callback);
	  computeCallback_ = callback;
	}

      private:
	PyObject* computeCallback_;
      };

      class DifferentiableFunction
	: virtual public ::roboptim::DifferentiableFunction,
	  public ::roboptim::core::python::Function
      {
      public:
	FORWARD_TYPEDEFS (::roboptim::DifferentiableFunction);

	explicit DifferentiableFunction (size_type inputSize,
					 size_type outputSize,
					 const std::string& name)
	  : roboptim::DifferentiableFunction (inputSize, outputSize, name),
	    Function (inputSize, outputSize, name),
	    gradientCallback_ (0),
	    jacobianCallback_ (0)
	{
	}

	virtual ~DifferentiableFunction ()
	{
	  if (gradientCallback_)
	    {
	      Py_DECREF (gradientCallback_);
	      gradientCallback_ = 0;
	    }
	  if (jacobianCallback_)
	    {
	      Py_DECREF (jacobianCallback_);
	      jacobianCallback_ = 0;
	    }
	}

	virtual void impl_compute (result_t& result, const argument_t& argument)
	  const
	{
	  ::roboptim::core::python::Function::impl_compute (result, argument);
	}

	virtual void impl_gradient (gradient_t& gradient,
				    const argument_t& argument,
				    size_type functionId)
	  const
	{
	  if (!gradientCallback_)
	    {
	      PyErr_SetString
		(PyExc_TypeError,
		 "gradient callback not set");
	      return;
	    }
	  if (!PyFunction_Check (gradientCallback_))
	    {
	      PyErr_SetString
		(PyExc_TypeError,
		 "gradient callback is not a function");
	      return;
	    }

	  npy_intp inputSize = static_cast<npy_intp>
	    (::roboptim::core::python::Function::inputSize ());

	  PyObject* gradientNumpy =
	    PyArray_SimpleNewFromData (1, &inputSize, NPY_DOUBLE, &gradient[0]);
	  if (!gradientNumpy)
	    {
	      PyErr_SetString (PyExc_TypeError, "cannot convert result");
	      return;
	    }

	  PyObject* argNumpy =
	    PyArray_SimpleNewFromData
	    (1, &inputSize, NPY_DOUBLE, const_cast<double*> (&argument[0]));
	  if (!argNumpy)
	    {
	      PyErr_SetString (PyExc_TypeError, "cannot convert argument");
	      return;
	    }

	  PyObject* arglist =
	    Py_BuildValue ("(OOi)", gradientNumpy, argNumpy, functionId);
	  if (!arglist)
	    {
	      Py_DECREF (arglist);
	      PyErr_SetString
		(PyExc_TypeError, "failed to build argument list");
	      return;
	    }

	  PyObject* resultPy = PyEval_CallObject (gradientCallback_, arglist);
	  Py_DECREF (arglist);
	  Py_XDECREF(resultPy);
	}


	void
	setGradientCallback (PyObject* callback)
	{
	  if (gradientCallback_)
	    {
	      Py_DECREF (gradientCallback_);
	      gradientCallback_ = 0;
	    }

	  Py_XINCREF (callback);
	  gradientCallback_ = callback;
	}

	void
	setJacobianCallback (PyObject* callback)
	{
	  if (jacobianCallback_)
	    {
	      Py_DECREF (jacobianCallback_);
	      jacobianCallback_ = 0;
	    }

	  Py_XINCREF (callback);
	  jacobianCallback_ = callback;
	}

      private:
	PyObject* gradientCallback_;
	PyObject* jacobianCallback_;
      };

      class TwiceDifferentiableFunction
	: virtual public ::roboptim::TwiceDifferentiableFunction,
	  public ::roboptim::core::python::DifferentiableFunction
      {
      public:
	FORWARD_TYPEDEFS (::roboptim::TwiceDifferentiableFunction);

	explicit TwiceDifferentiableFunction (size_type inputSize,
					      size_type outputSize,
					      const std::string& name)
	  : ::roboptim::TwiceDifferentiableFunction (inputSize, outputSize, name),
	    ::roboptim::DifferentiableFunction
	    (inputSize, outputSize, name),
	    ::roboptim::core::python::DifferentiableFunction
	    (inputSize, outputSize, name),
	    hessianCallback_ (0)
	{
	}

	virtual ~TwiceDifferentiableFunction ()
	{
	  if (hessianCallback_)
	    {
	      Py_DECREF (hessianCallback_);
	      hessianCallback_ = 0;
	    }
	}

	virtual void impl_compute (result_t& result, const argument_t& argument)
	  const
	{
	  ::roboptim::core::python::Function::impl_compute (result, argument);
	}

	virtual void impl_gradient
	(gradient_t& gradient, const argument_t& argument, size_type functionId)
	  const
	{
	  ::roboptim::core::python::DifferentiableFunction::impl_gradient
	    (gradient, argument, functionId);
	}

	virtual void
	impl_hessian (hessian_t& hessian,
		      const argument_t& argument,
		      size_type functionId) const
	{
	  //FIXME: implement this.
	}

      private:
	PyObject* hessianCallback_;
      };

    } // end of namespace python.
  } // end of namespace core.
} // end of namespace roboptim.

using roboptim::core::python::Function;
using roboptim::core::python::DifferentiableFunction;
using roboptim::core::python::TwiceDifferentiableFunction;

static const char* ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME =
  "roboptim_core_function";
static const char* ROBOPTIM_CORE_PROBLEM_CAPSULE_NAME =
  "roboptim_core_problem";
static const char* ROBOPTIM_CORE_SOLVER_CAPSULE_NAME =
  "roboptim_core_solver";
static const char* ROBOPTIM_CORE_RESULT_CAPSULE_NAME =
  "roboptim_core_result";
static const char* ROBOPTIM_CORE_RESULT_WITH_WARNINGS_CAPSULE_NAME =
  "roboptim_core_result_with_warnings";
static const char* ROBOPTIM_CORE_SOLVER_ERROR_CAPSULE_NAME =
  "roboptim_core_solver_error";

typedef roboptim::Problem<
  ::roboptim::DifferentiableFunction,
  boost::mpl::vector< ::roboptim::LinearFunction,
		      ::roboptim::DifferentiableFunction> >
problem_t;

typedef roboptim::Solver<
  ::roboptim::DifferentiableFunction,
  boost::mpl::vector< ::roboptim::LinearFunction,
		      ::roboptim::DifferentiableFunction> >
			  solver_t;

typedef roboptim::SolverFactory<solver_t> factory_t;

typedef roboptim::Result result_t;
typedef roboptim::ResultWithWarnings resultWithWarnings_t;
typedef roboptim::SolverError solverError_t;
typedef roboptim::Parameter parameter_t;
typedef solver_t::parameters_t parameters_t;

namespace detail
{
  template <typename T>
  void destructor (PyObject* obj);

  template <>
  void destructor<problem_t> (PyObject* obj)
  {
    problem_t* ptr = static_cast<problem_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_PROBLEM_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsule");
    if (ptr)
      delete ptr;
  }

  template <>
  void destructor<factory_t> (PyObject* obj)
  {
    factory_t* ptr = static_cast<factory_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_SOLVER_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsule");
    if (ptr)
      delete ptr;
  }

  template <>
  void destructor<result_t> (PyObject* obj)
  {
    result_t* ptr = static_cast<result_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_RESULT_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsule");
    if (ptr)
      delete ptr;
  }

  template <>
  void destructor<resultWithWarnings_t> (PyObject* obj)
  {
    resultWithWarnings_t* ptr = static_cast<resultWithWarnings_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_RESULT_WITH_WARNINGS_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsule");
    if (ptr)
      delete ptr;
  }

  template <>
  void destructor<solverError_t> (PyObject* obj)
  {
    solverError_t* ptr = static_cast<solverError_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_SOLVER_ERROR_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsule");
    if (ptr)
      delete ptr;
  }

  template <typename T>
  void destructor (PyObject* obj)
  {
    T* ptr = static_cast<T*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsule");
    if (ptr)
      delete ptr;
  }

  int
  functionConverter (PyObject* obj, Function** address)
  {
    assert (address);
    Function* ptr = static_cast<Function*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME));
    if (!ptr)
      {
	PyErr_SetString
	  (PyExc_TypeError,
	   "Function object expected but another type was passed");
	return 0;
      }
    *address = ptr;
    return 1;
  }

  int
  problemConverter (PyObject* obj, problem_t** address)
  {
    assert (address);
    problem_t* ptr = static_cast<problem_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_PROBLEM_CAPSULE_NAME));
    if (!ptr)
      {
	PyErr_SetString
	  (PyExc_TypeError,
	   "Problem object expected but another type was passed");
	return 0;
      }
    *address = ptr;
    return 1;
  }

  int
  factoryConverter (PyObject* obj, factory_t** address)
  {
    assert (address);
    factory_t* ptr = static_cast<factory_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_SOLVER_CAPSULE_NAME));
    if (!ptr)
      {
	PyErr_SetString
	  (PyExc_TypeError,
	   "Problem object expected but another type was passed");
	return 0;
      }
    *address = ptr;
    return 1;
  }

  int
  resultConverter (PyObject* obj, result_t** address)
  {
    assert (address);
    result_t* ptr = static_cast<result_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_RESULT_CAPSULE_NAME));
    if (!ptr)
      {
	PyErr_SetString
	  (PyExc_TypeError,
	   "Result object expected but another type was passed");
	return 0;
      }
    *address = ptr;
    return 1;
  }

  int
  resultWithWarningsConverter (PyObject* obj, resultWithWarnings_t** address)
  {
    assert (address);
    resultWithWarnings_t* ptr = static_cast<resultWithWarnings_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_RESULT_WITH_WARNINGS_CAPSULE_NAME));
    if (!ptr)
      {
	PyErr_SetString
	  (PyExc_TypeError,
	   "ResultWithWarnings object expected but another type was passed");
	return 0;
      }
    *address = ptr;
    return 1;
  }

  int
  solverErrorConverter (PyObject* obj, solverError_t** address)
  {
    assert (address);
    solverError_t* ptr = static_cast<solverError_t*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_SOLVER_ERROR_CAPSULE_NAME));
    if (!ptr)
      {
	PyErr_SetString
	  (PyExc_TypeError,
	   "SolverError object expected but another type was passed");
	return 0;
      }
    *address = ptr;
    return 1;
  }

  struct ParameterValueVisitor : public boost::static_visitor<PyObject*>
  {
    PyObject* operator () (const roboptim::Function::value_type& p) const
    {
      return PyFloat_FromDouble (p);
    }

    PyObject* operator () (const int& p) const
    {
      return PyInt_FromLong (p);
    }

    PyObject* operator () (const std::string& p) const
    {
      return PyString_FromString (p.c_str ());
    }
  };

  parameter_t::parameterValues_t toParameterValue (PyObject* obj)
  {
    // Value can be: double, int, or std::string

    // String
    if (PyString_Check (obj))
      {
	return PyString_AsString (obj);
      }
    // Unicode string
    else if (PyUnicode_Check (obj))
      {
	return PyString_AsString (PyUnicode_AsASCIIString (obj));
      }
    // Integer
    else if (PyInt_Check (obj))
      {
        return static_cast<int> (PyInt_AsLong (obj));
      }
    // Double
    else if (PyFloat_Check (obj))
      {
        return PyFloat_AsDouble (obj);
      }

    PyErr_SetString
      (PyExc_TypeError,
       "invalid parameter value (should be double, int or string).");

    return 0;
  }
} // end of namespace detail.

template <typename T>
static PyObject*
createFunction (PyObject*, PyObject* args)
{
  Function::size_type inSize = 0;
  Function::size_type outSize = 0;
  const char* name = 0;

  if (!PyArg_ParseTuple(args, "iiz", &inSize, &outSize, &name))
    return 0;

  std::string name_ = (name) ? name : "";
  T* function = new T (inSize, outSize, name_);

  PyObject* functionPy =
    PyCapsule_New (function, ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME,
		   &detail::destructor<T>);
  return functionPy;
}

static PyObject*
inputSize (PyObject*, PyObject* args)
{
  Function* function = 0;
  if (!PyArg_ParseTuple(args, "O&", &detail::functionConverter, &function))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "argument 1 should be a function object");
      return 0;
    }
  return Py_BuildValue("i", function->inputSize ());
}

static PyObject*
outputSize (PyObject*, PyObject* args)
{
  Function* function = 0;
  if (!PyArg_ParseTuple(args, "O&", &detail::functionConverter, &function))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "argument 1 should be a function object");
      return 0;
    }
  return Py_BuildValue("i", function->outputSize ());
}

static PyObject*
getName (PyObject*, PyObject* args)
{
  Function* function = 0;
  if (!PyArg_ParseTuple(args, "O&", &detail::functionConverter, &function))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "argument 1 should be a function object");
      return 0;
    }
  return Py_BuildValue("s", function->getName ().c_str ());
}

static PyObject*
createProblem (PyObject*, PyObject* args)
{
  Function* costFunction = 0;
  if (!PyArg_ParseTuple(args, "O&", &detail::functionConverter, &costFunction))
    return 0;

  DifferentiableFunction* dfunction =
    dynamic_cast<DifferentiableFunction*> (costFunction);
  if (!dfunction)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "argument 1 should be a differentiable function object");
      return 0;
    }

  problem_t* problem = new problem_t (*dfunction);
  PyObject* problemPy =
    PyCapsule_New (problem, ROBOPTIM_CORE_PROBLEM_CAPSULE_NAME,
		   &detail::destructor<problem_t>);

  return problemPy;
}

static PyObject*
createSolver (PyObject*, PyObject* args)
{
  char* pluginName = 0;
  problem_t* problem = 0;
  if (!PyArg_ParseTuple (args, "sO&",
			 &pluginName,
			 &detail::problemConverter, &problem))
    return 0;

  factory_t* factory = 0;

  try
    {
      factory = new factory_t (pluginName, *problem);
    }
  catch (...)
    {
      delete factory;
      Py_INCREF (Py_None);
      return Py_None;
    }

  PyObject* solverPy =
    PyCapsule_New (factory, ROBOPTIM_CORE_SOLVER_CAPSULE_NAME,
		   &detail::destructor<factory_t>);
  return solverPy;
}

static PyObject*
compute (PyObject*, PyObject* args)
{
  Function* function = 0;
  PyObject* x = 0;
  PyObject* result = 0;
  if (!PyArg_ParseTuple
      (args, "O&OO",
       detail::functionConverter, &function, &result, &x))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Failed to retrieve function object");
      return 0;
    }

  PyObject* resultNumpy =
    PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_OUT_ARRAY & NPY_C_CONTIGUOUS);

  // Try to build an array type from x.
  // All types providing a sequence interface are compatible.
  // Tuples, sequences and Numpy types for instance.
  PyObject* xNumpy =
    PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY & NPY_C_CONTIGUOUS);
  if (!xNumpy)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Argument cannot be converted to Numpy object");
      return 0;
    }

  // Directly map Eigen vector over Numpy x.
  Eigen::Map<Function::argument_t> xEigen
    (static_cast<double*> (PyArray_DATA (xNumpy)), function->inputSize ());

  // Directly map Eigen result to the numpy array data.
  Eigen::Map<Function::result_t> resultEigen
    (static_cast<double*>
     (PyArray_DATA (resultNumpy)), function->outputSize ());

  resultEigen = (*function) (xEigen);
  if (PyErr_Occurred ())
    return 0;

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
gradient (PyObject*, PyObject* args)
{
  Function* function = 0;
  PyObject* x = 0;
  PyObject* gradient = 0;
  Function::size_type functionId = 0;
  if (!PyArg_ParseTuple
      (args, "O&OOi",
       detail::functionConverter, &function, &gradient, &x, &functionId))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Failed to retrieve function object");
      return 0;
    }

  DifferentiableFunction* dfunction =
    dynamic_cast<DifferentiableFunction*> (function);
  if (!dfunction)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "argument 1 should be a differentiable function object");
      return 0;
    }

  PyObject* gradientNumpy =
    PyArray_FROM_OTF(gradient, NPY_DOUBLE, NPY_OUT_ARRAY & NPY_C_CONTIGUOUS);

  // Try to build an array type from x.
  // All types providing a sequence interface are compatible.
  // Tuples, sequences and Numpy types for instance.
  PyObject* xNumpy =
    PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_IN_ARRAY & NPY_C_CONTIGUOUS);
  if (!xNumpy)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Argument cannot be converted to Numpy object");
      return 0;
    }

  // Directly map Eigen vector over Numpy x.
  Eigen::Map<Function::argument_t> xEigen
    (static_cast<double*> (PyArray_DATA (xNumpy)), function->inputSize ());

  // Directly map Eigen result to the numpy array data.
  Eigen::Map<Function::result_t> gradientEigen
    (static_cast<double*>
     (PyArray_DATA (gradientNumpy)), dfunction->gradientSize ());

  gradientEigen = dfunction->gradient (xEigen, functionId);
  if (PyErr_Occurred ())
    return 0;

  Py_INCREF(Py_None);
  return Py_None;
}


static PyObject*
bindCompute (PyObject*, PyObject* args)
{
  Function* function = 0;
  PyObject* callback = 0;
  if (!PyArg_ParseTuple
      (args, "O&O:bindCompute",
       detail::functionConverter, &function, &callback))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Failed to retrieve function object");
      return 0;
    }
  if (!callback)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Failed to retrieve callback object");
      return 0;
    }
  if (!PyCallable_Check (callback))
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "2nd argument must be callable");
      return 0;
    }

  function->setComputeCallback (callback);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
bindGradient (PyObject*, PyObject* args)
{
  Function* function = 0;
  PyObject* callback = 0;
  if (!PyArg_ParseTuple
      (args, "O&O:bindGradient",
       detail::functionConverter, &function, &callback))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Failed to retrieve function object");
      return 0;
    }

  DifferentiableFunction* dfunction
    = dynamic_cast<DifferentiableFunction*> (function);

  if (!dfunction)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "instance of DifferentiableFunction expected as first argument");
      return 0;
    }
  if (!callback)
    {
      PyErr_SetString (PyExc_TypeError, "Failed to retrieve callback object");
      return 0;
    }
  if (!PyCallable_Check (callback))
    {
      PyErr_SetString (PyExc_TypeError, "2nd argument must be callable");
      return 0;
    }

  dfunction->setGradientCallback (callback);

  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject*
getStartingPoint (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  if (!PyArg_ParseTuple (args, "O&", &detail::problemConverter, &problem))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }

  npy_intp inputSize =
    static_cast<npy_intp> (problem->function ().inputSize ());

  PyObject* startingPoint = PyArray_SimpleNew(1, &inputSize, NPY_DOUBLE);
  Eigen::Map<Function::vector_t> startingPointEigen
    (static_cast<double*>
     (PyArray_DATA (startingPoint)), problem->function ().inputSize ());

  if (!problem->startingPoint ())
    {
      Py_INCREF (Py_None);
      return Py_None;
    }

  startingPointEigen = *(problem->startingPoint ());
  return startingPoint;
}

static PyObject*
setStartingPoint (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  PyObject* startingPoint = 0;
  if (!PyArg_ParseTuple
      (args, "O&O", &detail::problemConverter, &problem, &startingPoint))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }
  PyObject* startingPointNumpy =
    PyArray_FROM_OTF
    (startingPoint, NPY_DOUBLE, NPY_IN_ARRAY & NPY_C_CONTIGUOUS);
  if (!startingPointNumpy)
    {
      PyErr_SetString (PyExc_TypeError,
		       "failed to build numpy array from 2st argument");
      return 0;
    }

  if (PyArray_DIM (startingPointNumpy, 0) != problem->function ().inputSize ())
    {
      PyErr_SetString (PyExc_TypeError, "invalid size");
      return 0;
    }

  Eigen::Map<Function::argument_t> startingPointEigen
    (static_cast<double*> (PyArray_DATA (startingPointNumpy)),
     problem->function ().inputSize ());

  problem->startingPoint () = startingPointEigen;

  Py_INCREF (Py_None);
  return Py_None;
}

static PyObject*
getArgumentBounds (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  if (!PyArg_ParseTuple (args, "O&", &detail::problemConverter, &problem))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }

  npy_intp sdims[2] =
    {
      static_cast<npy_intp> (problem->function ().inputSize ()),
      2
    };

  PyObject* bounds = PyArray_SimpleNew (2, sdims, NPY_DOUBLE);

  for (unsigned i = 0; i < problem->function ().inputSize (); ++i)
    {
      *static_cast<double*> (PyArray_GETPTR2 (bounds, i, 0)) =
	problem->argumentBounds ()[i].first;
      *static_cast<double*> (PyArray_GETPTR2 (bounds, i, 1)) =
	problem->argumentBounds ()[i].second;
    }

  return bounds;
}

static PyObject*
setArgumentBounds (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  PyObject* bounds = 0;
  if (!PyArg_ParseTuple
      (args, "O&O", &detail::problemConverter, &problem, &bounds))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }
  PyObject* boundsNumpy =
    PyArray_FROM_OTF
    (bounds, NPY_DOUBLE, NPY_IN_ARRAY & NPY_C_CONTIGUOUS);
  if (!boundsNumpy)
    {
      PyErr_SetString (PyExc_TypeError,
		       "failed to build numpy array from 2st argument");
      return 0;
    }

  if (PyArray_DIM (bounds, 0) != problem->function ().inputSize ())
    {
      PyErr_SetString (PyExc_TypeError, "invalid size");
      return 0;
    }

  if (PyArray_DIM (bounds, 1) != 2)
    {
      PyErr_SetString (PyExc_TypeError, "invalid size");
      return 0;
    }

  Eigen::Map<Function::argument_t> boundsEigen
    (static_cast<double*> (PyArray_DATA (boundsNumpy)),
     problem->function ().inputSize ());

  for (unsigned i = 0; i < problem->function ().inputSize (); ++i)
    {
      problem->argumentBounds ()[i].first =
	*static_cast<double*> (PyArray_GETPTR2 (bounds, i, 0));
      problem->argumentBounds ()[i].second =
	*static_cast<double*> (PyArray_GETPTR2 (bounds, i, 1));
    }

  Py_INCREF (Py_None);
  return Py_None;
}

static PyObject*
getArgumentScales (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  if (!PyArg_ParseTuple (args, "O&", &detail::problemConverter, &problem))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }

  npy_intp inputSize =
    static_cast<npy_intp> (problem->function ().inputSize ());
  PyObject* scalesNumpy = PyArray_SimpleNew (1, &inputSize, NPY_DOUBLE);

  Eigen::Map<Function::vector_t> scalesEigen
    (static_cast<double*> (PyArray_DATA (scalesNumpy)),
     problem->function ().inputSize ());

  for (Function::size_type i = 0; i < inputSize; ++i)
    scalesEigen[i] = problem->argumentScales ()[i];
  return scalesNumpy;
}

static PyObject*
setArgumentScales (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  PyObject* scales = 0;
  if (!PyArg_ParseTuple
      (args, "O&O", &detail::problemConverter, &problem, &scales))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }
  PyObject* scalesNumpy =
    PyArray_FROM_OTF
    (scales, NPY_DOUBLE, NPY_IN_ARRAY & NPY_C_CONTIGUOUS);
  if (!scalesNumpy)
    {
      PyErr_SetString (PyExc_TypeError,
		       "failed to build numpy array from 2st argument");
      return 0;
    }

  if (PyArray_DIM (scalesNumpy, 0) != problem->function ().inputSize ())
    {
      PyErr_SetString (PyExc_TypeError, "invalid size");
      return 0;
    }

  Eigen::Map<Function::argument_t> scalesEigen
    (static_cast<double*> (PyArray_DATA (scalesNumpy)),
     problem->function ().inputSize ());

  for (Function::size_type i = 0; i < problem->function ().inputSize (); ++i)
    problem->argumentScales ()[i] = scalesEigen[i];


  Py_INCREF (Py_None);
  return Py_None;
}


static PyObject*
addConstraint (PyObject*, PyObject* args)
{
  problem_t* problem = 0;
  Function* function = 0;
  double min = 0.;
  double max = 0.;

  if (!PyArg_ParseTuple
      (args, "O&O&(dd)",
       &detail::problemConverter, &problem,
       &detail::functionConverter, &function,
       &min, &max))
    return 0;
  if (!problem)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a problem");
      return 0;
    }
  if (!function)
    {
      PyErr_SetString (PyExc_TypeError, "2nd argument must be a function");
      return 0;
    }

  DifferentiableFunction* dfunction =
    dynamic_cast<DifferentiableFunction*> (function);

  if (!dfunction)
    {
      PyErr_SetString (PyExc_TypeError,
		       "2nd argument must be a differentiable function");
      return 0;
    }

  //FIXME: this will make everything segv.
  //contraint will be freed when problem disappear...
  // boost::shared_ptr<DifferentiableFunction> constraint (dfunction);
  // problem->addConstraint (constraint, Function::makeInterval (min, max));

  Py_INCREF (Py_None);
  return Py_None;
}


static PyObject*
solve (PyObject*, PyObject* args)
{
  factory_t* factory = 0;
  if (!PyArg_ParseTuple (args, "O&",
			 &detail::factoryConverter, &factory))
    return 0;

  (*factory) ().solve ();
  Py_INCREF (Py_None);
  return Py_None;
}

static PyObject*
minimum (PyObject*, PyObject* args)
{
  factory_t* factory = 0;
  if (!PyArg_ParseTuple (args, "O&",
			 &detail::factoryConverter, &factory))
    return 0;

  solver_t::result_t result = (*factory) ().minimum ();

  npy_intp inputSize = static_cast<npy_intp>
    ((*factory) ().problem ().function ().inputSize ());

  switch (result.which ())
    {
      // should never happen
    case solver_t::SOLVER_NO_SOLUTION:
      {
	PyErr_SetString (PyExc_TypeError, "problem not yet solved");
	return 0;
      }
    case solver_t::SOLVER_VALUE:
      {
	result_t* result_ = new result_t (boost::get<result_t> (result));
	PyObject* resultPy =
	  PyCapsule_New (result_, ROBOPTIM_CORE_RESULT_CAPSULE_NAME,
			 &detail::destructor<result_t>);
	return Py_BuildValue
	  ("(s,O)", ROBOPTIM_CORE_RESULT_CAPSULE_NAME, resultPy);
      }
    case solver_t::SOLVER_VALUE_WARNINGS:
      {
	resultWithWarnings_t* warn =
	  new resultWithWarnings_t (boost::get<resultWithWarnings_t> (result));
	PyObject* warnPy =
	  PyCapsule_New (warn, ROBOPTIM_CORE_RESULT_WITH_WARNINGS_CAPSULE_NAME,
			 &detail::destructor<resultWithWarnings_t>);
	return Py_BuildValue
	  ("(s,O)", ROBOPTIM_CORE_RESULT_WITH_WARNINGS_CAPSULE_NAME, warnPy);
      }
    case solver_t::SOLVER_ERROR:
      {
	solverError_t* err = new solverError_t
	  (boost::get<solverError_t> (result));
	PyObject* errPy =
	  PyCapsule_New (err, ROBOPTIM_CORE_SOLVER_ERROR_CAPSULE_NAME,
			 &detail::destructor<solverError_t>);
	return Py_BuildValue
	  ("(s,O)", ROBOPTIM_CORE_SOLVER_ERROR_CAPSULE_NAME, errPy);
      }
    }
  Py_INCREF (Py_None);
  return Py_None;
}

static PyObject*
getParameter (const parameter_t& parameter)
{
  PyObject* description = PyString_FromString (parameter.description.c_str ());
  PyObject* value = boost::apply_visitor (detail::ParameterValueVisitor (),
                                          parameter.value);

  return PyTuple_Pack (2, description, value);
}

static PyObject*
getParameters (PyObject*, PyObject* args)
{
  factory_t* factory = 0;
  if (!PyArg_ParseTuple (args, "O&",
			 &detail::factoryConverter, &factory))
    return 0;

  if (!factory)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a solver.");
      return 0;
    }

  solver_t& solver = (*factory) ();

  // In C++, parameters are: std::map<std::string, Parameter>
  PyObject* parameters = PyDict_New ();

  for (parameters_t::const_iterator iter = solver.parameters ().begin ();
       iter != solver.parameters ().end (); iter++)
    {
      // Insert object to Python dictionary
      PyDict_SetItemString (parameters, (iter->first).c_str (),
                            getParameter (iter->second));
    }

  return Py_BuildValue ("O", parameters);
}

static PyObject*
setParameters (PyObject*, PyObject* args)
{
  factory_t* factory = 0;
  PyObject* py_parameters = 0;

  if (!PyArg_ParseTuple (args, "O&O",
			 &detail::factoryConverter, &factory, &py_parameters))
    return 0;

  if (!factory)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a solver.");
      return 0;
    }

  if (!PyDict_Check (py_parameters))
    {
      PyErr_SetString (PyExc_TypeError, "2nd argument must be a dictionary.");
      return 0;
    }

  solver_t& solver = (*factory) ();

  // In C++, parameters are: std::map<std::string, Parameter>
  parameters_t& parameters = solver.parameters ();
  parameters.clear ();

  PyObject *key, *value;
  Py_ssize_t pos = 0;
  parameter_t parameter;

  // Iterate over dictionary
  while (PyDict_Next (py_parameters, &pos, &key, &value))
    {
      std::string str_key = "";

      if (PyBytes_Check (key))
	{
	  str_key = PyBytes_AsString (key);
	}
      else if (PyUnicode_Check (key))
	{
	  str_key = PyBytes_AsString (PyUnicode_AsASCIIString (key));
	}
      else
	{
	  continue;
	}
      if (!PyTuple_Check (value))
        continue;

      parameter.description = PyBytes_AsString (PyTuple_GetItem (value, 0));
      parameter.value = detail::toParameterValue (PyTuple_GetItem (value, 1));
      parameters[str_key] = parameter;
    }

  Py_INCREF (Py_None);
  return Py_None;
}

template <typename T>
PyObject*
toDict (PyObject*, PyObject* args);

template <>
PyObject*
toDict<result_t> (PyObject*, PyObject* args)
{
  result_t* result = 0;
  if (!PyArg_ParseTuple (args, "O&",
			 &detail::resultConverter, &result))
    return 0;

  if (!result)
    {
      PyErr_SetString (PyExc_TypeError, "1st argument must be a result.");
      return 0;
    }

  // In C++, parameters are: std::map<std::string, Parameter>
  PyObject* dict_result = PyDict_New ();

  PyDict_SetItemString (dict_result, "inputSize",
                        PyInt_FromLong (result->inputSize));
  PyDict_SetItemString (dict_result, "outputSize",
                        PyInt_FromLong (result->outputSize));


  npy_intp npy_size = static_cast<npy_intp> (result->x.size ());
  PyObject* xNumpy =
    PyArray_SimpleNewFromData (1, &npy_size,
			       NPY_DOUBLE, result->x.data ());
  if (!xNumpy)
    {
      PyErr_SetString (PyExc_TypeError, "cannot convert result.x");
      return 0;
    }
  PyDict_SetItemString (dict_result, "x", xNumpy);

  npy_size = static_cast<npy_intp> (result->value.size ());
  PyObject* valueNumpy =
    PyArray_SimpleNewFromData (1, &npy_size,
			       NPY_DOUBLE, result->value.data ());
  if (!valueNumpy)
    {
      PyErr_SetString (PyExc_TypeError, "cannot convert result.value");
      return 0;
    }
  PyDict_SetItemString (dict_result, "value", valueNumpy);

  npy_size = static_cast<npy_intp> (result->constraints.size ());
  PyObject* constraintsNumpy =
    PyArray_SimpleNewFromData (1, &npy_size,
			       NPY_DOUBLE, result->constraints.data ());
  if (!constraintsNumpy)
    {
      PyErr_SetString (PyExc_TypeError, "cannot convert result.constraints");
      return 0;
    }
  PyDict_SetItemString (dict_result, "constraints", constraintsNumpy);

  npy_size = static_cast<npy_intp> (result->lambda.size ());
  PyObject* lambdaNumpy =
    PyArray_SimpleNewFromData (1, &npy_size,
			       NPY_DOUBLE, result->lambda.data ());
  if (!lambdaNumpy)
    {
      PyErr_SetString (PyExc_TypeError, "cannot convert result.lambda");
      return 0;
    }
  PyDict_SetItemString (dict_result, "lambda", lambdaNumpy);

  return Py_BuildValue ("O", dict_result);
}

template <typename T>
PyObject*
print (PyObject*, PyObject* args);

template <>
PyObject*
print<problem_t> (PyObject*, PyObject* args)
{
  problem_t* obj = 0;
  if (!PyArg_ParseTuple
      (args, "O&", &detail::problemConverter, &obj))
    return 0;
  if (!obj)
    {
      PyErr_SetString (PyExc_TypeError, "failed to retrieve object");
      return 0;
    }

  std::stringstream ss;
  ss << *obj;

  return Py_BuildValue ("s", ss.str ().c_str ());
}

template <>
PyObject*
print<factory_t> (PyObject*, PyObject* args)
{
  factory_t* obj = 0;
  if (!PyArg_ParseTuple
      (args, "O&", &detail::factoryConverter, &obj))
    return 0;
  if (!obj)
    {
      PyErr_SetString (PyExc_TypeError, "failed to retrieve object");
      return 0;
    }

  std::stringstream ss;
  ss << (*obj) ();

  return Py_BuildValue ("s", ss.str ().c_str ());
}

template <>
PyObject*
print<result_t> (PyObject*, PyObject* args)
{
  result_t* obj = 0;
  if (!PyArg_ParseTuple
      (args, "O&", &detail::resultConverter, &obj))
    return 0;
  if (!obj)
    {
      PyErr_SetString (PyExc_TypeError, "failed to retrieve object");
      return 0;
    }

  std::stringstream ss;
  ss << (*obj);

  return Py_BuildValue ("s", ss.str ().c_str ());
}

template <>
PyObject*
print<resultWithWarnings_t> (PyObject*, PyObject* args)
{
  resultWithWarnings_t* obj = 0;
  if (!PyArg_ParseTuple
      (args, "O&", &detail::resultWithWarningsConverter, &obj))
    return 0;
  if (!obj)
    {
      PyErr_SetString (PyExc_TypeError, "failed to retrieve object");
      return 0;
    }

  std::stringstream ss;
  ss << (*obj);

  return Py_BuildValue ("s", ss.str ().c_str ());
}

template <>
PyObject*
print<solverError_t> (PyObject*, PyObject* args)
{
  solverError_t* obj = 0;
  if (!PyArg_ParseTuple
      (args, "O&", &detail::solverErrorConverter, &obj))
    return 0;
  if (!obj)
    {
      PyErr_SetString (PyExc_TypeError, "failed to retrieve object");
      return 0;
    }

  std::stringstream ss;
  ss << (*obj);

  return Py_BuildValue ("s", ss.str ().c_str ());
}

template <typename T>
PyObject*
print (PyObject*, PyObject* args)
{
  T* obj = 0;
  if (!PyArg_ParseTuple
      (args, "O&", &detail::functionConverter, &obj))
    return 0;
  if (!obj)
    {
      PyErr_SetString (PyExc_TypeError, "failed to retrieve object");
      return 0;
    }

  std::stringstream ss;
  ss << *obj;

  return Py_BuildValue ("s", ss.str ().c_str ());
}

static PyMethodDef RobOptimCoreMethods[] =
  {
    {"Function",  createFunction<Function>, METH_VARARGS,
     "Create a Function object."},
    {"inputSize",  inputSize, METH_VARARGS,
     "Return function input size."},
    {"outputSize",  outputSize, METH_VARARGS,
     "Return function output size."},
    {"getName",  getName, METH_VARARGS,
     "Return function name."},

    {"DifferentiableFunction",  createFunction<DifferentiableFunction>,
     METH_VARARGS, "Create a DifferentiableFunction object."},
    {"TwiceDifferentiableFunction", createFunction<TwiceDifferentiableFunction>,
     METH_VARARGS, "Create a TwiceDifferentiableFunction object."},
    {"Problem",  createProblem, METH_VARARGS,
     "Create a Problem object."},
    {"Solver",  createSolver, METH_VARARGS,
     "Create a Solver object through the solver factory."},
    {"compute",  compute, METH_VARARGS,
     "Evaluate a function."},
    {"gradient",  gradient, METH_VARARGS,
     "Evaluate a function gradient."},
    {"bindCompute",  bindCompute, METH_VARARGS,
     "Bind a Python function to function computation."},
    {"bindGradient",  bindGradient, METH_VARARGS,
     "Bind a Python function to gradient computation."},

    {"getStartingPoint", getStartingPoint, METH_VARARGS,
     "Get the problem starting point."},
    {"setStartingPoint", setStartingPoint, METH_VARARGS,
     "Set the problem starting point."},
    {"getArgumentBounds", getArgumentBounds, METH_VARARGS,
     "Get the problem argument bounds."},
    {"setArgumentBounds", setArgumentBounds, METH_VARARGS,
     "Set the problem argument bounds."},
    {"getArgumentScales", getArgumentScales, METH_VARARGS,
     "Get the problem scales."},
    {"setArgumentScales", setArgumentScales, METH_VARARGS,
     "Set the problem scales."},
    {"addConstraint", addConstraint, METH_VARARGS,
     "Add a constraint to the problem."},

    // Solver functions
    {"solve",  solve, METH_VARARGS,
     "Solve the optimization problem."},
    {"minimum",  minimum, METH_VARARGS,
     "Retrieve the optimization result."},
    {"getParameters", getParameters, METH_VARARGS,
     "Get the solver parameters."},
    {"setParameters", setParameters, METH_VARARGS,
     "Set the solver parameters."},

    // Result functions
    {"resultToDict", toDict<result_t>, METH_VARARGS,
     "Convert a Result object to a Python dictionary."},

    // Print functions
    {"strFunction",  print<Function>, METH_VARARGS,
     "Print a function as a Python string."},
    {"strProblem",  print<problem_t>, METH_VARARGS,
     "Print a problem as a Python string."},
    {"strSolver",  print<factory_t>, METH_VARARGS,
     "Print a solver as a Python string."},
    {"strResult",  print<result_t>, METH_VARARGS,
     "Print a result as a Python string."},
    {"strResultWithWarnings",  print<resultWithWarnings_t>, METH_VARARGS,
     "Print a result as a Python string."},
    {"strSolverError",  print<solverError_t>, METH_VARARGS,
     "Print a solver error as a Python string."},
    {0, 0, 0, 0}
  };

PyMODINIT_FUNC
initwrap ()
{
  PyObject* m = 0;

  m = Py_InitModule("wrap", RobOptimCoreMethods);

  // Initialize numpy.
  import_array ();

  if (m == 0)
    return;
}
