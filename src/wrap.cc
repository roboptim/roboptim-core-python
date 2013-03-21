#include <stdexcept>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <roboptim/core/differentiable-function.hh>
#include <roboptim/core/function.hh>
#include <roboptim/core/io.hh>
#include <roboptim/core/linear-function.hh>
#include <roboptim/core/problem.hh>
#include <roboptim/core/solver-factory.hh>
#include <roboptim/core/solver.hh>
#include <roboptim/core/twice-differentiable-function.hh>

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

	virtual ~Function () throw ()
	{
	  if (computeCallback_)
	    {
	      Py_DECREF (computeCallback_);
	      computeCallback_ = 0;
	    }
	}

	virtual void
	impl_compute (result_t& result, const argument_t& argument)
	  const throw ()
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
	explicit DifferentiableFunction (size_type inputSize,
					 size_type outputSize,
					 const std::string& name)
	  : roboptim::DifferentiableFunction (inputSize, outputSize, name),
	    Function (inputSize, outputSize, name),
	    gradientCallback_ (0),
	    jacobianCallback_ (0)
	{
	}

	virtual ~DifferentiableFunction () throw ()
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
	  const throw ()
	{
	  ::roboptim::core::python::Function::impl_compute (result, argument);
	}

	virtual void impl_gradient (gradient_t& gradient,
				    const argument_t& argument,
				    size_type functionId)
	  const throw ()
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

	virtual ~TwiceDifferentiableFunction () throw ()
	{
	  if (hessianCallback_)
	    {
	      Py_DECREF (hessianCallback_);
	      hessianCallback_ = 0;
	    }
	}

	virtual void impl_compute (result_t& result, const argument_t& argument)
	  const throw ()
	{
	  ::roboptim::core::python::Function::impl_compute (result, argument);
	}

	virtual void impl_gradient
	(gradient_t& gradient, const argument_t& argument, size_type functionId)
	  const throw ()
	{
	  ::roboptim::core::python::DifferentiableFunction::impl_gradient
	    (gradient, argument, functionId);
	}

	virtual void
	impl_hessian (hessian_t& hessian,
		      const argument_t& argument,
		      size_type functionId) const throw ()
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
    {"solve",  solve, METH_VARARGS,
     "Solve the optimization problem."},
    {"strFunction",  print<Function>, METH_VARARGS,
     "Print a function as a Python string."},
    {"strProblem",  print<problem_t>, METH_VARARGS,
     "Print a problem as a Python string."},
    {"strSolver",  print<factory_t>, METH_VARARGS,
     "Print a solver as a Python string."},
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
