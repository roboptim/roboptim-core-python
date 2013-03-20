#include <stdexcept>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <roboptim/core/function.hh>
#include <roboptim/core/differentiable-function.hh>

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
	: public ::roboptim::DifferentiableFunction,
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
	}

	virtual void impl_compute (result_t& result, const argument_t& argument)
	  const throw ()
	{
	  ::roboptim::core::python::Function::impl_compute (result, argument);
	}

	virtual void impl_gradient (gradient_t& gradient,
				    const argument_t& argument,
				    size_type /*functionId*/)
	  const throw ()
	{
	  gradient = argument;
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

      private:
	PyObject* gradientCallback_;
	PyObject* jacobianCallback_;
      };
    } // end of namespace python.
  } // end of namespace core.
} // end of namespace roboptim.

using roboptim::core::python::Function;
using roboptim::core::python::DifferentiableFunction;

static const char* ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME =
  "roboptim_core_function";

namespace detail
{
  template <typename T>
  static void destructor (PyObject* obj)
  {
    T* ptr = static_cast<T*>
      (PyCapsule_GetPointer
       (obj, ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME));
    assert (ptr && "failed to retrieve pointer from capsulte");
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
compute (PyObject*, PyObject* args)
{
  Function* function = 0;
  PyObject* x = 0;
  PyObject* result = 0;
  if (!PyArg_ParseTuple
      (args, "O&OO",
       detail::functionConverter, &function, &x, &result))
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
  std::cout << "foo" << dfunction << std::endl;
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


static PyMethodDef RobOptimCoreMethods[] =
  {
    {"Function",  createFunction<Function>, METH_VARARGS,
     "Create a Function object."},
    {"DifferentiableFunction",  createFunction<DifferentiableFunction>,
     METH_VARARGS, "Create a DifferentiableFunction object."},
    {"compute",  compute, METH_VARARGS,
     "Evaluate a function."},
    {"bindCompute",  bindCompute, METH_VARARGS,
     "Bind a Python function to function computation."},
    {"bindGradient",  bindGradient, METH_VARARGS,
     "Bind a Python function to gradient computation."},
    {0, 0, 0, 0}
  };

PyMODINIT_FUNC
initcore ()
{
  PyObject* m = 0;

  m = Py_InitModule("core", RobOptimCoreMethods);

  // Initialize numpy.
  import_array ();

  if (m == 0)
    return;
}
