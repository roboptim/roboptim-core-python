#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>
#include <roboptim/core/function.hh>

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
	  : roboptim::Function (inputSize, outputSize, name)
	{
	}

	virtual void
	impl_compute (result_t& result, const argument_t& argument)
	  const throw ()
	{
	  result[0] = 2 * argument[0]; //FIXME:
	}
      };
    } // end of namespace python.
  } // end of namespace core.
} // end of namespace roboptim.

using roboptim::core::python::Function;

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

static PyObject*
createFunction (PyObject*, PyObject* args)
{
  Function::size_type inSize = 0;
  Function::size_type outSize = 0;
  const char* name = 0;

  if (!PyArg_ParseTuple(args, "iiz", &inSize, &outSize, &name))
    return 0;

  std::string name_ = (name) ? name : "";
  Function* function = new Function (inSize, outSize, name_);

  PyObject* functionPy =
    PyCapsule_New (function, ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME,
		   &detail::destructor<Function>);
  return functionPy;
}

static PyObject*
compute (PyObject*, PyObject* args)
{
  Function* function = 0;
  PyObject* x = 0;
  if (!PyArg_ParseTuple(args, "O&O", detail::functionConverter, &function, &x))
    return 0;
  if (!function)
    {
      PyErr_SetString
	(PyExc_TypeError,
	 "Failed to retrieve function object");
      return 0;
    }

  npy_intp dims = static_cast<npy_intp> (function->outputSize ());
  PyObject* result =
    PyArray_SimpleNew (1, &dims, PyArray_FLOAT32);

  // Try to build an array type from x.
  // All types providing a sequence interface are compatible.
  // Tuples, sequences and Numpy types for instance.
  PyObject* xNumpy =
    PyArray_FromAny (x, 0, 1, 1, NPY_C_CONTIGUOUS, 0);
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
  Eigen::Map<Function::result_t> res
    (static_cast<double*> (PyArray_DATA (result)), function->outputSize ());

  res = (*function) (xEigen);

  std::cout << xEigen << std::endl;
  std::cout << res << std::endl;
  std::cout << static_cast<double*> (PyArray_DATA (result))[0] << std::endl;

  return result;
}


static PyMethodDef RobOptimCoreMethods[] =
  {
    {"Function",  createFunction, METH_VARARGS,
     "Create a Function object."},
    {"compute",  compute, METH_VARARGS,
     "Evaluate a function."},
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
