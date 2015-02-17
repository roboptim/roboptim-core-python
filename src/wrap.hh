#ifndef ROBOPTIM_CORE_PYTHON_WRAP_HH
# define ROBOPTIM_CORE_PYTHON_WRAP_HH

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
#include <roboptim/core/finite-difference-gradient.hh>


#define FORWARD_TYPEDEFS_(X)				\
  ROBOPTIM_DIFFERENTIABLE_FUNCTION_FWD_TYPEDEFS_ (X)

#define FORWARD_TYPEDEFS(X)				\
  ROBOPTIM_DIFFERENTIABLE_FUNCTION_FWD_TYPEDEFS (X)

// Capsule names
static const char* ROBOPTIM_CORE_FUNCTION_CAPSULE_NAME =
  "roboptim_core_function";
static const char* ROBOPTIM_CORE_PROBLEM_CAPSULE_NAME =
  "roboptim_core_problem";
static const char* ROBOPTIM_CORE_SOLVER_CAPSULE_NAME =
  "roboptim_core_solver";
static const char* ROBOPTIM_CORE_SOLVER_CALLBACK_CAPSULE_NAME =
  "roboptim_core_solver_callback";
static const char* ROBOPTIM_CORE_SOLVER_STATE_CAPSULE_NAME =
  "roboptim_core_solver_state";
static const char* ROBOPTIM_CORE_RESULT_CAPSULE_NAME =
  "roboptim_core_result";
static const char* ROBOPTIM_CORE_RESULT_WITH_WARNINGS_CAPSULE_NAME =
  "roboptim_core_result_with_warnings";
static const char* ROBOPTIM_CORE_SOLVER_ERROR_CAPSULE_NAME =
  "roboptim_core_solver_error";


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
                           const std::string& name);

        virtual ~Function ();

        virtual void
	impl_compute (result_t& result, const argument_t& argument) const;

        void setComputeCallback (PyObject* callback);

        PyObject* getComputeCallback () const;

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
                                         const std::string& name);

        virtual ~DifferentiableFunction ();

        size_type inputSize () const;

        size_type outputSize () const;

        virtual void impl_compute (result_t& result, const argument_t& argument)
          const;

        virtual void impl_gradient (gradient_t& gradient,
                                    const argument_t& argument,
                                    size_type functionId)
          const;

        virtual void impl_jacobian (jacobian_t& jacobian,
                                    const argument_t& argument)
          const;


        virtual std::ostream& print (std::ostream& o) const;

        void
	setGradientCallback (PyObject* callback);

        void
	setJacobianCallback (PyObject* callback);

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
                                              const std::string& name);

        virtual ~TwiceDifferentiableFunction ();

        virtual void impl_compute (result_t& result, const argument_t& argument)
          const;


        virtual void impl_gradient
	(gradient_t& gradient, const argument_t& argument, size_type functionId)
          const;


        virtual void
	impl_hessian (hessian_t& /*hessian*/,
		      const argument_t& /*argument*/,
		      size_type /*functionId*/) const;

      private:
        PyObject* hessianCallback_;
      };

      template <typename FdgPolicy>
      class FiniteDifferenceGradient
	: virtual public ::roboptim::GenericFiniteDifferenceGradient
      < ::roboptim::EigenMatrixDense, FdgPolicy>,
        public ::roboptim::core::python::DifferentiableFunction
      {
      public:
        typedef ::roboptim::GenericFiniteDifferenceGradient
	< ::roboptim::EigenMatrixDense, FdgPolicy> fd_t;

        typedef ::roboptim::core::python::Function inPyFunction_t;
        typedef ::roboptim::core::python::DifferentiableFunction outPyFunction_t;

        typedef ::roboptim::DifferentiableFunction outFunction_t;

        FORWARD_TYPEDEFS_ (fd_t);

        explicit FiniteDifferenceGradient (const inPyFunction_t& f,
                                           typename fd_t::value_type e = ::roboptim::finiteDifferenceEpsilon)
          : fd_t (f, e),
	    outFunction_t (f.inputSize (), f.outputSize (), f.getName ()),
	    outPyFunction_t (f.inputSize (), f.outputSize (), f.getName ())
        {
          setComputeCallback (f.getComputeCallback ());
        }

        virtual void impl_gradient (gradient_t& gradient,
                                    const argument_t& argument,
                                    size_type functionId)
          const
	{
	  fd_t::impl_gradient (gradient, argument, functionId);
	}
      };


      template <typename S>
      class SolverCallback
      {
      public:

        /// \brief Solver type.
        typedef S solver_t;

        typedef typename solver_t::callback_t callback_t;
        typedef typename solver_t::problem_t problem_t;
        typedef typename solver_t::solverState_t solverState_t;

        SolverCallback (PyObject* pb);
        virtual ~SolverCallback ();

        void setCallback (PyObject* callback);

        callback_t callback ();

      protected:

        void wrappedCallback (const problem_t& pb, solverState_t& state);

      private:
        PyObject* callback_;
        PyObject* pb_;
      };
    } // end of namespace python.
  } // end of namespace core.
} // end of namespace roboptim.

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
typedef solver_t::solverState_t solverState_t;
typedef roboptim::StateParameter<problem_t::function_t> stateParameter_t;
typedef solverState_t::parameters_t stateParameters_t;

typedef roboptim::finiteDifferenceGradientPolicies::Simple
< ::roboptim::EigenMatrixDense> simplePolicy_t;
typedef roboptim::finiteDifferenceGradientPolicies::FivePointsRule
< ::roboptim::EigenMatrixDense> fivePointsPolicy_t;

namespace detail
{
  namespace rcp = roboptim::core::python;

  template <typename T>
  void destructor (PyObject* obj);

  int functionConverter (PyObject* obj, rcp::Function** address);
  int problemConverter (PyObject* obj, problem_t** address);
  int factoryConverter (PyObject* obj, factory_t** address);
  int solverCallbackConverter (PyObject* obj, rcp::SolverCallback<solver_t>** address);
  int solverStateConverter (PyObject* obj, solverState_t** address);
  int resultConverter (PyObject* obj, result_t** address);
  int resultWithWarningsConverter (PyObject* obj, resultWithWarnings_t** address);
  int solverErrorConverter (PyObject* obj, solverError_t** address);

  struct ParameterValueVisitor;
  struct StateParameterValueVisitor;
  struct null_deleter;
} // end of namespace detail

# include "wrap.hxx"

#endif //! ROBOPTIM_CORE_PYTHON_WRAP_HH
