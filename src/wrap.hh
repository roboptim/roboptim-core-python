#ifndef ROBOPTIM_CORE_PYTHON_WRAP_HH
# define ROBOPTIM_CORE_PYTHON_WRAP_HH

#include <vector>

#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/make_shared.hpp>

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
#include <roboptim/core/function-pool.hh>
#include <roboptim/core/optimization-logger.hh>
#include <roboptim/core/finite-difference-gradient.hh>

#include <roboptim/core/callback/multiplexer.hh>

#include <roboptim/core/decorator/cached-function.hh>

#include <roboptim/core/detail/utility.hh>


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
static const char* ROBOPTIM_CORE_CALLBACK_MULTIPLEXER_CAPSULE_NAME =
  "roboptim_core_callback_multiplexer";
static const char* ROBOPTIM_CORE_SOLVER_STATE_CAPSULE_NAME =
  "roboptim_core_solver_state";
static const char* ROBOPTIM_CORE_OPTIMIZATION_LOGGER_CAPSULE_NAME =
  "roboptim_core_optimization_logger";
static const char* ROBOPTIM_CORE_RESULT_CAPSULE_NAME =
  "roboptim_core_result";
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
	impl_compute (result_ref result, const_argument_ref argument) const;

        void setComputeCallback (PyObject* callback);

        PyObject* getComputeCallback () const;

        ROBOPTIM_DEFINE_FLAG_TYPE();
        virtual flag_t getFlags() const
        {
          return flags;
        }
        static const flag_t flags = ::roboptim::Function::flags;

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

        virtual void impl_compute (result_ref result, const_argument_ref argument)
          const;

        virtual void impl_gradient (gradient_ref gradient,
                                    const_argument_ref argument,
                                    size_type functionId)
          const;

        virtual void impl_jacobian (jacobian_ref jacobian,
                                    const_argument_ref argument)
          const;


        virtual std::ostream& print (std::ostream& o) const;

        const std::string& getName () const;

        void
	setGradientCallback (PyObject* callback);

        void
	setJacobianCallback (PyObject* callback);

        ROBOPTIM_DEFINE_FLAG_TYPE();
        virtual flag_t getFlags() const
        {
          return flags;
        }
        static const flag_t flags = ::roboptim::DifferentiableFunction::flags;

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

        virtual void impl_compute (result_ref result, const_argument_ref argument)
          const;


        virtual void impl_gradient
	(gradient_ref gradient, const_argument_ref argument, size_type functionId)
          const;


        virtual void
	impl_hessian (hessian_ref /*hessian*/,
		      const_argument_ref /*argument*/,
		      size_type /*functionId*/) const;

        ROBOPTIM_DEFINE_FLAG_TYPE();
        virtual flag_t getFlags() const
        {
          return flags;
        }
        static const flag_t flags = ::roboptim::TwiceDifferentiableFunction::flags;

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

        virtual ~FiniteDifferenceGradient () {}

        virtual void impl_gradient (gradient_ref gradient,
                                    const_argument_ref argument,
                                    size_type functionId)
          const
	{
	  fd_t::impl_gradient (gradient, argument, functionId);
	}

        virtual void impl_jacobian (jacobian_ref jacobian,
                                    const_argument_ref argument)
          const
	{
	  fd_t::impl_jacobian (jacobian, argument);
	}
      };

      class CachedFunction
	: virtual public ::roboptim::DifferentiableFunction,
	  public ::roboptim::core::python::DifferentiableFunction
      {
      public:
        typedef ::roboptim::DifferentiableFunction function_t;
        typedef DifferentiableFunction pyFunction_t;

        typedef ::roboptim::CachedFunction<function_t> cache_t;

        FORWARD_TYPEDEFS_ (cache_t);

        explicit CachedFunction (boost::shared_ptr<pyFunction_t> f, size_t cache_size);

        virtual ~CachedFunction ();

        virtual void impl_compute (result_ref result,
                                   const_argument_ref argument)
          const;

        virtual void impl_gradient (gradient_ref gradient,
                                    const_argument_ref argument,
                                    size_type functionId)
          const;

        virtual void impl_jacobian (jacobian_ref jacobian,
                                    const_argument_ref argument)
          const;

        virtual std::ostream& print (std::ostream& o) const;

      private:
	boost::shared_ptr<cache_t> cached_f_;
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

      /// \brief Iteration callback multiplexer.
      /// \tparam S solver type.
      template <typename S>
      class Multiplexer
      {
      public:
        /// \brief Solver type.
        typedef S solver_t;

        typedef roboptim::SolverFactory<solver_t> factory_t;
        typedef boost::shared_ptr<factory_t> factory_ptr;

        typedef SolverCallback<solver_t> callbackWrapper_t;
        typedef roboptim::OptimizationLogger<solver_t> logger_t;

        // TODO: do not treat logger separately
        /// \brief Allowed types for callbacks:
        ///   - Python callback
        ///   - Optimization logger
        typedef boost::mpl::vector<callbackWrapper_t, logger_t> callback_t;
        typedef typename roboptim::detail::shared_ptr_variant<callback_t>::type callback_ptr;

        typedef std::vector<callback_ptr> callbacks_t;

        typedef roboptim::callback::Multiplexer<solver_t> multiplexer_t;
        typedef typename multiplexer_t::solverCallback_t::callback_t
          callbackFunction_t;

      public:
        Multiplexer (factory_ptr factory);
        virtual ~Multiplexer ();

        void add (callback_ptr callback);
        void remove (size_t i);

      private:
        factory_ptr   factory_;
        multiplexer_t multiplexer_;
        callbacks_t   callbacks_;
      };


      class FunctionPool : virtual public ::roboptim::DifferentiableFunction,
			   public ::roboptim::core::python::DifferentiableFunction
      {
      public:
        FORWARD_TYPEDEFS (::roboptim::DifferentiableFunction);

        typedef ::roboptim::core::python::DifferentiableFunction pyFunction_t;
        typedef ::roboptim::DifferentiableFunction function_t;

        typedef ::roboptim::FunctionPool<function_t,
					 boost::mpl::vector<function_t> > pool_t;
        typedef pool_t::callback_t callback_t;
        typedef pool_t::callback_ptr callback_ptr;
        typedef pool_t::functionList_t functionList_t;

        explicit FunctionPool (const callback_ptr callback,
                               const functionList_t& functions,
                               const std::string& name);

        virtual ~FunctionPool ();

        virtual void impl_compute (result_ref result, const_argument_ref x) const;

        virtual void impl_gradient (gradient_ref gradient,
                                    const_argument_ref x,
                                    size_type functionId = 0) const;

        virtual void impl_jacobian (jacobian_ref jacobian,
                                    const_argument_ref x) const;

        virtual std::ostream& print (std::ostream&) const;

      private:
	pool_t pool_;
      };
    } // end of namespace python.
  } // end of namespace core.
} // end of namespace roboptim.

typedef roboptim::Problem< ::roboptim::EigenMatrixDense> problem_t;
typedef roboptim::Solver< ::roboptim::EigenMatrixDense> solver_t;

typedef roboptim::SolverFactory<solver_t> factory_t;
typedef roboptim::OptimizationLogger<solver_t> logger_t;
typedef roboptim::callback::Multiplexer<solver_t> multiplexer_t;

typedef roboptim::Result result_t;
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
  int functionListConverter (PyObject* obj, rcp::FunctionPool::functionList_t** address);
  int problemConverter (PyObject* obj, problem_t** address);
  int factoryConverter (PyObject* obj, factory_t** address);
  int solverCallbackConverter (PyObject* obj, rcp::SolverCallback<solver_t>** address);
  int multiplexerConverter (PyObject* obj, rcp::Multiplexer<solver_t>** address);
  int solverStateConverter (PyObject* obj, solverState_t** address);
  int resultConverter (PyObject* obj, result_t** address);
  int solverErrorConverter (PyObject* obj, solverError_t** address);

  struct ParameterValueVisitor;
  struct StateParameterValueVisitor;
  struct null_deleter;
  struct pyobject_deleter;

  template <typename T>
  boost::shared_ptr<T> to_shared_ptr (T* o, PyObject* py_o);
} // end of namespace detail

# include "wrap.hxx"

#endif //! ROBOPTIM_CORE_PYTHON_WRAP_HH
