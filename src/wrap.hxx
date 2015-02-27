#ifndef ROBOPTIM_CORE_PYTHON_WRAP_HXX
# define ROBOPTIM_CORE_PYTHON_WRAP_HXX

# include <boost/bind.hpp>

namespace roboptim
{
  namespace core
  {
    namespace python
    {
      namespace
      {
        template <typename S>
        struct MultiplexerCallbackVisitor
          : public boost::static_visitor<typename Multiplexer<S>::callbackFunction_t>
        {
          template <typename U>
          typename Multiplexer<S>::callbackFunction_t operator () (const U& callback) const
          {
            return callback->callback ();
          }
        };
      } // end of unnamed namespace

      template <typename S>
      Multiplexer<S>::Multiplexer (factory_ptr factory)
      : factory_ (factory),
        multiplexer_ ((*factory)()),
        callbacks_ ()
      {
      }

      template <typename S>
      Multiplexer<S>::~Multiplexer ()
      {
        // DECREF is handled in the shared_ptr destructor
      }

      template <typename S>
      void Multiplexer<S>::add (callback_ptr callback)
      {
        callbacks_.push_back (callback);
        multiplexer_.callbacks ().push_back
          (boost::apply_visitor (MultiplexerCallbackVisitor<S> (), callback));
      }

      template <typename S>
      void Multiplexer<S>::remove (size_t i)
      {
        multiplexer_.callbacks ().erase (multiplexer_.callbacks ().begin () + i);
        callbacks_.erase (callbacks_.begin () + i);
      }

      template <typename S>
      SolverCallback<S>::SolverCallback (PyObject* pb)
	: callback_ (0)
      {
        Py_XINCREF (pb);
        pb_ = pb;
      }

      template <typename S>
      SolverCallback<S>::~SolverCallback ()
      {
        if (callback_)
	  {
	    Py_DECREF (callback_);
	    callback_ = 0;
	  }

        if (pb_)
	  {
	    Py_DECREF (pb_);
	    pb_ = 0;
	  }
      }

      template <typename S>
      void SolverCallback<S>::setCallback (PyObject* callback)
      {
        if (callback_)
	  {
	    Py_DECREF (callback_);
	    callback_ = 0;
	  }

        Py_XINCREF (callback);
        callback_ = callback;
      }

      template <typename S>
      typename SolverCallback<S>::callback_t
      SolverCallback<S>::callback ()
      {
        return boost::bind (&SolverCallback<S>::wrappedCallback,
                            this, _1, _2);
      }

      template <typename S>
      void SolverCallback<S>::wrappedCallback (const problem_t& /*pb*/,
                                               solverState_t& state)
      {
        if (!callback_)
	  {
	    PyErr_SetString
	      (PyExc_TypeError,
	       "solver callback not set");
	    return;
	  }

        if (!PyFunction_Check (callback_))
	  {
	    PyErr_SetString
	      (PyExc_TypeError,
	       "solver callback is not a valid callback");
	    return;
	  }

        // TODO: need to make sure that pb_ == pb.
        PyObject* statePy =
          PyCapsule_New (&state, ROBOPTIM_CORE_SOLVER_STATE_CAPSULE_NAME, NULL);

        PyObject* arglist = Py_BuildValue ("(OO)", pb_, statePy);
        if (!arglist)
	  {
	    Py_DECREF (arglist);
	    PyErr_SetString
	      (PyExc_TypeError, "failed to build argument list");
	    return;
	  }

        PyEval_CallObject (callback_, arglist);
        Py_DECREF (arglist);
        Py_XDECREF (statePy);

        return;
      }
    } // end of namespace python
  } // end of namespace core
} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_PYTHON_WRAP_HXX
