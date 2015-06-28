// Copyright (C) 2015 by Benjamin Chrétien, CNRS-LIRMM.
//
// This file is part of roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <stdexcept>

#include <boost/format.hpp>

#include "common.hh"

#include <frameobject.h>

namespace roboptim
{
  namespace python
  {
    std::string toString (PyObject* obj)
    {
      // string
      if (PyString_Check (obj))
      {
        return PyString_AsString (obj);
      }
      // unicode string
      else if (PyUnicode_Check (obj))
      {
        return PyString_AsString (PyUnicode_AsUTF8String (obj));
      }
      // object
      else
      {
        PyObject* pyStr = PyObject_Str (obj);
        std::string str = PyString_AsString (pyStr);
        Py_DECREF (pyStr);
        return str;
      }
    }

    void checkPythonError ()
    {
      // Catch error (if any)
      // TODO: improve the trace
      PyObject *ptype, *pvalue, *ptraceback;
      PyErr_Fetch (&ptype, &pvalue, &ptraceback);
      if (ptype)
      {
        std::string strErrorMessage = toString (pvalue);
        std::string strTraceback = "";

        PyThreadState* tstate = PyThreadState_GET ();
        if (tstate && tstate->frame)
        {
          PyFrameObject* frame = tstate->frame;

          strTraceback += "Python stack trace:\n";
          while (frame)
          {
            int line = frame->f_lineno;
            const char* filename = PyString_AsString (frame->f_code->co_filename);
            const char* funcname = PyString_AsString (frame->f_code->co_name);
            strTraceback += (boost::format ("    %1%(%2%): %3%\n")
                             % filename % line % funcname).str ();
            frame = frame->f_back;
          }
        }

        throw std::runtime_error
          ((boost::format ("Error occurred in Python code: %1%\n%2%")
            % strErrorMessage % strTraceback).str ());
      }
    }
  } // end of namespace python
} // end of namespace roboptim
