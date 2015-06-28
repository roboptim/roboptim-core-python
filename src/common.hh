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

#include <Python.h>

// Python 3 support
#if PY_MAJOR_VERSION >= 3
# define PyInt_FromLong      PyLong_FromLong
# define PyInt_AsLong        PyLong_AsLong
# define PyInt_Check         PyLong_Check
# define PyString_FromString PyBytes_FromString
# define PyString_Check      PyBytes_Check
# define PyString_AsString   PyBytes_AsString
#endif //! PY_MAJOR_VERSION

namespace roboptim
{
  namespace python
  {
    std::string toString (PyObject* obj);

    void checkPythonError ();
  } // end of namespace python
} // end of namespace roboptim
