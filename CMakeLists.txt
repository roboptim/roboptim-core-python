# Copyright 2013, Thomas Moulard, CNRS-AIST JRL
#
# This file is part of roboptim-core-python.
# roboptim-core is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# roboptim-core-python is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Lesser Public License for more details.  You should have
# received a copy of the GNU Lesser General Public License along with
# roboptim-core.  If not, see <http://www.gnu.org/licenses/>.

# Requires at least CMake 2.6 to configure the package.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/python.cmake)

SET(PROJECT_NAME roboptim-core-python)
SET(PROJECT_DESCRIPTION "RobOptim Core Python")
SET(PROJECT_URL "http://github.com/roboptim/roboptim-core-python")

SET(HEADERS
  )

SET(CXX_DISABLE_WERROR 1) #FIXME: disable for now.

SETUP_PROJECT()

# Search for dependencies.
SET(BOOST_COMPONENTS filesystem system thread date_time)
SEARCH_FOR_BOOST()
ADD_REQUIRED_DEPENDENCY("roboptim-core >= 3.1")

# Python 2 and 3 are supported
FINDPYTHON()

# Detect numpy.
EXECUTE_PROCESS(
  COMMAND "${PYTHON_EXECUTABLE}" "-c"
  "import numpy; print (numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIRS
  ERROR_QUIET)

HEADER_INSTALL("${HEADERS}")

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tests)

SETUP_PROJECT_FINALIZE()
