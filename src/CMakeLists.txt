INCLUDE_DIRECTORIES(SYSTEM ${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(SYSTEM ${NUMPY_INCLUDE_DIRS})

#######################################################################
#                             C++ library                             #
#######################################################################

# Compile library
ADD_LIBRARY(roboptim-core-python SHARED
  ${HEADERS}
  to-python.cc)

TARGET_LINK_LIBRARIES(roboptim-core-python ${PYTHON_LIBRARIES})
SET_TARGET_PROPERTIES(roboptim-core-python
  PROPERTIES SOVERSION 3 VERSION 3.2.0)
INSTALL(TARGETS roboptim-core-python DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Add library to pkg-config file.
PKG_CONFIG_APPEND_LIBS(roboptim-core-python)

#######################################################################
#                            Python module                            #
#######################################################################

SET(LOCAL_MODULE_DIR "${PROJECT_BINARY_DIR}/src")

# Compile module
ADD_LIBRARY(wrap MODULE common.cc wrap.cc)
PKG_CONFIG_USE_DEPENDENCY(wrap roboptim-core)
SET_TARGET_PROPERTIES(wrap PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${LOCAL_MODULE_DIR}/roboptim/core
  SOVERSION 3 VERSION 3.2.0)

# Link against Boost.
TARGET_LINK_LIBRARIES(wrap ${Boost_LIBRARIES})

SET_TARGET_PROPERTIES(wrap PROPERTIES PREFIX "")

# Install libraries
INSTALL(TARGETS wrap DESTINATION ${PYTHON_SITELIB}/roboptim/core)

PYTHON_INSTALL_ON_SITE(roboptim __init__.py)
PYTHON_INSTALL_ON_SITE(roboptim/core __init__.py)
PYTHON_INSTALL_ON_SITE(roboptim/core/visualization __init__.py)
PYTHON_INSTALL_ON_SITE(roboptim/core/visualization plotter.py)
