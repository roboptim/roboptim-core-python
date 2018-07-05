#######################################################################
#                            Python module                            #
#######################################################################

# Use adequate Python executable for tests
SET(TEST_COMMAND "${PYTHON_EXECUTABLE}")

MACRO(REGISTER_TEST NAME)
  ADD_TEST(${NAME} "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.py")
  # Update PYTHONPATH to find the Python module
  SET_TESTS_PROPERTIES(${NAME} PROPERTIES ENVIRONMENT
    "PYTHONPATH=${CMAKE_BINARY_DIR}/src")
ENDMACRO()

# Check roboptim-py internal API.
REGISTER_TEST(wrap)

# Check high-level API.
REGISTER_TEST(function)

# Test some actual problems
REGISTER_TEST(schittkowski)

# Check solver callback support.
REGISTER_TEST(solver_callback)

# Check finite-differences support.
REGISTER_TEST(finite_differences)

# Check cached function support.
REGISTER_TEST(cached_function)

# Check function pool support.
REGISTER_TEST(function_pool)

# Check plotting support.
REGISTER_TEST(plot)

# Check numpy/roboptim matrix translation.
REGISTER_TEST(mpc)

#######################################################################
#                             C++ library                             #
#######################################################################

# Make Boost.Test generates the main function in test cases.
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK -DBOOST_TEST_MAIN)

# To locate built plug-ins.
ADD_DEFINITIONS(-DPLUGIN_PATH="${ROBOPTIM_CORE_PLUGINDIR}")

# To locate test data.
ADD_DEFINITIONS(-DTESTS_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# ROBOPTIM_CORE_PYTHON_TEST(NAME)
# ------------------------
#
# Define a test named `NAME'.
#
# This macro will create a binary from `NAME.cc', link it against
# relevant libraries and add it to the test suite.
#
MACRO(ROBOPTIM_CORE_PYTHON_TEST NAME)
  ADD_EXECUTABLE(${NAME} ${NAME}.cc)
  ADD_TEST(NAME ${NAME}
    COMMAND ${CHECK_PREFIX} ${CMAKE_CURRENT_BINARY_DIR}/${NAME})

  PKG_CONFIG_USE_DEPENDENCY(${NAME} roboptim-core)
  TARGET_LINK_LIBRARIES(${NAME} roboptim-core-python)

  # Link against Boost.
  TARGET_LINK_LIBRARIES(${NAME} ${Boost_LIBRARIES})
ENDMACRO(ROBOPTIM_CORE_PYTHON_TEST)

ROBOPTIM_CORE_PYTHON_TEST(to-python)
