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
