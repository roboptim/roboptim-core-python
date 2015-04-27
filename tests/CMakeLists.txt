# Use adequate Python executable for tests
SET(TEST_COMMAND "${PYTHON_EXECUTABLE}")

# Check roboptim-py internal API.
ADD_TEST(wrap "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/wrap.py")

# Check high-level API.
ADD_TEST(function
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/function.py")

# Test some actual problems
ADD_TEST(schittkowski
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/schittkowski.py")

# Check solver callback support.
ADD_TEST(solver_callback
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/solver_callback.py")

# Check finite-differences support.
ADD_TEST(finite_differences
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/finite_differences.py")

# Check cached function support.
ADD_TEST(cached_function
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/cached_function.py")

# Check function pool support.
ADD_TEST(function_pool
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/function_pool.py")

# Check plotting support.
ADD_TEST(plot
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/plot.py")
