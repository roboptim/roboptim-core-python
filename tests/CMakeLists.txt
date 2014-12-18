# Update PYTHONPATH to use built files
SET(ENV{PYTHONPATH} "${CMAKE_BINARY_DIR}/src")

SET(TEST_COMMAND "${PYTHON_EXECUTABLE}")

# Check roboptim-py internal API.
ADD_TEST(wrap "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/wrap.py")

# Check high-level API.
ADD_TEST(function
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/function.py")

# Test some actual problems
ADD_TEST(schittkowski
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/schittkowski.py")

# Check finite-differences support.
ADD_TEST(finite_differences
  "${TEST_COMMAND}" "${CMAKE_CURRENT_SOURCE_DIR}/finite_differences.py")
