INCLUDE_DIRECTORIES(SYSTEM ${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(SYSTEM ${NUMPY_INCLUDE_DIRS})

ADD_LIBRARY(core MODULE roboptim.cc)
PKG_CONFIG_USE_DEPENDENCY(core roboptim-core)

SET_TARGET_PROPERTIES(core PROPERTIES PREFIX "")

INSTALL(TARGETS core
  DESTINATION ${PYTHON_SITELIB}/roboptim/)