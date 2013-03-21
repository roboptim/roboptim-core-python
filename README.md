roboptim-core-python
====================

[![Build Status](https://travis-ci.org/roboptim/roboptim-core-python.png?branch=master)](https://travis-ci.org/roboptim/roboptim-core-python)

This package provides Python bindings for roboptim-core. It relies on
Numpy arrays to provide vectors and matrices which are easy to use.

Note that all vectors and matrices input can be any objects satisfying
the random access interface (i.e. `__getitem__` is provided) while
output types will be Numpy arrays.

In practice, it means that through this package, optimization problems
can be defined and solved directly from Python.

The goal is to match as closely as possible the C++ API while relying
on the dynamic nature of Python to ease problem writing.

See tests/function.py for examples.

For general information about the project, please refer to its
homepage: http://www.roboptim.net/
