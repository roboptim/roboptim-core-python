#!/usr/bin/env python

import roboptim.core
import numpy

f = roboptim.core.Function (1, 1, "test function")

x = [42.,]
result = roboptim.core.compute (f, x)
print (type (result))
print (result)
print (result[0])
