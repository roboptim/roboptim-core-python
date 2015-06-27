#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import copy

import roboptim.core
from roboptim.core.visualization import Plotter2D, PlotStyle2D

from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np, numpy.testing

import unittest
import math
from copy import copy

class IterCallback (roboptim.core.PySolverCallback):
    def __init__ (self, pb):
        roboptim.core.PySolverCallback.__init__ (self, pb)
        self.x = list()

    def callback (self, pb, state):
        self.x.append(np.empty_like(state.x))
        self.x[-1][:] = state.x

class Problem16_Cost (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "100 (x₁ - x₀²)² + (1 - x₀)²")

    def impl_compute (self, result, x):
        result[0] = 100. * (x[1] - x[0]**2)**2 + (1. - x[0])**2

    def impl_gradient (self, result, x, functionId):
        result[0] = 400. * (x[0] ** 3 - x[0] * x[1]) + 2. * x[0] - 2.
        result[1] = 200. * (-x[0]**2 + x[1])

class Problem16_G1 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "x₀ + x₁²")

    def impl_compute (self, result, x):
        result[0] = x[0] + x[1]**2

    def impl_gradient (self, result, x, functionId):
        result[0] = 1.
        result[1] = 2. * x[1]

class Problem16_G2 (roboptim.core.PyDifferentiableFunction):
    def __init__ (self):
        roboptim.core.PyDifferentiableFunction.__init__ \
            (self, 2, 1, "x₀² + x₁")

    def impl_compute (self, result, x):
        result[0] = x[0]**2 + x[1]

    def impl_gradient (self, result, x, functionId):
        result[0] = 2. * x[0]
        result[1] = 1.

class TestFunctionPy(unittest.TestCase):

    def test_problem_16(self):
        """
        Schittkowski problem #16
        """
        cost = Problem16_Cost ()
        problem = roboptim.core.PyProblem (cost)
        problem.startingPoint = numpy.array([-2, 1., ])
        problem.argumentBounds = numpy.array([[-2., 0.5],
                                              [-float("inf"), 1.]])

        g1 = Problem16_G1 ()
        problem.addConstraint (g1, [0., float("inf"),])
        g2 = Problem16_G2 ()
        problem.addConstraint (g2, [0., float("inf"),])

        # Check starting value
        numpy.testing.assert_almost_equal (cost (problem.startingPoint)[0], 909.)

        # Initialize callback
        callback = IterCallback (problem)

        # Let the test fail if the solver does not exist.
        try:
            # Create solver
            log_dir = "/tmp/roboptim-core-python/problem_16"
            solver = roboptim.core.PySolver ("ipopt", problem, log_dir = log_dir)

            # Add callback
            solver.addIterationCallback(callback)

            print (solver)
            solver.solve ()
            r = solver.minimum ()
            print (r)

            # Plot results
            plotter = Plotter2D([-2.1,0.6],[0,1.1])
            plotter.x_res = 100
            plotter.y_res = 100
            plotter.plot(cost, plot_style = PlotStyle2D.PColorMesh, vmax=10,
                         norm=LogNorm())

            # Set up a colormap:
            cdict = {'red':   ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),

                     'green': ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),

                     'blue':  ((0.0, 0.0, 0.0),
                               (1.0, 0.0, 0.0)),

                     'alpha': ((0.0, 0.0, 0.0),
                               (1.0, 1.0, 1.0))
                    }
            cstr_cmap = LinearSegmentedColormap('Mask', cdict)
            cstr_cmap.set_under('r', alpha=0)
            cstr_cmap.set_over('w', alpha=0)
            cstr_cmap.set_bad('g', alpha=0)

            plotter.plot(g1, plot_style=PlotStyle2D.Contourf,
                         linewidth=10, alpha=None,
                         cmap=cstr_cmap, vmax=0, fontsize=20)
            plotter.plot(g1, plot_style=PlotStyle2D.Contour,
                         linewidth=10, alpha=None, levels=[0],
                         vmax=0, fontsize=20, colors="k")
            plotter.plot(g2, plot_style=PlotStyle2D.Contourf,
                         linewidth=10, alpha=None,
                         cmap=cstr_cmap, vmax=0)

            # Print iterations
            X = zip(*callback.x)[0]
            Y = zip(*callback.x)[1]
            # Show evolution
            plotter.add_marker(X, Y,
                               color="white", marker=".", markersize=5)
            # First point
            plotter.add_marker(X[0], Y[0],
                               color="white", marker="o", markersize=10, markeredgewidth=2)
            # Final result
            plotter.add_marker(X[-1], Y[-1],
                               color="white", marker="s", markersize=10, markeredgewidth=2)

            # Print actual global minimum
            plotter.add_marker(0.5, 0.25,
                               color="black", marker="x", markersize=14, markeredgewidth=6)
            plotter.add_marker(0.5, 0.25,
                               color="white", marker="x", markersize=10, markeredgewidth=3)

            plotter.show()

        except Exception as e:
            print ("Error: %s" % e)


if __name__ == '__main__':
    unittest.main ()
