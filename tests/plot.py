#!/usr/bin/env python
from __future__ import \
    print_function, unicode_literals, absolute_import, division

import matplotlib
# Prevent any image from being displayed during testing:
matplotlib.use('Template')

import unittest
import numpy, numpy.testing

import roboptim.core
from roboptim.core.visualization import PlotStyle2D ,Plotter2D, PlotStyle3D, Plotter3D

class F(roboptim.core.PyFunction):
    def __init__ (self):
        roboptim.core.PyFunction.__init__ (self, 2, 1, "dummy function")

    def impl_compute (self, result, x):
        result[0] = x[0] * x[1]

class TestPlot(unittest.TestCase):

    def test_2d_plot(self):
        f = F()
        plotter = Plotter2D([-10,10],[-10,10])
        plotter.x_res = 10
        plotter.y_res = 10
        plotter.plot(f, plot_style=PlotStyle2D.Contour, label=True)
        plotter.plot(f, plot_style=PlotStyle2D.Contourf)
        plotter.plot(f, plot_style=PlotStyle2D.PColorMesh)
        plotter.add_marker([0], [0], color="black", marker="x",
                           markersize=10, markeredgewidth=2)
        plotter.show()

    def test_3d_plot(self):
        f = F()
        plotter = Plotter3D([-10,10],[-10,10])
        plotter.x_res = 10
        plotter.y_res = 10
        plotter.plot(f, plot_style=PlotStyle3D.Contour)
        plotter.plot(f, plot_style=PlotStyle3D.Contourf)
        plotter.plot(f, plot_style=PlotStyle3D.Wireframe)
        plotter.plot(f, plot_style=PlotStyle3D.Triangle)
        plotter.add_marker([0], [0], color="black", marker="x",
                           markersize=10, markeredgewidth=2)
        plotter.show()

if __name__ == '__main__':
    unittest.main()
