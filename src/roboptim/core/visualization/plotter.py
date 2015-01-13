from abc import ABCMeta, abstractproperty

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class PlotStyle2D:
    Contour, PColorMesh = range(2)

class PlotStyle3D:
    Contour, Wireframe, Triangle = range(3)

class Plotter(object):
    """
    Plotter abstract class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=self.projection())
        self.ax.set_xlabel('x[0]')
        self.ax.set_ylabel('x[1]')

    @abstractproperty
    def projection(self):
        pass

    def add_marker(self, x, y, *args, **kwargs):
        self.ax.plot(x, y, *args, **kwargs)

    def show(self):
        plt.show()

    def compute_z(self,f,X,Y):
        z = np.zeros(X.shape)
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                z[i,j] = f([X[i,j],Y[i,j]])
        return z

    try:
        basestring  # attempt to evaluate basestring
        def isstr(self,s):
            return isinstance(s, basestring)
    except NameError:
        def isstr(self,s):
            return isinstance(s, str)


class Plotter2D(Plotter):
    """
    2D plotter class for RobOptim functions.
    """
    def __init__(self, x_range, y_range):
        super(Plotter2D, self).__init__(x_range, y_range)
        self.x_res = 10
        self.y_res = 10

    def projection(self):
        return None

    def plot(self, f, plot_style=PlotStyle2D.PColorMesh,
             levels=None, vmin=None, vmax=None, linewidth=0.5, alpha=None,
             cmap=None, colors=None):
        """
        Plot a RobOptim function as a 2D surface.
        """
        x = np.linspace(self.x_range[0], self.x_range[1], self.x_res)
        y = np.linspace(self.y_range[0], self.y_range[1], self.y_res)
        X, Y = np.meshgrid(x, y)

        if cmap is not None:
            if self.isstr(cmap):
                cmap = cm.get_cmap(cmap)

        Z = self.compute_z(f,X,Y)

        # Contour plotting
        if plot_style == PlotStyle2D.Contour:
            self.ax.contourf(X, Y, Z, 1,
                             alpha=alpha, cmap=cmap,colors=colors,
                             vmin=vmin, vmax=vmax)
            if levels is not None:
                self.ax.contour(X, Y, Z, max(self.x_res, self.y_res),
                                linewidth=linewidth, colors='k', levels=levels)
        # Color mesh plotting
        elif plot_style == PlotStyle2D.PColorMesh:
            mesh = self.ax.pcolormesh(X, Y, Z, cmap=cmap, edgecolors='face',
                                      vmin=vmin,vmax=vmax)
            plt.colorbar(mesh)
        else:
            return


class Plotter3D(Plotter):
    """
    3D plotter class for RobOptim functions.
    """
    def __init__(self, x_range, y_range):
        super(Plotter3D, self).__init__(x_range, y_range)
        self.x_res = 10
        self.y_res = 10

    def projection(self):
        return "3d"

    def plot(self, f, plot_style=PlotStyle3D.Triangle,
             levels=None, vmin=None, vmax=None, linewidth=0.5, alpha=None,
             cmap=None, colors=None):
        """
        Plot a RobOptim function as a 3D surface.
        """
        x = np.linspace(self.x_range[0], self.x_range[1], self.x_res)
        y = np.linspace(self.y_range[0], self.y_range[1], self.y_res)
        X, Y = np.meshgrid(x, y)

        if cmap is not None:
            if isstr(cmap):
                cmap = cm.get_cmap(cmap)

        Z = self.compute_z(f,X,Y)

        # Wireframe plotting
        if plot_style == PlotStyle3D.Wireframe:
            self.ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        # Triangle plotting
        elif plot_style == PlotStyle3D.Triangle:
            self.ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(),
                                 cmap=cmap, linewidth=linewidth)
        # Contour plotting
        elif plot_style == PlotStyle3D.Contour:
            self.ax.contourf(X, Y, Z, 1,
                             alpha=alpha, cmap=cmap,colors=colors,
                             vmin=vmin, vmax=vmax)
            if levels is not None:
                self.ax.contour(X, Y, Z, max(self.x_res, self.y_res),
                                linewidth=linewidth, colors='k', levels=levels)
        else:
            return
