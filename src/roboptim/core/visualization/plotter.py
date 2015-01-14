from abc import ABCMeta, abstractproperty, abstractmethod

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

class PlotStyle2D(object):
    Contour, PColorMesh = range(2)

class PlotStyle3D(object):
    Contour, Wireframe, Triangle = range(3)

class Plotter(object):
    """
    Plotter abstract class.
    """
    __metaclass__ = ABCMeta

    def __init__(self, x_range, y_range, x_res=10, y_res=10):
        self.x_range = x_range
        self.y_range = y_range
        self.x_res = x_res
        self.y_res = y_res
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection=self.projection())
        self.ax.set_xlabel('x[0]')
        self.ax.set_ylabel('x[1]')

    @abstractmethod
    def plot(self, f, plot_style, cmap=None, *args, **kwargs):
        pass

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

    def projection(self):
        return None

    def plot(self, f, plot_style=PlotStyle2D.PColorMesh, cmap=None, *args, **kwargs):
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
            self.ax.contourf(X, Y, Z, 1, cmap=cmap, *args, **kwargs)
            if 'levels' in kwargs:
                self.ax.contour(X, Y, Z, max(self.x_res, self.y_res),
                                *args, **kwargs)
        # Color mesh plotting
        elif plot_style == PlotStyle2D.PColorMesh:
            mesh = self.ax.pcolormesh(X, Y, Z, cmap=cmap,
                                      *args, **kwargs)
            plt.colorbar(mesh)
        else:
            return


class Plotter3D(Plotter):
    """
    3D plotter class for RobOptim functions.
    """
    def __init__(self, x_range, y_range):
        super(Plotter3D, self).__init__(x_range, y_range)

    def projection(self):
        return "3d"

    def plot(self, f, plot_style=PlotStyle3D.Triangle, cmap=None,
             *args, **kwargs):
        """
        Plot a RobOptim function as a 3D surface.
        """
        x = np.linspace(self.x_range[0], self.x_range[1], self.x_res)
        y = np.linspace(self.y_range[0], self.y_range[1], self.y_res)
        X, Y = np.meshgrid(x, y)

        if cmap is not None:
            if self.isstr(cmap):
                cmap = cm.get_cmap(cmap)

        Z = self.compute_z(f,X,Y)

        # Wireframe plotting
        if plot_style == PlotStyle3D.Wireframe:
            self.ax.plot_wireframe(X, Y, Z, *args, **kwargs)
        # Triangle plotting
        elif plot_style == PlotStyle3D.Triangle:
            self.ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(),
                                 cmap=cmap, *args, **kwargs)
        # Contour plotting
        elif plot_style == PlotStyle3D.Contour:
            self.ax.contourf(X, Y, Z, 1, cmap=cmap, *args, **kwargs)
            if 'levels' in kwargs:
                self.ax.contour(X, Y, Z, max(self.x_res, self.y_res),
                                *args, **kwargs)
        else:
            return
