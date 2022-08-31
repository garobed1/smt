"""
Author: Dr. John T. Hwang <hwangjt@umich.edu>

This package is distributed under New BSD license.

Multi-dimensional Rosenbrock function.
"""
import numpy as np

from smt.problems.problem import Problem


class Rosenbrock(Problem):
    def _initialize(self):
        self.options.declare("name", "Rosenbrock", types=str)

    def _setup(self):
        self.xlimits[:, 0] = -2.0
        self.xlimits[:, 1] = 2.0

        if(self.options["ndim"] == 2):
            self.mean = 455.6666666666667
            self.stdev = 606.5602418425782

        if(self.options["ndim"] == 4):
            self.mean = 1367.0418194476738
            self.stdev = 1141.4087164117796

        if(self.options["ndim"] == 8):
            self.mean = 3189.825166315156
            self.stdev = 1781.1374067903641

    def _evaluate(self, x, kx):
        """
        Arguments
        ---------
        x : ndarray[ne, nx]
            Evaluation points.
        kx : int or None
            Index of derivative (0-based) to return values with respect to.
            None means return function value rather than derivative.

        Returns
        -------
        ndarray[ne, 1]
            Functions values if kx=None or derivative values if kx is an int.
        """
        ne, nx = x.shape

        y = np.zeros((ne, 1), complex)
        if kx is None:
            for ix in range(nx - 1):
                y[:, 0] += (
                    100.0 * (x[:, ix + 1] - x[:, ix] ** 2) ** 2 + (1 - x[:, ix]) ** 2
                )
        else:
            if kx < nx - 1:
                y[:, 0] += -400.0 * (x[:, kx + 1] - x[:, kx] ** 2) * x[:, kx] - 2 * (
                    1 - x[:, kx]
                )
            if kx > 0:
                y[:, 0] += 200.0 * (x[:, kx] - x[:, kx - 1] ** 2)

        return y
