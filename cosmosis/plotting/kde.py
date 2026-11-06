import scipy.stats
import numpy as np
from packaging.version import Version

try:
    parent_class = scipy.stats.gaussian_kde
except AttributeError:
    # older version
    parent_class = scipy.stats.kde.gaussian_kde


class KDE(parent_class):
    def __init__(self, points, factor=1.0, weights=None):

        if Version(scipy.__version__) <= Version('1.1.0'):
            raise ValueError("The KDE class for 2D plotting is not available for this version of scipy.  Please upgrade to scipy 1.1 or later.")

        points = np.array(np.atleast_2d(points))

        self._factor = factor
        d, n = points.shape
        self.norms = []
        normalized_points = []

        for column in points:
            col_mean = column.mean()
            col_std = column.std()
            self.norms.append((col_mean,col_std))
            normalized_points.append((column-col_mean)/col_std)
        super(KDE,self).__init__(normalized_points, weights=weights)

    def covariance_factor(self):
        return self.scotts_factor() * self._factor

    def grid_evaluate(self, n, ranges):
        if isinstance(ranges,tuple):
            ranges = [ranges]
        slices = [slice(xmin,xmax,n*1j) for (xmin,xmax) in ranges]
        grids = np.mgrid[slices]
        axes = [ax.squeeze() for ax in np.ogrid[slices]]
        flats = [(grid.flatten()-norm[0])/norm[1]
                    for (grid,norm) in zip(grids,self.norms)]

        shape = grids[0].shape
        flats = np.array(flats)
        like_flat = self.evaluate(flats)
        like = like_flat.reshape(*shape)
        if len(axes)==1:
            axes = axes[0]
        return axes,like

    def normalize_and_evaluate(self, points):
        points = np.array([(p-norm[0])/norm[1] for norm, p in zip(self.norms, points)])
        return self.evaluate(points)
