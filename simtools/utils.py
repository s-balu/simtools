import numpy as np
from sklearn.neighbors import KernelDensity
import tabulate


def recenter_coordinates(position, boxsize):
    if isinstance(boxsize, (float, np.floating, int, np.integer)):
        boxsize = boxsize * np.ones(3)
    position = boxsize/2 - ((boxsize/2 - position) % boxsize)
    return position


def vector_norm(vectors, return_magnitude=True, return_unit_vectors=False):
    vmags = np.sqrt(np.einsum('...i,...i', vectors, vectors))
    if return_magnitude and return_unit_vectors:
        return vmags, vectors / vmags[:, np.newaxis]
    elif return_magnitude:
        return vmags
    elif return_unit_vectors:
        return vectors / vmags[:, np.newaxis]


def simple_derivative(x, y, window_length=1):
    winds = np.unique(np.append(np.arange(0, len(y), window_length), len(y)-1))
    x_ = x[winds]
    derivative = np.diff(y[winds]) / np.diff(x_)
    return np.interp(x[:-1], x_[:-1], derivative)


def kernel_density_estimate(data, kernel, bandwidth, resolution):

    data_ = np.zeros(np.shape(data))

    def rescale(x):
        xmin = np.min(x)
        x_ = x - xmin
        xmax = np.max(x_)
        return x_ / xmax, xmin, xmax

    minmax = []
    for i, d in enumerate(data.T):
        q = rescale(d)
        data_[:, i] = q[0]
        minmax.append(q[1:])
    minmax = np.array(minmax)
    mins, maxs = minmax[:, 0], minmax[:, 1]

    # interpolation grid
    nvar = np.shape(data)[1]
    resolution = [resolution] * nvar if isinstance(resolution, int) else \
        resolution
    scaled_axes = [np.linspace(0, 1, res) for res in resolution]
    grid = np.meshgrid(*scaled_axes, indexing='ij')
    coords = np.vstack(list(map(np.ravel, grid))).T

    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data_)
    log_density = kde.score_samples(coords).reshape(*resolution)
    log_density -= np.sum(np.log(maxs))

    axes = [ax * axmax + axmin for ax, axmax, axmin in zip(
        scaled_axes, maxs, mins)]

    return log_density, axes


def pretty_print(quantities, labels, title):

    info_table_labels = np.array(labels, dtype=object)
    info_table_quantities = np.array(quantities)
    info_table = np.vstack((info_table_labels, info_table_quantities)).T

    print("\n\n\t {}\n".format(title))
    print(tabulate.tabulate(info_table))

    return


def myin1d(a, b, kind=None):
    """
    Returns the indices of a with values that are also in b, in the order that
    those elements appear in b.

    """
    loc = np.in1d(a, b, kind=kind)
    order = a[loc].argsort()[b.argsort().argsort()]
    return np.where(loc)[0][order]


def churazov_smooth(x, y, width=None):

    """
    Smoothing algorithm described in Appendix B of Churazov et al. 2010,
    doi:10.1111/j.1365-2966.2010.16377.x

    """

    def weight(r0, rr, w):
        return np.exp(-np.log10(r0 / rr) ** 2 / (2 * w ** 2))

    def calc_coeffs(r0, rr, yy, w):
        W = weight(r0, rr, w)
        lrr = np.log10(rr)
        lyy = np.log10(yy)
        a1 = np.sum(lrr * W * lyy) * np.sum(W) - \
            np.sum(W * lyy) * np.sum(lrr * W)
        a2 = np.sum(lrr ** 2 * W) * np.sum(W) - np.sum(lrr * W) ** 2

        a = a1 / a2
        b = (np.sum(W * lyy) - a * np.sum(lrr * W)) / np.sum(W)

        return a, b

    lx = np.log10(x)
    if width is None:
        width = lx[1] - lx[0]
    coeffs = [calc_coeffs(xi, x, y, width) for xi in x]

    return np.array([a * lxi + b for (a, b), lxi in zip(coeffs, lx)])
