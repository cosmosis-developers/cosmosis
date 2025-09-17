from .utils import std_weight
import numpy as np





def smooth_density_estimate_1d(samples, xmin, xmax, weights=None, N=1024, smoothing=1, fix_boundary=True):
    """
    Generate a smooth estimate of a 1D PDF from some samples using Kernel Density Estimation,
    correcting at the boundaries.
    
    The boundary fix algorithm comes from Jones & Foster 1996, by way of the getdist code,
    which this implementation follows.  Like that code, it uses convolutions to do the main calculations.
    Without the fix, parameters whose posterior has a hard boundary that cuts off the PDF
    where it is still large will have a spurious decrease in the smoothed PDF.
    
    If the boundary fix is applied, the upper and lower limit are assumed to be
    hard cut-offs in the PDF, not just the range where you happen to want the plots.
    
    Parameters
    ----------
    samples: 1D array
        samples from the PDF
    xmin: float
        lower bound, used for both the choice of output range and the boundary fix (if active)
    xmax: float
        upper bound, ditto
    weights: array
        optiona, default None. Weights per sample
    N: int
        optional, default 1024.  Number of sample points to return.
        Much faster if a power of 2
    smoothing: float/int
        optional, default 1.  Widening of the kernel, to smooth more
    fix_boundary: bool
        optional, default True.  Apply the fix to the boundaries
    """
    from scipy.signal import fftconvolve

    # Compute the things we need for the bandwidth choice:
    if weights is None:
        neff = len(samples)
        stdev = samples.std()
    else:
        neff = weights.sum() ** 2 / (weights**2).sum()
        stdev = std_weight(samples, weights)

    # This is a standard factor for 1D KDEs
    scott_factor = neff**(-0.2)

    # We optionally allow additional smoothing specified by the user
    width = stdev * scott_factor * smoothing

    # First we get the raw histogram that forms the basis of
    # the rest of this.
    edges =  np.linspace(xmin, xmax, N+1)
    x = 0.5 * (edges[1:] + edges[:-1])
    P_histogram, _ = np.histogram(samples, bins=edges, weights=weights)
    w = (xmax - xmin) / N

    # smoothing scale in units of the bin width
    s = width / w

    # Make the Gaussian kernel with which we are convolving.
    # We go out to 3 sigma
    window_width = int(3 * s)
    window_x = np.arange(-window_width, window_width+1)
    kernel = np.exp(-0.5 * (window_x/s)**2)
    
    # Generate the smoothed version.  If we do not need
    # the boundary smoothing then this is our final output
    P_smooth = fftconvolve(P_histogram, kernel, 'same')
    
    # If so, normalize and return it here
    if not fix_boundary:
        P_smooth /= P_smooth.sum() * (x[1] - x[0])        
        return x, P_smooth

    # Generate the mask, a top-hat which cuts off where the
    # boundaries are
    full_width = N + 2 * window_width
    mask = np.ones(full_width)
    mask[:window_width] = 0
    mask[window_width] = 0.5
    mask[-window_width:] = 0
    mask[-(window_width+1)] = 0.5

    # Apply the correction, which uses
    # modified versions of the convolution kernel to
    # get better estimates of the edge region
    
    a0 = fftconvolve(mask, kernel, 'valid')
    
    # Avoid a divide-by-zero
    ix = np.nonzero(a0 * P_smooth)
    a0 = a0[ix]
    
    # Main correction calculation
    P_norm = P_smooth[ix] / a0
    xK = window_x * kernel
    x2K = xK * window_x
    a1 = fftconvolve(mask, xK, mode='valid')[ix]
    a2 = fftconvolve(mask, x2K, mode='valid')[ix]
    xP = fftconvolve(P_histogram, xK, mode='same')[ix]

    # Apply the correction, which is done inside the exponential
    # to keep everything positive.
    scaling = (P_smooth[ix] * a2 - xP * a1) / (a0 * a2 - a1 ** 2)
    P_final = P_smooth.copy()
    P_final[ix] = P_norm * np.exp(np.minimum(scaling / P_norm, 4) - 1)

    # Normalize and return
    P_final /= P_final.sum() * (x[1] - x[0])
    return x, P_final

def smooth_density_estimate_2d(x, y, xmin, xmax, ymin, ymax, weights=None, N=256, smoothing=1, fix_boundary=True):
    """
    Generate a smooth estimate of a 2D PDF from some samples using Kernel Density Estimation,
    
    The boundary fix algorithm comes from Jones & Foster 1996, by way of the getdist code,
    which this implementation follows.  Like that code, it uses convolutions to do the main calculations.
    Without the fix, parameters whose posterior has a hard boundary that cuts off the PDF
    where it is still large will have a spurious decrease in the smoothed PDF.
    
    If the boundary fix is applied, the upper and lower limit are assumed to be
    hard cut-offs in the PDF, not just the range where you happen to want the plots.
    
    Parameters
    ----------
    x: 1D array
        samples from one parameter of the PDF
    y: 1D array, corresponding samples from a second parameter
    xmin: float
        lower bound for x used for both the choice of output range and the boundary fix (if active)
    xmax: float
        upper bound, ditto
    ymin: float
        lower bound for y used for both the choice of output range and the boundary fix (if active)
    ymax: float
        upper bound, ditto
    weights: array
        optiona, default None. Weights per sample
    N: int
        optional, default 256.  Min number of sample points to return  per dimension.
        Much faster if a power of 2
    smoothing: float/int
        optional, default 1.  Widening of the kernel, to smooth more
    fix_boundary: bool
        optional, default True.  Apply the fix to the boundaries
    """
    from scipy.signal import fftconvolve

    # Compute the things we need for the bandwidth choice:
    if weights is None:
        neff = len(x)
        covmat = np.cov([x, y])
    else:
        neff = weights.sum() ** 2 / (weights**2).sum()
        covmat = np.cov([x, y], aweights=weights)

    # make the correlation matrix from the covariance
    corr = covmat.copy()
    for i in range(2):
        si = corr[i, i]**0.5
        corr[i, :] /= si
        corr[:, i] /= si
    rho = corr[0, 1]
    
    if rho > 0.6:
        N = max(512, N)
        
    # This is a standard factor for 1D KDEs
    scott_factor = neff**(-1/6.)

    # First we get the raw histogram that forms the basis of
    # the rest of this.
    xedges = np.linspace(xmin, xmax, N+1)
    yedges = np.linspace(ymin, ymax, N+1)
    P_histogram, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=weights)
    xmid = 0.5 * (xedges[1:] + xedges[:-1])
    ymid = 0.5 * (yedges[1:] + yedges[:-1])
    
    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N

    # smoothing scale in units of the bin width
    width_x = np.sqrt(covmat[0, 0]) * scott_factor * smoothing / dx
    width_y = np.sqrt(covmat[1, 1]) * scott_factor * smoothing / dy
    
    # get the smoothing kernel
    kernel_C = np.array([[width_x**2, width_x * width_y * rho], [width_x * width_y * rho, width_y**2]])
    kernel_Cinv = np.linalg.inv(kernel_C)

    # Make the Gaussian kernel with which we are convolving.
    # We go out to 3 sigma
    window_width = int(3 * max(width_x, width_y))
    
    window_xy = np.mgrid[-window_width:window_width + 1, -window_width:window_width + 1]
    r2 = np.einsum('kij,kl,lij->ij', window_xy, kernel_Cinv, window_xy)
    kernel = np.exp(-0.5*r2)
    kernel /= kernel.sum()
    
    # Generate the smoothed version.  If we do not need
    # the boundary smoothing then this is our final output
    P_smooth = fftconvolve(P_histogram, kernel, 'same')
    
    # If so, normalize and return it here
    if not fix_boundary:
        P_smooth /= P_smooth.sum() * dx * dy
        return xmid, ymid, P_smooth

    # Generate the mask, a top-hat which cuts off where the
    # boundaries are
    full_width = N + 2 * window_width
    mask1d = make_1d_mask(N, window_width)
    mask = np.outer(mask1d, mask1d)
    
    # Apply the correction, which uses
    # modified versions of the convolution kernel to
    # get better estimates of the edge PDF
    a00 = fftconvolve(mask, kernel, 'valid')
    
    # Avoid a divide-by-zero
    nz = (a00 * P_smooth) > P_smooth.max() * 1e-8
    a00 = a00[nz]
    
    # Main correction calculation
    P_norm = P_smooth[nz] / a00
    xK = window_xy[0] * kernel
    x2K = window_xy[0] * xK
    yK = window_xy[1] * kernel
    y2K = window_xy[1] * yK
    xyK = window_xy[1] * xK
    a10 = fftconvolve(mask, xK, mode='valid')[nz]
    a01 = fftconvolve(mask, yK, mode='valid')[nz]
    a11 = fftconvolve(mask, xyK, mode='valid')[nz]
    a20 = fftconvolve(mask, x2K, mode='valid')[nz]
    a02 = fftconvolve(mask, y2K, mode='valid')[nz]
    
    xP = fftconvolve(P_histogram, xK, mode='same')[nz]
    yP = fftconvolve(P_histogram, yK, mode='same')[nz]

    # Apply the correction, which is done inside the exponential
    # to keep everything positive.
    denom = (a20 * a01 ** 2 + a10 ** 2 * a02 - a00 * a02 * a20 + a11 ** 2 * a00 - 2 * a01 * a10 * a11)
    A = a11 ** 2 - a02 * a20
    Ax = a10 * a02 - a01 * a11
    Ay = a01 * a20 - a10 * a11
    scaling = (P_smooth[nz] * A + xP * Ax + yP * Ay) / denom
    
    P_final = P_smooth.copy()
    P_final[nz] = P_norm * np.exp(np.minimum(scaling / P_norm, 4) - 1)

    # Normalize and return
    P_final /= P_final.sum() * dx * dy
    return xmid, ymid, P_final



def make_1d_mask(N, window_width):
    full_width = N + 2 * window_width
    mask = np.ones(full_width)
    mask[:window_width] = 0
    mask[window_width] = 0.5
    mask[-window_width:] = 0
    mask[-(window_width+1)] = 0.5
    return mask
