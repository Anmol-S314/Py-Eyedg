#!python
import numpy as np
import numpy as np
from scipy.signal import ellip, lfilter
import numpy as _np
from scipy.interpolate import interp1d as _interp1d
#from brescount import bres_curve_count as _bres_curve_count
import matplotlib.pyplot as plt


__all__ = ['grid_count']



use_fast = True
try:
    from brescount import bres_curve_count
except ImportError:
    print ("The cython version of the curve counter is not available")
    use_fast = False


def bres_segment_count_slow(x0, y0, x1, y1, grid):
    """Bresenham's algorithm.

    The value of grid[x,y] is incremented for each x,y
    in the line from (x0,y0) up to but not including (x1, y1).
    """

    nrows, ncols = grid.shape

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 0
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    sy = 0
    if y0 < y1:
        sy = 1
    else:
        sy = -1

    err = dx - dy

    while True:
        # Note: this test is moved before setting
        # the value, so we don't set the last point.
        if x0 == x1 and y0 == y1:
            break

        if 0 <= x0 < nrows and 0 <= y0 < ncols:
            grid[x0, y0] += 1

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def bres_curve_count_slow(x, y, grid):
    for k in range(x.size - 1):
        x0 = x[k]
        y0 = y[k]
        x1 = x[k+1]
        y1 = y[k+1]
        if(use_fast == True):
            bres_segment_count(x0, y0, x1, y1, grid)
        else:
            bres_segment_count_slow(x0, y0, x1, y1, grid)


def grid_count(y, window_size, offset, size=None, fuzz=True, bounds=None):
    """
    Parameters
    ----------
    `y` is the 1-d array of signal samples.

    `window_size` is the number of samples to show horizontally in the
    eye diagram.  Typically this is twice the number of samples in a
    "symbol" (i.e. in a data bit).

    `offset` is the number of initial samples to skip before computing
    the eye diagram.  This allows the overall phase of the diagram to
    be adjusted.

    `size` must be a tuple of two integers.  It sets the size of the
    array of counts, (height, width).  The default is (800, 640).

    `fuzz`: If True, the values in `y` are reinterpolated with a
    random "fuzz factor" before plotting in the eye diagram.  This
    reduces an aliasing-like effect that arises with the use of
    Bresenham's algorithm.

    `bounds` must be a tuple of two floating point values, (ymin, ymax).
    These set the y range of the returned array.  If not given, the
    bounds are `(y.min() - 0.05*A, y.max() + 0.05*A)`, where `A` is
    `y.max() - y.min()`.

    Return Value
    ------------
    Returns a numpy array of integers.

    """
    if size is None:
        size = (800, 640)
    height, width = size
    dt = width / window_size
    counts = _np.zeros((width, height), dtype=_np.int32)

    if bounds is None:
        ymin = y.min()
        ymax = y.max()
        yamp = ymax - ymin
        ymin = ymin - 0.05*yamp
        ymax = ymax + 0.05*yamp
    else:
        ymin, ymax = bounds

    start = offset
    while start + window_size < len(y):
        end = start + window_size
        yy = y[start:end+1]
        k = _np.arange(len(yy))
        xx = dt*k
        if fuzz:
            f = _interp1d(xx, yy, kind='cubic')
            jiggle = dt*(_np.random.beta(a=3, b=3, size=len(xx)-2) - 0.5)
            xx[1:-1] += jiggle
            yd = f(xx)
        else:
            yd = yy
        iyd = (height * (yd - ymin)/(ymax - ymin)).astype(_np.int32)
        if(use_fast == True):
            bres_curve_count(xx.astype(_np.int32), iyd, counts)
        else:
            bres_curve_count_slow(xx.astype(_np.int32), iyd, counts)


        start = end
    return counts



def demo_data(num_symbols, samples_per_symbol):
    """
    Generate some data for demonstrations.

    `num_symbols` is the number of symbols (i.e. bits) to include
    in the data stream.

    `samples_per_symbol` is the number of samples per symbol.

    The total length of the result is `num_symbols` * `samples_per_symbol`.

    """
    # A random stream of "symbols" (i.e. bits)
    bits = np.random.randint(0, 2, size=num_symbols)

    # Upsample the bit stream.
    sig = np.repeat(bits, samples_per_symbol)

    # Convert the edges of the symbols to ramps.
    r = min(5, samples_per_symbol // 2)
    # print(sig)
    sig = np.convolve(sig, [1./r]*r, mode='same')
    # print(sig)

    # Add some noise and pass the signal through a lowpass filter.
    b, a = ellip(4, 0.087, 30, 0.15)
    y = lfilter(b, a, sig + 0.075*np.random.randn(len(sig)))
    # print(y)

    return y


def eyediagram(y, window_size, period=0, offset=0, colorbar=True, **imshowkwargs):
    """
    Plot an eye diagram using matplotlib by creating an image and calling
    the `imshow` function.
    """
    counts = grid_count(y, window_size, offset)
    counts = counts.astype(_np.float32)
    counts[counts == 0] = np.nan
    ymax = y.max()
    ymin = y.min()
    yamp = ymax - ymin
    min_y = ymin - 0.05*yamp
    max_y = ymax + 0.05*yamp
    x = counts.T[::-1, :]  # array-like or PIL image
    extent = [-period, period, min_y, max_y]  # floats (left, right, bottom, top)
    plt.imshow(X=x, extent=extent, aspect='auto',**imshowkwargs)
    ax = plt.gca()
    ax.set_facecolor('k')
    plt.grid(color='w')
    if colorbar:
        plt.colorbar()

if __name__ == "__main__":
    
    num_symbols = 5000
    samples_per_symbol = 24
    y = demo_data(num_symbols, samples_per_symbol)
    eyediagram(y, 2*samples_per_symbol,  period=40, offset=16, cmap=plt.cm.coolwarm)

    plt.show()