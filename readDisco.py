import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import triangle
import triangle.plot
#import scipy.signal as signal
import pandas as pd


import colormaps as cmaps
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

'''
prim[:,0] = rho
prim[:,1] = P
prim[:,2] = u_r
prim[:,3] = u_phi
prim[:,4] = u_z
'''


def loadcheckpoint(filename):

    f = h5.File(filename, "r")

    # In ['Data']['Cells'] there are 6 variables:
    # 5 primitive variables (rho, p, u_r, u_phi, u_z)
    # and the phi for each cell.
    # piph is the list of all phi values for each cell:
    piph = f['Data']['Cells'][:,-1][...]
    # prim contains the five primitive variables:
    prim = f['Data']['Cells'][:,:-1][...]
    # Index gives you the index at which you switch to 
    # a new annulus (increase in r by delta(r)), if you
    # were to count around the cells in each annulus starting
    # from the inner radius:
    index = f['Grid']['Index'][...]
    # idPhi0 tells you the index of which cells is at phi = 0
    idPhi0 = f['Grid']['Id_phi0'][...]
    # nphi is the number of phi zones in each annulus 
    # (Same as taking index[i] - index[i-1])
    nphi = f['Grid']['Np'][...]
    t = f['Grid']['T'][0]
    riph = f['Grid']['r_jph'][...]
    ziph = f['Grid']['z_kph'][...]
    # Planet data:
    # this returns 6 variables for each planet
    # planets[0,:] = [M1,v_r1,omega1,r1,phi1,eps1]
    # planets[1,:] = [M2,v_r2,omega2,r2,phi2,eps2]
    plan = f['Data']['Planets']


    r = np.zeros(piph.shape)
    R = 0.5*(riph[1:] + riph[:-1])
    for i in xrange(index.shape[0]):
        for k in xrange(index.shape[1]):
            ind0 = index[i,k]
            ind1 = ind0 + nphi[i,k]
            r[ind0:ind1] = R[i]


    return t, r, piph, prim, plan


def loadreport(filename):

    t,torque = np.genfromtxt(filename,unpack=True,usecols=(0,1))

    return t, torque


def loadplanet(filename):
    f = h5.File(filename, "r")
    t = f['Grid']['T'][0] /2./np.pi # orbits
    plan = f['Data']['Planets']
    m1 = plan[0,0]
    m2 = plan[1,0]
    r2 = plan[1,3]

    return t, m2, r2


def plot2d(file,savedir):
    t,r,phi,prim,plan = loadcheckpoint(file)

    print t/2./np.pi, ' orb'

    dens = prim[:,0]
    pres = prim[:,1]
    velr = prim[:,2]
    vphi = prim[:,3]
    velz = prim[:,4]
    orb = t/2./np.pi

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    triang = tri.Triangulation(x, y)
    # mask off unwanted traingles
    min_radius = 1.01*np.min(r)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)


    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(11,8))

    im1 = ax1.tricontourf(triang,np.log10(dens),200,cmap=cmaps.inferno)
    f.colorbar(im1, ax=ax1, label=r'$\log{\rho}$')

    im2 = ax2.tricontourf(triang,np.log10(pres),200,cmap=cmaps.magma)
    plt.colorbar(im2, ax=ax2, label=r'$\log{p}$')

    im3 = ax3.tricontourf(triang,vphi,200,cmap=cmaps.plasma)
    f.colorbar(im3, ax=ax3, label=r'$v_{\phi}$')

    im4 = ax4.tricontourf(triang,velr,200,cmap=cmaps.viridis)
    f.colorbar(im4, ax=ax4, label=r'$v_r$')

    plt.tight_layout()

    #plt.savefig('%s2d_%i_orb.png'%(savedir,orb))
    plt.show()



def plottorque(dir,mach,drift,savedir):
    rep_file = dir+'report.dat'
    check_file = dir + 'checkpoint_0039.h5'
    t, torque = loadreport(rep_file)
    t_snap,r,phi,prim,plan = loadcheckpoint(check_file)


    # This doesn't make sense because the torque
    # changes over time, and this is the separation
    # at a single snap shot....
    # Shouldn't we normalize it by the Type I torque
    # throughout the evolution?

    # distance of secondary
    a = plan[1,3]
    # ang velocity of secondary
    om2 = plan[1,2]

    q = plan[1,0]
    sigma_0 = 1.
    # type 1 torque:
    T_0 = q**2 * mach**2 * sigma_0 * a**4 * om2**2

    # cut data from beginning out for smooting
    # provide index to cut out till
    cut = 10000
    t = t[cut:]
    torque = torque[cut:]

    orb = t/2/np.pi

    window = int(orb[-1]-orb[0])/2
    print 'window = ',window
    #poly_order = 3
    #smoothed = savitzky_golay(torque[cut:]/T_0, window, poly_order)
    #smoothed = signal.medfilt(torque[cut:]/T_0,1115)
    smoothed = pd.rolling_mean(torque/T_0,window)

    plt.plot(orb,torque/T_0,color='#81B6B3',alpha=0.9)
    plt.plot(orb,smoothed,color='#025652',alpha=0.9)
    plt.xlabel(r'$t \, \rm [orb]$')
    plt.ylabel(r'$\rm T/T_0$')
    plt.xlim([orb[0],orb[-1]])

    # horizontal dotted line at y=0
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=1, linestyle='--',color='k')

    avg_torq = np.average(smoothed[-1001:-1])

    plt.annotate(r'$\mathcal{M} = %i$'%mach,xy=(500,0.07))
    plt.annotate(r'$\dot{a}/\Omega a = %.1e$'%drift,xy=(500,0.045))
    plt.annotate(r'$T/T_0 = %.2f$'%avg_torq,xy=(500,0.02))

    plt.tight_layout()
    plt.show()

    return orb, torque/T_0, smoothed



def periodogram(x,y):
    import scipy.signal
    f = np.linspace(0.01, 10, len(x))


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


