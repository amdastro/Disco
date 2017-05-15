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
    index = f['Grid']['Index'][...].T
    # idPhi0 tells you the index of which cells is at phi = 0
    idPhi0 = f['Grid']['Id_phi0'][...].T
    # nphi is the number of phi zones in each annulus 
    # (Same as taking index[i] - index[i-1])
    nphi = f['Grid']['Np'][...].T
    t = f['Grid']['T'][0]
    riph = f['Grid']['r_jph'][...]
    ziph = f['Grid']['z_kph'][...]
    # Planet data:
    # this returns 6 variables for each planet
    # planets[0,:] = [M1,v_r1,omega1,r1,phi1,eps1]
    # planets[1,:] = [M2,v_r2,omega2,r2,phi2,eps2]
    plan = f['Data']['Planets']

    # There are currently 5 radial diagnostics:
    # these are averaged azimuthally.
    # diag[:,0] = rho
    # diag[:,1] = mdot
    # diag[:,2] = torque density
    # diag[:,3] = advected angular momentum
    # diag[:,4] = omega
    diag = f['Data']['Radial_Diagnostics'][...]


    r = np.zeros(piph.shape)
    R = 0.5*(riph[1:] + riph[:-1])
    for i in xrange(index.shape[0]):
        for k in xrange(index.shape[1]):
            ind0 = index[i,k]
            ind1 = ind0 + nphi[i,k]
            r[ind0:ind1] = R[i]


    return t, riph[1:], r, piph, prim, plan, diag


def loadreport(filename):

    t,torque,torque2,r2 = np.genfromtxt(filename,unpack=True,usecols=(0,1,2,3))

    return t, torque, torque2, r2


def loadplanet(filename):
    f = h5.File(filename, "r")
    t = f['Grid']['T'][0] /2./np.pi # orbits
    plan = f['Data']['Planets']
    m1 = plan[0,0]
    m2 = plan[1,0]
    r2 = plan[1,3]
    phi2 = plan[1,4]

    return t, m2, r2, phi2


def grav_pot_cell(r, phi, r_p, phi_p, M_1):
    '''
    this reads in coordinate data (r,phi) for ONE CELL in the grid
    and the planet data (r_p, phi_p). probably the primary pplanet.
    * Needs to be adapted for the secondary / the enire binary.
    This can later be summed up over all cells in another function.

    Adapted from Paul's DISCO planetaryForce() routine
    '''

    dx = r*np.cos(phi) - r_p*np.cos(phi_p)
    dy = r*np.sin(phi) - r_p*np.sin(phi_p)

    script_r = np.sqrt(dx**2+dy**2)

    f1 = M_1/np.sqrt(dx**2 + dy**2)
    # f2 = ??

    cosa = dx/script_r 
    sina = dy/script_r 

    cosa_p = cosa*np.cos(phi_p) + sina*np.sin(phi_p)
    sina_p = sina*np.cos(phi_p) + cosa*np.sin(phi_p) 

    f_r = cosa_p*f1 
    f_phi = sina_p * f1

    return f_r, f_phi



def grav_pot_1(file):
    t,r_ax,r,phi,prim,plan,diag = loadcheckpoint(file)

    dens = prim[:,0]
    orb = t/2./np.pi

    # where is the planet?
    r_p = plan[0,3]
    r_p2 = plan[1,3]
    phi_p = plan[0,4] # this is in radians
    phi_p2 = plan[1,4]
    x_plan = np.multiply(r_p,np.cos(phi_p))
    y_plan = np.multiply(r_p,np.sin(phi_p))
    x_plan2 = np.multiply(r_p2,np.cos(phi_p2))
    y_plan2 = np.multiply(r_p2,np.sin(phi_p2))

    f_r = np.zeros(len(phi))
    f_phi = np.zeros(len(phi))

    # MASS = 1 for primary
    M1 = 1.

    # fi
    for i in range(0,len(phi)):
        f_r[i],f_phi[i] = grav_pot_cell(r[i],phi[i],r_p,phi_p,M1)


    # Now triangulate with x,y to contour the data

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    # rotate the grid
    a = -phi_p2
    x_rot = x*np.cos(a) - y*np.sin(a)
    y_rot = y*np.cos(a) + x*np.sin(a)
    x_plan2_rot = x_plan2*np.cos(a) - y_plan2*np.sin(a)
    y_plan2_rot = y_plan2*np.cos(a) + x_plan2*np.sin(a)

    triang = tri.Triangulation(x_rot, y_rot)
    # mask off unwanted traingles
    min_radius = 1.01*np.min(r)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)

    ###fig, ax = plt.subplots()
    cnt = plt.tricontourf(triang,f_phi,100,cmap=cmaps.inferno, rasterized=True)
    plt.colorbar(label=r'$f_{\phi}$')


    # Plot circles for the planets
    fig = plt.gcf()
    ax = fig.gca()
    # Define the positions (secondary is rotated to phi=0)
    planet1 = plt.Circle((x_plan, y_plan), 0.05, color='#1A7989')
    planet2 = plt.Circle((x_plan2_rot, y_plan2_rot), 0.03, color='#1A7989')
    ax.add_artist(planet1)
    ax.add_artist(planet2)


    #plt.clim(vmin=-1.5,vmax=1.3)
    # This is the fix for the white lines between contour levels
    for c in cnt.collections:
        c.set_edgecolor("face")

    plt.show()





def radial_diag(file,mach):
    t,r_ax,r,phi,prim,plan,diag = loadcheckpoint(file)
    # diag[:,0] = rho
    # diag[:,1] = mdot
    # diag[:,2] = torque density
    # diag[:,3] = advected angular momentum
    # diag[:,4] = omega

    rhoavg = diag[:,0]
    mdot   = diag[:,1]
    torq   = diag[:,2]
    angmom = diag[:,3]
    omega  = diag[:,4]

    a = plan[1,3]
    q = plan[1,0]
    om2 = plan[1,2]

    sigma_0 = 1.
    # type 1 torque:
    T_0 = q**2 * mach**2 * sigma_0 * a**4 * om2**2
    print T_0

    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    ax1.plot(r_ax-a,rhoavg,color='r')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\rho$')

    ax2.plot(r_ax-a,torq,color='darkcyan')
    ax2.set_ylabel(r'$\Gamma_{\phi}$')
    ax2.set_xlim([-0.75,0.75])
    ax2.set_xlabel(r'$r - r_2$')
    #ax2.set_yscale('log')
    #ax2.set_ylim([1e-5,1e2])
    #ax2.set_ylabel(r'$l$')

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0.1)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.show()


def plot2d(file,savedir):
    t,r_ax,r,phi,prim,plan,diag = loadcheckpoint(file)

    print t/2./np.pi, ' orb'

    dens = prim[:,0]
    pres = prim[:,1]
    velr = prim[:,2]
    vphi = prim[:,3]
    velz = prim[:,4]
    orb = t/2./np.pi

    torq_dens = diag[:,3]

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



def zoom_cont(file):
    t,r_ax,r,phi,prim,plan,diag = loadcheckpoint(file)

    print t/2./np.pi, ' orb'

    dens = prim[:,0]
    pres = prim[:,1]
    velr = prim[:,2]
    vphi = prim[:,3]
    velz = prim[:,4]
    orb = t/2./np.pi

    # where is the planet?
    r2 = plan[1,3]
    phi2 = plan[1,4] # this is in radians
    x_plan = np.multiply(r2,np.cos(phi2))
    y_plan = np.multiply(r2,np.sin(phi2))

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    # test rotation
    a = -phi2
    x_rot = x*np.cos(a) - y*np.sin(a)
    y_rot = y*np.cos(a) + x*np.sin(a)
    x_plan_rot = x_plan*np.cos(a) - y_plan*np.sin(a)
    y_plan_rot = y_plan*np.cos(a) + x_plan*np.sin(a)

    triang = tri.Triangulation(x_rot, y_rot)
    # mask off unwanted traingles
    min_radius = 1.01*np.min(r)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)


    cnt = plt.tricontourf(triang,np.log10(dens),100,cmap=cmaps.inferno, rasterized=True)

    plt.clim(vmin=-1.5,vmax=1.3)
    # This is the fix for the white lines between contour levels
    for c in cnt.collections:
        c.set_edgecolor("face")

    plt.xlim([x_plan_rot-0.5,x_plan_rot+0.5])
    plt.ylim([y_plan_rot-0.5,y_plan_rot+0.5])
    plt.colorbar(label=r'$\log{\Sigma}$')
    #plt.annotate(r'$%i \, \rm orb$'%orb,xy=(x_plan_rot+0.2,y_plan_rot+0.32),color='white')

    plt.show()





def readtorque(chk_file,rep_file,mach):
    t, torque, torque2, r2 = loadreport(rep_file)
    t_snap,r_ax,r,phi,prim,plan = loadcheckpoint(chk_file)

    # latest distance of secondary
    a = plan[1,3]
    print 'a = ',a
    # ang velocity of secondary
    om2 = plan[1,2]
    print 'omega = ',om2

    orb = t/2/np.pi

    q = plan[1,0]
    sigma_0 = 1.
    # type 1 torque:
    T_0 = q**2 * mach**2 * sigma_0 * a**4 * om2**2

    window = int(orb[-1]-orb[0])/2
    smoothed = pd.rolling_mean(torque/T_0,window)

    avg_torq = np.average(smoothed[-1001:-1])
    print 'averaging from ',orb[-1001], ' to ',orb[-1], ' orbits'

    return avg_torq

# new version given the smoothed torq!
def plottorq(dir,mach,drift,chk_num):
    t,torque,torque2 = np.genfromtxt('%s/smoothed.txt'%dir,unpack=True)
    check_file = '%scheckpoint_%s.h5'%(dir,chk_num)
    t_snap,r_ax,r,phi,prim,plan,diag = loadcheckpoint(check_file)

    # latest distance of secondary
    a = plan[1,3]

    # ang velocity of secondary
    om2 = plan[1,2]


    q = plan[1,0]
    sigma_0 = 1.
    # type 1 torque:
    T_0 = q**2 * mach**2 * sigma_0 * a**4 * om2**2

    orb = t/2/np.pi

    plt.plot(orb,torque2/T_0,color='#48BBD0',alpha=0.5)
    plt.xlabel(r'$t \, \rm [orb]$')
    plt.ylabel(r'$\rm T/T_0$')
    plt.xlim([orb[0],orb[-1]])
    #plt.ylim([-1,1])

    #plt.close()
    # horizontal dotted line at y=0
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=0.5, linestyle='--',color='k')

    print 'averaging over ',orb[-1] - orb[-51],' orbits '

    avg_torq = np.average(torque[-51:-1]/T_0)
    #print 'avg normalized torque = ',avg_torq

    #plt.tight_layout()
    plt.show()

    #return orb, torque/T_0
    return avg_torq



def plottorque(dir,mach,drift,savedir, chk_num):
    rep_file = dir+'report.dat'
    check_file = dir + 'checkpoint_%s.h5'%chk_num
    t, torque, torque2, r2 = loadreport(rep_file)
    t_snap,r_ax,r,phi,prim,plan,diag = loadcheckpoint(check_file)

    # latest distance of secondary
    a = plan[1,3]
    print 'a = ',a
    # ang velocity of secondary
    om2 = plan[1,2]
    print 'omega = ',om2

    q = plan[1,0]
    sigma_0 = 1.
    # type 1 torque:
    T_0 = q**2 * mach**2 * sigma_0 * a**4 * om2**2

    # cut data from beginning out for smooting
    # provide index to cut out till
    cut = 10
    t = t[cut:]
    torque = torque[cut:]

    orb = t/2/np.pi

    window = int(orb[-1]-orb[0])/2
    print 'window = ',window
    poly_order = 3
    #smoothed = savitzky_golay(torque/T_0, window, poly_order)
    #smoothed = signal.medfilt(torque[cut:]/T_0,1115)
    smoothed = pd.rolling_mean(torque/T_0,window)

    #plt.plot(orb,torque/T_0,color='#48BBD0',alpha=0.5)
    plt.plot(orb[::100],smoothed[::100],color='#C20000')
    plt.xlabel(r'$t \, \rm [orb]$')
    plt.ylabel(r'$\rm T/T_0$')
    plt.xlim([orb[0],orb[-1]])
    plt.ylim([-1,1])

    # horizontal dotted line at y=0
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=0.5, linestyle='--',color='k')

    avg_torq = np.average(smoothed[-1001:-1])
    print 'avg normalized torque = ',avg_torq

    #plt.annotate(r'$\mathcal{M} = %i$'%mach,xy=(500,0.07))
    #plt.annotate(r'$\dot{a}/\Omega a = %.1e$'%drift,xy=(500,0.045))
    #plt.annotate(r'$T/T_0 = %.2f$'%avg_torq,xy=(500,0.02))

    plt.tight_layout()
    plt.show()

    return orb, torque/T_0, smoothed


def dens_prof(file):
    #t,r,phi,prim,plan = loadcheckpoint(file)

    f = h5.File(file, "r")

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
    # Planet data:
    # this returns 6 variables for each planet
    # planets[0,:] = [M1,v_r1,omega1,r1,phi1,eps1]
    # planets[1,:] = [M2,v_r2,omega2,r2,phi2,eps2]
    plan = f['Data']['Planets']

    orb = t/2./np.pi
    print orb, ' orb'

    dens = prim[:,0]


    # where is the planet?
    r2 = plan[1,3]
    phi2 = plan[1,4] # this is in radians

    # create a radial array that stores the
    # average density per annulus
    dens_prof = np.zeros(len(riph)-1)
    dens_prof[0] = np.sum(dens[0:index[1][0]-1]) / nphi[0][0]

    for i in range(1,len(riph)-2):
        sum = np.sum(dens[index[i][0]:index[i+1][0]-1])
        dens_prof[i] =  sum / nphi[i][0]

    # confused with index of index array...
    r_array = riph[:-2]
    rho_avg = dens_prof[:-1]

    plt.axvline(x=r2, ymin=0, ymax=100, linewidth=2,\
        linestyle='--',color='#DF3A01')
    plt.plot(r_array,rho_avg,color='black',linewidth=2)
    plt.xlabel(r'$r$')
    plt.yscale('log')
    plt.ylim([1e-2,1e2])
    plt.annotate(r'$%i \, \rm orb$'%orb,\
        xy=(2,5e1),color='k')

    plt.ylabel(r'$\rho_{\rm avg}$')

    plt.show()

    return t, r2, r_array, rho_avg    


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


