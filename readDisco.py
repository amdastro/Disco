import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import triangle
import triangle.plot
#import scipy.signal as signal
import pandas as pd
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal

from scipy.interpolate import griddata

from matplotlib.colors import LogNorm
#matplotlib.use('Agg')

from matplotlib import colors, ticker, cm
import sys
import math
import h5py
import os



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


    return t, riph[1:], r, piph, prim, plan, diag, index, nphi


def loadreport(filename):
    # jdot is read in after mdot2 in (most) runs with sinks
    t,torque,r2,p2,mdot,jdot = np.genfromtxt(filename,unpack=True,usecols=(0,1,3,4,5,6))
    orb = t/(2.*np.pi)

    return orb, torque, r2, p2, mdot, jdot


def loadplanet(filename):
    f = h5.File(filename, "r")
    t = f['Grid']['T'][0] /2./np.pi # orbits
    plan = f['Data']['Planets']
    m1 = plan[0,0]
    m2 = plan[1,0]
    r1 = plan[0,3]
    r2 = plan[1,3]
    phi1 = plan[0,4]
    phi2 = plan[1,4]
    eps2 = plan[1,5]

    return t, m2, r1, r2, phi1, phi2, eps2


def xy_interp(r,phi,data,angle,res):
    '''
    This reads in a data 1d array, the r and phi arrays,
    and the position of the planet to rotate the data.
    It interpolates it to an xy grid,
    rotates it,
    and returns the array to be contoured
    by contour(Xi,Yi,Zt)
    '''

    # general grid info
    up = 3.0
    low = -3.0

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    # rotate
    a = -angle
    x_rot = x*np.cos(a) - y*np.sin(a)
    y_rot = y*np.cos(a) + x*np.sin(a)


    # Making a new grid
    xi = np.linspace(low,up,res)
    yi = np.linspace(low,up,res)
    Xi,Yi = np.meshgrid(xi,yi)

    print 'gridding data...'

    phi_ax = np.linspace(0.,2.*np.pi,len(xi))
    #fig=plt.figure()
    Zi = griddata((x_rot,y_rot),data,(Xi,Yi))
    #Zi = griddata((xi,phi_ax), data, (Xi,Yi))
    #ax = fig.add_subplot(111, aspect='equal')
    #contour_levels =np.exp(np.linspace(math.log(1e-1),math.log(1e5),500))
    #CS=ax.contourf(Xi,Yi,Zi,contour_levels,norm=LogNorm(),cmap='afmhot' )
    #cnt = plt.contourf(Xi,Yi,np.log10(Zi),100,cmap=cmaps.inferno)#,norm=LogNorm())
    #cbar = plt.colorbar(cnt)#,ticks=(1  e-3,1e-2,1e-1,1e0,1e1,1e2,1e3))
    #plt.clim(vmin=-2,vmax=5)

    # This is the fix for the white lines between contour levels
    #for c in cnt.collections:
    #    c.set_edgecolor("face")

    #plt.xlim([x_plan_rot-0.5,x_plan_rot+0.5])
    #plt.ylim([y_plan_rot-0.5,y_plan_rot+0.5])
    #plt.colorbar(label=r'$\log{\Sigma}$')
    #plt.xlabel(r'$r/a$')

    #plt.show()

    # return the ROTATED interpolated Z data
    print 'Xi,Yi = ',np.shape(Xi),np.shape(Yi)
    print 'returning Z...'
    return Xi,Yi,Zi





def time_avg(files):
    # specify the checkpoint files to average over
    #num = np.arange(194,199)

    # Read the first file to initialize
    # the length of the average arrays
    #filenum0 = str(num[0]).zfill(4)
    #file = '%s/checkpoint_%s.h5'%(dir,filenum)

    # Need to rotate the data in the co-rotating frame
    # before summing and averaging


    # general grid info
    rmax = 2.5
    rmin = -2.5
    res = 64
    # Making a new grid
    # this will be the shape of the interpolated 2d output data array
    xi = np.linspace(rmin,rmax,res)
    yi = np.linspace(rmin,rmax,res)
    data_sum = np.zeros((res,res))

#    for i in range(1,len(num)):
    for i in range(0,len(files)):

        #filenum = str(num[i]).zfill(4)
        #file = '%s/checkpoint_%s.h5'%(dir,filenum)
        #print 'file = ',file


        t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(files[i])
        print 't = ', t/2./np.pi, ' orbits'
        dens = prim[:,0]
        # where are the planets?
        r_p = plan[0,3]
        r_p2 = plan[1,3]
        phi_p = plan[0,4] # this is in radians
        phi_p2 = plan[1,4]
        eps1 = plan[0,5]
        eps2 = plan[1,5]
        M1 = plan[0,0]
        M2 = plan[1,0]


        f_r1 = np.zeros(len(phi))
        f_phi1 = np.zeros(len(phi))
        f_r2 = np.zeros(len(phi))
        f_phi2 = np.zeros(len(phi))
        f_tot = np.zeros(len(phi))

        # get grav force components for all cells
        for i in range(0,len(phi)):
            f_1,f_r1[i],f_phi1[i] = fgrav_cell(r[i],phi[i],r_p,phi_p,M1, eps1) 
            f_2,f_r2[i],f_phi2[i] = fgrav_cell(r[i],phi[i],r_p2,phi_p2,M2, eps2)
            #f_r[i] = f_r1 + f_r2
            #f_phi[i] = f_phi1 + f_phi2
            f_tot[i] = f_1 + f_2

        f_r_tot = f_r1 + f_r2
        f_phi_tot = f_phi1 + f_phi2
        # torq exerted by gas on just the secondary
        torq = -dens*(f_phi2*r_p2)

        # Hill sphere
        r_hill = (1e-3/3)**(1./3)*r_p2
        print 'r_hill = ',r_hill

        # Cut out a region around the planet
        # Make a data array that gives distance from gas to planet 
        dx = r*np.cos(phi) - r_p2*np.cos(phi_p2)
        dy = r*np.sin(phi) - r_p2*np.sin(phi_p2)
        # distance from fluid element to planet
        script_r = np.sqrt(dx**2 + dy**2)
        f_rH = 1.
        cut_indx = np.where(script_r < f_rH*r_hill)
        torq[cut_indx] = 0.


        #data_sum is in xy format
        X_ax, Y_ax, data = xy_interp(r,phi,dens,phi_p2)
        data_sum += data



    # divide by number of files
    data_avg = data_sum / len(files)


    cnt = plt.contourf(X_ax,Y_ax,np.log10(data_avg),100,cmap=cmaps.magma)#,norm=LogNorm())
    cbar = plt.colorbar(cnt)#,ticks=(1  e-3,1e-2,1e-1,1e0,1e1,1e2,1e3))
    #plt.clim(vmin=-2,vmax=5)

    # This is the fix for the white lines between contour levels
    #for c in cnt.collections:
    #    c.set_edgecolor("face")

    #plt.xlim([x_plan_rot-0.5,x_plan_rot+0.5])
    #plt.ylim([y_plan_rot-0.5,y_plan_rot+0.5])
    #plt.colorbar(label=r'$\log{\Sigma}$')
    plt.xlabel(r'$r/a$')

    plt.show()


    # This returns an array that can then be triangulated
    # mayve triangulate it here and make the figure
    return X_ax, Y_ax, data_avg







def fgrav_cell(r, phi, r_p, phi_p, M, eps):
    '''
    this reads in coordinate data (r,phi) for ONE CELL in the grid
    and the planet data (r_p, phi_p), and calculates
    the gravitational force ON the planet BY the gas.

    ** Need to multiply by density of gas to get force **

    Summing this up over both planets gives components
    of the gravitational force on the binary by the gas.

    '''

    dx = r*np.cos(phi) - r_p*np.cos(phi_p)
    dy = r*np.sin(phi) - r_p*np.sin(phi_p)

    # distance from fluid element to planet
    script_r = np.sqrt(dx**2 + dy**2)

    f = -M*script_r/(script_r**2 + eps**2)**(3./2)

    cosa = dx/script_r
    sina = dy/script_r

    cosa_p = cosa*np.cos(phi_p) + sina*np.sin(phi_p)
    sina_p = sina*np.cos(phi_p) - cosa*np.sin(phi_p) 

    f_r = cosa_p * f
    f_phi = sina_p * f

    return f, f_r, f_phi

def zgrav(r,phi,r_p,phi_p,M):
    theta=phi_p-phi
    r1=r**2 + r_p**2 -2*r*r_p*np.cos(theta)
    F=-M/r1**2
    sAlpha=np.sin(theta)*r_p/r1
    cAlpha=np.sqrt(1-sAlpha**2)
    return F,cAlpha*F,sAlpha*F



def torq_cont(file,f_rH):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)

    dens = prim[:,0]
    orb = t/2./np.pi

    # where are the planets?
    r_p = plan[0,3]
    r_p2 = plan[1,3]
    phi_p = plan[0,4] # this is in radians
    phi_p2 = plan[1,4]
    eps1 = plan[0,5]
    eps2 = plan[1,5]
    x_plan = np.multiply(r_p,np.cos(phi_p))
    y_plan = np.multiply(r_p,np.sin(phi_p))
    x_plan2 = np.multiply(r_p2,np.cos(phi_p2))
    y_plan2 = np.multiply(r_p2,np.sin(phi_p2))


    f_r1 = np.zeros(len(phi))
    f_phi1 = np.zeros(len(phi))
    f_r2 = np.zeros(len(phi))
    f_phi2 = np.zeros(len(phi))
    f_tot = np.zeros(len(phi))

    M1 = plan[0,0]
    M2 = plan[1,0]

    # get grav force components for all cells
    # 
    for i in range(0,len(phi)):
        f_1,f_r1[i],f_phi1[i] = fgrav_cell(r[i],phi[i],r_p,phi_p,M1, eps1) 
        f_2,f_r2[i],f_phi2[i] = fgrav_cell(r[i],phi[i],r_p2,phi_p2,M2, eps2)
        #f_r[i] = f_r1 + f_r2
        #f_phi[i] = f_phi1 + f_phi2
        f_tot[i] = f_1 + f_2

    f_r_tot = f_r1 + f_r2
    f_phi_tot = f_phi1 + f_phi2
    # torq exerted by gas on just the secondary
    torq = -dens*(f_phi2*r_p2)




    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    # rotate
    a = -phi_p2
    x_rot = x*np.cos(a) - y*np.sin(a)
    y_rot = y*np.cos(a) + x*np.sin(a)
    x_plan2_rot = x_plan2*np.cos(a) - y_plan2*np.sin(a)
    y_plan2_rot = y_plan2*np.cos(a) + x_plan2*np.sin(a)

    # Hill sphere
    r_hill = (1e-3/3)**(1./3)*r_p2
    print 'r_hill = ',r_hill

    # Cut out a region around the planet
    # Make a data array that gives distance from gas to planet 
    dx = r*np.cos(phi) - r_p2*np.cos(phi_p2)
    dy = r*np.sin(phi) - r_p2*np.sin(phi_p2)
    # distance from fluid element to planet
    script_r = np.sqrt(dx**2 + dy**2)

    cut_indx = np.where(script_r < f_rH*r_hill)
    torq[cut_indx] = 0.

    # Now triangulate with x,y to contour the data

    triang = tri.Triangulation(x_rot, y_rot)
    ## mask off unwanted triangles
    min_radius = 1.01*np.min(r)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)


    #contour_levels =np.exp(np.linspace(math.log(1e0),math.log(1e5),100))
    contour_neg = -np.logspace(1,np.log10(np.max(torq)),100)[::-1]
    contour_pos = np.logspace(1,np.log10(np.max(torq)),100)
    #contour_levels = np.append(contour_neg,contour_pos)
    #contour_levels = np.linspace(-0.07,0.07,200)
    contour_levels = np.linspace(-0.95*np.max(abs(torq)),0.95*np.max(abs(torq)),100)
    #contour_levels = np.linspace(-1550.,1550.,200)
    cnt = plt.tricontourf(triang,torq,contour_levels,\
        cmap='RdBu', rasterized=True)


    plt.colorbar(label=r'$T$')#,ticks=(-1e5,-1e3,0,1e3,1e5))
    #plt.clim(vmin=-50000,vmax=50000)
    #plt.xlim([0.9,1.1])
    #plt.ylim([-0.1,0.1])
    plt.xlim([0.5,1.5])
    plt.ylim([-0.5,0.5])

    # phi = 0 line:
    #plt.plot([-2.5,2.5],[0,0.],linestyle='--',
    #        color='black',linewidth=1.)
    # non rotated secondary position:
    #plt.plot(r_p2*np.cos(phi_p2),r_p2*np.sin(phi_p2),linestyle='none',marker='x',
    #        color='black',markersize=7,markeredgewidth=1)
    # rotated secondary position:
    plt.plot(x_plan2_rot,y_plan2_rot,linestyle='none',marker='x',
            color='black',markersize=7,markeredgewidth=1)
    #plt.clim(vmin=-1.5,vmax=1.5)

    # Plot circles for the hill sphere, or planets
    fig = plt.gcf()
    ax = fig.gca()
    hillsph = plt.Circle((x_plan2_rot, y_plan2_rot), r_hill, \
         fill=False, color='k',linestyle='dashed')
    ax.add_artist(hillsph)
    #ax.add_artist(planet2)


    #plt.clim(vmin=-1.5,vmax=1.3)
    # This is the fix for the white lines between contour levels
    #for c in cnt.collections:
    #    c.set_edgecolor("face")

    plt.show()

    return triang, torq


def get_the_torq(file):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)

    dens = prim[:,0]
    orb = t/2./np.pi

    # where are the planets?
    r_p = plan[0,3]
    r_p2 = plan[1,3]
    phi_p = plan[0,4] # this is in radians
    phi_p2 = plan[1,4]
    eps1 = plan[0,5]
    eps2 = plan[1,5]
    x_plan = np.multiply(r_p,np.cos(phi_p))
    y_plan = np.multiply(r_p,np.sin(phi_p))
    x_plan2 = np.multiply(r_p2,np.cos(phi_p2))
    y_plan2 = np.multiply(r_p2,np.sin(phi_p2))


    f_r1 = np.zeros(len(phi))
    f_phi1 = np.zeros(len(phi))
    f_r2 = np.zeros(len(phi))
    f_phi2 = np.zeros(len(phi))
    f_tot = np.zeros(len(phi))

    M1 = plan[0,0]
    M2 = plan[1,0]

    # get grav force components for all cells
    # 
    for i in range(0,len(phi)):
        f_1,f_r1[i],f_phi1[i] = fgrav_cell(r[i],phi[i],r_p,phi_p,M1, eps1) 
        f_2,f_r2[i],f_phi2[i] = fgrav_cell(r[i],phi[i],r_p2,phi_p2,M2, eps2)
        #f_r[i] = f_r1 + f_r2
        #f_phi[i] = f_phi1 + f_phi2
        f_tot[i] = f_1 + f_2

    f_r_tot = f_r1 + f_r2
    f_phi_tot = f_phi1 + f_phi2
    # torq exerted by gas on just the secondary
    torq = -dens*(f_phi2*r_p2)

    return r, phi, phi_p2, torq


def get_the_triang(file):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    r_p2 = plan[1,3]
    phi_p2 = plan[1,4]
    x_plan2 = np.multiply(r_p2,np.cos(phi_p2))
    y_plan2 = np.multiply(r_p2,np.sin(phi_p2))


    # rotate
    a = -phi_p2
    x_rot = x*np.cos(a) - y*np.sin(a)
    y_rot = y*np.cos(a) + x*np.sin(a)
    x_plan2_rot = x_plan2*np.cos(a) - y_plan2*np.sin(a)
    y_plan2_rot = y_plan2*np.cos(a) + x_plan2*np.sin(a)


    # Now triangulate with x,y

    triang = tri.Triangulation(x_rot, y_rot)
    ## mask off unwanted triangles
    min_radius = 1.01*np.min(r)
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)

    return triang



def data_avg(dir):

    # this isn't working YET because data needs to be averaged with the 2D interpolation
    # and I want to specify quentities with the 1D array...

    files = []
    for f in os.listdir(dir):
      if os.path.isfile(os.path.join(dir,f)) and f.startswith('checkpoint'):
        files = np.append(files,dir+f)

    print 'averaging over: \n', files

    # Needs to be interpolated to 2D array to average
    # general grid info
    rmax = 3.0
    rmin = -3.0
    res = 500
    # Making a new grid
    # this will be the shape of the interpolated 2d output data array
    xi = np.linspace(rmin,rmax,res)
    yi = np.linspace(rmin,rmax,res)
    data_sum = np.zeros((res,res))


    for i in range(0,len(files)):
        r, phi, phi_p2, data_1d = get_the_torq(files[i])
        # define torq inner and torq outer from 1D array, then sum/average from 2D grid

        #data_sum is in xy format
        x_axis, y_axis, data_2d = xy_interp(r,phi,data_1d,phi_p2,res)
        data_sum += data_2d



    # divide by number of files
    data_avg = data_sum / len(files)
 

    # only need one triangulation


    return x_axis, y_axis, data_avg




def radial_diag(file,mach,f_rH):
    # f_rH is the fraction of the hill sphere to cut out
    t,r_ax,r,phi,prim,plan,diag, index,nphi = loadcheckpoint(file)
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

    rp_2 = plan[1,3]
    q = plan[1,0]
    om2 = plan[1,2]

    sigma_0 = 1.
    # type 1 torque:
    # array?:
    #T_0 = q**2 * mach**2 * sigma_0 * r_ax**4 * omega**2
    # single value?
    T_0 = - q**2 * mach**2 * sigma_0 * rp_2**4 * om2**2

    # denote hill sphere
    r_hill = (1e-3/3)**(1./3) * rp_2
    # define scale height to scale axes
    height = 1./mach

    cut_indx = np.where((r_ax-rp_2 < f_rH*r_hill) & (r_ax-rp_2 > -f_rH*r_hill))
    torq[cut_indx] = 0.


    output_arr = np.array([r_ax,rhoavg,torq]).T
    np.savetxt("/Users/Shark/Desktop/DISCO/Sequin/m20_radial.dat", output_arr, '%10.5f', delimiter='   ')
    print 'file made'



    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    ax1.plot((r_ax-rp_2)/height,rhoavg,color='k',alpha=0.6)

    #ax1.axvline(x=0, linewidth=2, linestyle='--',color='r')
    ax1.axvline(x=-r_hill/height, linestyle='--', linewidth=2, color='darkcyan')
    ax1.axvline(x=r_hill/height, linestyle='--', linewidth=2, color='darkcyan')
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$\rho$')


    ax2.plot((r_ax-rp_2)/height,torq,color='k',alpha=0.6)
    #ax2.axvline(x=0, linewidth=2, color='r')
    ax2.axvline(x=-r_hill/height, linestyle='--', linewidth=2, color='darkcyan')
    ax2.axvline(x=r_hill/height, linestyle='--', linewidth=2, color='darkcyan')
    ax2.set_ylabel(r'$dT/dr$')
    plt.axhline(y=0.,linestyle='--',color='k',linewidth=0.5)
    #ax2.set_xlim([-0.75,0.75])
    ax2.set_xlabel(r'$(r - r_2)/h$')
    #ax2.set_yscale('symlog')
    #ax2.set_ylabel(r'$l$')

    plt.tight_layout()

    # Fine-tune figure; make subplots close to each other and hide x ticks for
    # all but bottom plot.
    f.subplots_adjust(hspace=0.1)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.show()
    #return r_ax-rp_2, rhoavg, torq



def plot2d(file,savedir):
    t,r_ax,r,phi,prim,plan,diag = loadcheckpoint(file)

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



def zoom_cont(file):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)


    dens = prim[:,0]
    pres = prim[:,1]
    velr = prim[:,2]
    vphi = prim[:,3]
    velz = prim[:,4]
    orb = t/2./np.pi

    # where is the planet?
    r2 = plan[1,3]
    phi2 = plan[1,4] # this is in radians
    x_plan2 = np.multiply(r2,np.cos(phi2))
    y_plan2 = np.multiply(r2,np.sin(phi2))

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    # Hill radius
    #r_hill = (1e-3/3)**(1./3)*r2

    # no rotation
    #x_rot = x
    #y_rot = y

    #  rotation
    a = -phi2
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

    contour_levels =np.exp(np.linspace(math.log(1e-2),math.log(5e3),200))

    cnt = plt.tricontourf(triang,dens,contour_levels,\
        norm=LogNorm(),cmap=cmaps.viridis, rasterized=True)
    #cnt = plt.tricontourf(triang,dens,100,cmap=cmaps.inferno)

    #plt.colorbar()
    plt.colorbar(ticks=(1e-2,1e-1,1e0,1e1,1e2,1e3),label='$\Sigma$')
    # This is the fix for the white lines between contour levels
    #for c in cnt.collections:
    #    c.set_edgecolor("face")

    #plt.xlim([x_plan2_rot-0.5,x_plan2_rot+0.5])
    #plt.ylim([y_plan2_rot-0.5,y_plan2_rot+0.5])
    plt.xlabel(r'$r/r_0$')

    #fig = plt.gcf()
    #ax = fig.gca()
    #hillsph = plt.Circle((x_plan2_rot, y_plan2_rot), r_hill, \
    #     fill=False, color='white',linestyle='dashed')
    #ax.add_artist(hillsph)

    plt.show()
    return triang





def torq_movie(dir):

    rep_file = dir+'report.dat'
    #check_file = dir + 'checkpoint_%s.h5'%chk_num
    orb, torque, r2, p2, mdot2 = loadreport(rep_file)

    window = int(orb[-1]-orb[0])/2
    print 'window = ',window
    poly_order = 3
    #smoothed = savitzky_golay(torque/T_0, window, poly_order)
    #smoothed = signal.medfilt(torque[cut:]/T_0,1115)
    smoothed = pd.rolling_mean(torque,window)


    x = orb
    y = smoothed
    size = np.random.randint(150, size=2)
    colors = np.random.choice(["r", "g", "b"], size=2)

    fig = plt.figure()
    plt.xlabel(r'$t \, \rm [orb]$', fontsize=25, fontweight='bold')
    plt.ylabel(r'$ $', fontsize=25, fontweight='bold')
    plt.xlim([orb[0],orb[-1]])
    plt.ylim([-0.005,0.003])
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=0.5, linestyle='--',color='k')


    for i in range(22000,len(smoothed),200):
        plt.plot(orb[0:i:100],smoothed[0:i:100],linewidth=3,color='darkred')
        plt.savefig('/Users/Shark/Desktop/GW_m20_lownu_nosink/torq_movie/torq_%i'%i)






def readtorque(chk_file,rep_file,mach):
    # Might have mistakes in normalizing
    t, torque, torque2, r2 = loadreport(rep_file)
    t_snap,r_ax,r,phi,prim,plan,index,nphi = loadcheckpoint(chk_file)

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
    smoothed = pd.rolling_mean(torque,window)

    avg_torq = np.average(smoothed[-1001:-1])
    print 'averaging from ',orb[-1001], ' to ',orb[-1], ' orbits'

    return avg_torq

# new version given the smoothed torq!
def plottorq(dir):

    #  SOMETHING NEEDS TO BE UPDATED HERE!


    t,torque,torque2 = np.genfromtxt('%s/smoothed.txt'%dir,unpack=True)
    #check_file = '%scheckpoint_%s.h5'%(dir,chk_num)
    #t_snap,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(check_file)

    # latest distance of secondary
    #a = plan[1,3]

    # ang velocity of secondary
    #om2 = plan[1,2]


    #q = plan[1,0]
    #sigma_0 = 1.
    # type 1 torque:
    #T_0 = q**2 * mach**2 * sigma_0 * a**4 * om2**2

    orb = t/2/np.pi

    plt.plot(orb,torque,color='#48BBD0',alpha=1)
    plt.xlabel(r'$t \, \rm [orb]$')
    plt.ylabel(r'$\rm T$')
    plt.xlim([orb[0],orb[-1]])
    #plt.ylim([-1,1])

    #plt.close()
    # horizontal dotted line at y=0
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=0.5, linestyle='--',color='k')

    # smooth over last 150 orbits
    #avg_ind = 20

    #print 'averaging over ',orb[-1] - orb[-avg_ind-1],' orbits '

    #avg_torq = np.average(torque[-avg_ind-1:-1]/T_0)
    #print 'avg normalized torque = ',avg_torq

    #plt.tight_layout()
    plt.show()

    return orb, torque 



def plottorque(dir):
    rep_file = dir+'report.dat'
    #check_file = dir + 'checkpoint_%s.h5'%chk_num
    orb, torque, r2, p2, mdot2, jdot2 = loadreport(rep_file)
    #t_snap,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(check_file)

    # latest distance of secondary
    #a = plan[1,3]
    #print 'a = ',a
    # ang velocity of secondary
    #om2 = plan[1,2]
    #print 'omega = ',om2

    #q = plan[1,0]
    #sigma_0 = 1.
    # type 1 torque:
    # need to divide by separation and omega throughout the sim?
    # (dont have updated omega in repfile)
    #T_0 = q**2 * mach**2 * sigma_0 * r2**4 * om2**2

    # cut data from beginning out for smooting
    # provide index to cut out till
    #cut = 10
    #t = t[cut:]
    #torque = torque[cut:]

    window = int(orb[-1]-orb[0])/2
    print 'window = ',window
    poly_order = 3
    #smoothed = savitzky_golay(torque/T_0, window, poly_order)
    #smoothed = signal.medfilt(torque[cut:]/T_0,1115)
    smoothed = pd.rolling_mean(torque,window) 

    #plt.plot(orb,torque/T_0,color='#48BBD0',alpha=0.5)
    plt.plot(orb[::100],torque[::100],color='teal')
    plt.xlabel(r'$t \, \rm [orb]$')
    plt.ylabel(r'$\rm T$')
    plt.xlim([orb[0],orb[-1]])
    #plt.ylim([-1,1])

    # horizontal dotted line at y=0
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=0.5, linestyle='--',color='k')

    avg_torq = np.average(smoothed[-1001:-1])
    print 'avg normalized torque = ',avg_torq

    #plt.annotate(r'$\mathcal{M} = %i$'%mach,xy=(500,0.07))
    #plt.annotate(r'$\dot{a}/\Omega a = %.1e$'%drift,xy=(500,0.045))
    #plt.annotate(r'$T/T_0 = %.2f$'%avg_torq,xy=(500,0.02))

    plt.tight_layout()
    plt.show()

    return orb, torque, smoothed



def torq_prof(file,mach,f_rH):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)
    orb = t/2./np.pi
    print orb, ' orb'

    torq = diag[:,2]

    # where are the planets?
    r_p2 = plan[1,3]


    # denote hill sphere
    r_hill = (1e-3/3)**(1./3) * r_p2

    # define scale height to scale axes
    height = 1./mach

    cut_indx = np.where((r_ax-r_p2 < f_rH*r_hill) & (r_ax-r_p2 > -f_rH*r_hill))
    torq[cut_indx] = 0.



    return r_ax, torq


def phi_avg(file,mach,f_rH):

    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)

    orb = t/2./np.pi
    print orb, ' orb'

    # where are the planets?
    r_p = plan[0,3]
    r_p2 = plan[1,3]
    phi_p = plan[0,4] # this is in radians
    phi_p2 = plan[1,4]
    eps1 = plan[0,5]
    eps2 = plan[1,5]


    dens = prim[:,0]

    f_r2 = np.zeros(len(phi))
    f_phi2 = np.zeros(len(phi))

    M1 = plan[0,0]
    M2 = plan[1,0]

    # get grav force components for all cells
    # 
    for i in range(0,len(phi)):
        f_2,f_r2[i],f_phi2[i] = fgrav_cell(r[i],phi[i],r_p2,phi_p2,M2,eps2)

    # torq exerted by gas on just the secondary
    torq = -dens*f_phi2*r_p2


    # choose your data to average
    data = torq


    # create a radial array that stores the
    # average data per annulus
    data_prof = np.zeros(len(r_ax)-1)
    data_prof[0] = np.sum(data[0:index[1][0]-1]) / nphi[0][0]

    for i in range(1,len(r_ax)-2):
        summ = np.sum(data[index[i][0]:index[i+1][0]-1])
        data_prof[i] = summ / nphi[i][0]

    # confused with index of index array...
    r_array = r_ax[:-2]
    phi_avg = data_prof[:-1]

    # denote hill sphere
    r_hill = (1e-3/3)**(1./3) * r_p2

    # define scale height to scale axes
    height = 1./mach

    cut_indx = np.where((r_array-r_p2 < f_rH*r_hill) & (r_array-r_p2 > -f_rH*r_hill))
    data_prof[cut_indx] = 0.

    #plt.axvline(x=0., linewidth=2,color='r')
    plt.plot((r_array-r_p2)/height,phi_avg,color='black',linewidth=2)
    plt.xlabel(r'$(r-r_p)/h$')
    #plt.yscale('log')
    #plt.ylim([1e-2,1e2])
    #plt.ylabel(r'$\rho_{\rm avg}$')

    plt.show()

    print 'max = ',np.max(phi_avg)

    return r_array,phi_avg




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


