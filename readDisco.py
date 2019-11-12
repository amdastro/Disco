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
from scipy.interpolate import interp1d

from scipy.integrate import trapz, simps

from matplotlib.colors import LogNorm
#matplotlib.use('Agg')

from matplotlib import colors, ticker, cm
import math
import h5py
import os

from scipy.signal import savgol_filter


plt.close("all")


#import colormaps as cmaps

label_size = 16
fontsize = 16

#plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=label_size)
plt.rc('ytick', labelsize=label_size)
plt.rc('axes', labelsize=label_size)
plt.rc({'figure.autolayout': True})



plt.rcParams['lines.linewidth'] = 2.

from matplotlib.colors import ListedColormap, BoundaryNorm

import seaborn as sns


cmapgreen = ListedColormap(sns.cubehelix_palette(10, start=.5, rot=-.75))
cmappink = ListedColormap(sns.cubehelix_palette(10,dark=0.25, light=0.9, hue=1.2, rot=.5))
cmappink_r = ListedColormap(sns.cubehelix_palette(10,dark=0.2, light=0.9, hue=1.2, rot=.5,reverse=True))
cmap_cf = ListedColormap(sns.cubehelix_palette(10, start=2.2, dark=0.3, light=0.85, rot=.7, hue=1.2, gamma=.7))
cmap_purp = sns.cubehelix_palette(40,dark=0., light=0.9, hue=1., gamma=.7, rot=.5,start=2.3,as_cmap=True)

# diverging
cmap_rdbu = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

'''
prim[:,0] = rho
prim[:,1] = P
prim[:,2] = u_r
prim[:,3] = u_phi
prim[:,4] = u_z
'''



def loadCheckpointData(filename):
    ## for nyu-Cal Disco

    f = h5.File(filename, "r")

    piph = f['Data']['Cells'][:,-1][...]
    prim = f['Data']['Cells'][:,:-1][...]
    index = f['Grid']['Index'][...]
    idPhi0 = f['Grid']['Id_phi0'][...]
    nphi = f['Grid']['Np'][...]
    t = f['Grid']['T'][0]
    riph = f['Grid']['r_jph'][...]
    ziph = f['Grid']['z_kph'][...]

    r = np.zeros(piph.shape)
    z = np.zeros(piph.shape)
    phi = np.zeros(piph.shape)
    R = 0.5*(riph[1:] + riph[:-1])
    Z = 0.5*(ziph[1:] + ziph[:-1])
    primPhi0 = np.zeros((index.shape[0], index.shape[1], prim.shape[1]))
    for k in range(index.shape[0]):
        for j in range(index.shape[1]):
            ind0 = index[k,j]
            ind1 = ind0 + nphi[k,j]
            r[ind0:ind1] = R[j]
            z[ind0:ind1] = Z[k]
            piph_strip = piph[ind0:ind1]
            pimh = np.roll(piph_strip, 1)
            pimh[pimh>piph_strip] -= 2*np.pi
            phi[ind0:ind1] = 0.5*(pimh+piph_strip)
            primPhi0[k,j,:] = prim[idPhi0[k,j],:]

    return t, r, phi, z, prim #, (riph, ziph, primPhi0, piph)










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

    t,torque,torque_hill,r2,p2,mdot,jdot,fr = np.genfromtxt(filename,unpack=True,usecols=(0,1,2,4,5,6,7,8))
    orb = t/(2.*np.pi)


    # get rid of overlap from simualtion restarts
    uniq_orb, ind = np.unique(orb,return_index=True)
    orb = orb[ind]
    r2 = r2[ind]
    p2 = p2[ind]
    fr = fr[ind]
    torque = torque[ind]
    torque_hill = torque_hill[ind]
    mdot = mdot[ind]
    jdot = jdot[ind]


    return orb, torque, torque_hill, r2, p2, mdot, jdot, fr


def loadreport_old(filename):
    # old runs:
    t,torque,torque_halfeps,torque_eps,torque_hill,r2,p2,mdot,jdot,fr = np.genfromtxt(filename,unpack=True,usecols=(0,1,2,3,4,6,7,8,9,10))
    orb = t/(2.*np.pi)

    return orb, torque, torque_halfeps, torque_eps, torque_hill, r2, p2, mdot, jdot, fr







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
    up = 2.
    low = -2.

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

    #phi_ax = np.linspace(0.,2.*np.pi,len(xi))
    Zi = griddata((x_rot,y_rot),data,(Xi,Yi))
    #Zi = griddata((xi,phi_ax), data, (Xi,Yi))


    # return the ROTATED interpolated Z data
    print 'Xi,Yi = ',np.shape(Xi),np.shape(Yi)

    return Xi,Yi,Zi



def vel_mag(file):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)
    dens = prim[:,0]
    velp = prim[:,3]
    velr = prim[:,2]

    x = np.multiply(r,np.cos(phi)) 
    y = np.multiply(r,np.sin(phi))

    # position of planet to rotate data
    phi_p = plan[1,4]
    # and convert to corotating frame
    r_p = plan[1,3]

    # subtract global keplerian velocity at planet radius
    velp_co = velp - (1./r_p)**0.5

    # get magnitude of velocity
    # polar coordinates are orthonormal
    vel_mag_co = (velr**2 + velp_co**2)**0.5

    # Make a data array that gives distance from gas to planet 
    dx = r*np.cos(phi) - r_p*np.cos(phi_p)
    dy = r*np.sin(phi) - r_p*np.sin(phi_p)
    # distance from fluid element to planet
    script_r = np.sqrt(dx**2 + dy**2)

    # Cut data to region around planet
    # Can't get that to work for some reason,
    # so for now I am setting all the data to 0 except
    # for a box around the planet. 
    gap_cut = np.where((x < (r_p - 0.25)) | (x>(r_p+0.25)) | (abs(y)>0.25))[0]
    #index_gap = np.where(r >1.)

    # Cut out a region around the planet
    #vel_mag_co_cut = vel_mag_co[index_gap]
    #r_cut = r[index_gap]
    #phi_cut = r[index_gap]
    vel_mag_co[gap_cut] = 0.

    #x_cut = np.multiply(r_cut,np.cos(phi_cut)) 
    #y_cut = np.multiply(r_cut,np.sin(phi_cut))


    triang = tri.Triangulation(x, y)



    #x,y,vmag2d = xy_interp(r,phi,vel_mag_co,phi_p,100)

    plt.figure()
    #plt.pcolormesh(x,y,vmag2d,cmap=cmaps.magma,vmin=0.0,vmax=0.3)
    #plt.axis([x.min(), x.max(), y.min(), y.max()])

    plt.tricontourf(triang,vel_mag_co,50,cmap=cmaps.magma,vmin=0.0,vmax=0.4)
    plt.colorbar()
    plt.show()

    #plt.figure()
    #plt.contourf(x,y,vmag2d,50,cmap=cmaps.magma,vmin=0.0,vmax=0.3)
    #plt.colorbar()
    #plt.show()



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
    res = 100
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



def vel_field(file):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)
    dens = prim[:,0]
    velp = prim[:,3]
    velr = prim[:,2]

    # position of planet to rotate data
    phi_p2 = plan[1,4]
    r_p2 = plan[1,3]
    m_p2 = plan[1,0]
    om_p2 = plan[1,2]

    # subtract keplerian velocity of gas at planet radius
    print 'rp, vphi(rp) = ',r_p2, (1./r_p2)**0.5
    velp_co = velp - (1./r_p2)**0.5

    # convert polar coordinates to cartesian 
    # (before or after interpolation?) <------ TRY BOTH
    velx = velr*np.cos(phi) - velp_co*np.sin(phi)
    vely = velr*np.sin(phi) + velp_co*np.cos(phi)

    res = 512
    vel_res = 110
    # convert this data to 2d data in x,y
    x_ax, y_ax, dens_2d = xy_interp(r,phi,dens,phi_p2,res)
    x_ax2, y_ax2, velx_2d = xy_interp(r,phi,velx,phi_p2,vel_res)
    x_ax2, y_ax2, vely_2d = xy_interp(r,phi,vely,phi_p2,vel_res)

    # cut out the middle region
    r_ax2 = (x_ax2**2 + y_ax2**2)**0.5
    r_ax = (x_ax**2 + y_ax**2)**0.5
    velx_2d[np.where(r_ax2 < np.min(r))] = np.nan
    vely_2d[np.where(r_ax2 < np.min(r))] = np.nan
    dens_2d[np.where(r_ax < np.min(r))] = np.nan

    # Normalize the velocities
    # Normalize the arrows:
    #U = velx_2d *10 #/ np.sqrt(velx_2d**2 + vely_2d**2);
    #V = vely_2d *10 #/ np.sqrt(velx_2d**2 + vely_2d**2);
    U = velx_2d *5#/ np.sqrt(velx_2d**2 + vely_2d**2);
    V = vely_2d *5#/ np.sqrt(velx_2d**2 + vely_2d**2);

    r_hill = (m_p2/3)**(1./3)*r_p2
    eps = plan[1,5]

    fig=plt.figure()
    cnt = plt.contourf(x_ax,y_ax,np.log10(dens_2d),100,cmap='magma_r', rasterized=True)
    plt.clim(vmin=-1.8,vmax=3.0)
    cbar = fig.colorbar(cnt, ticks=[-1, 0, 1, 2, 3],label=r'$\Sigma$')
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'])
    
    #9FF781
    plt.quiver(x_ax2,y_ax2,U,V,pivot='mid',scale=6, scale_units='inches',color='#2D1F4D',width=.0035)

    #plt.streamplot(x_ax2,y_ax2,U,V,color='k',density=100) ## this is slow

    plt.scatter([r_p2],[0.0],color='k',marker='x',s=50,linewidth=2,alpha=0.7)
    fig = plt.gcf()
    ax = fig.gca()
    hillsph = plt.Circle((r_p2, 0.), r_hill, \
         fill=False, color='white',linewidth=1.4,linestyle='dashed',alpha=0.8)
    smooth = plt.Circle((r_p2, 0.), eps, \
         fill=False, color='orange',linewidth=2,alpha=0.7)
    ax.add_artist(hillsph)
    #ax.add_artist(smooth)

    plt.xlabel(r'$x \, [r_{0}]$')
    plt.ylabel(r'$y \, [r_{0}]$')
    plt.xlim([r_p2-0.25,r_p2+0.25])
    plt.ylim([-0.25,0.25])
    # plt.xlim([r_p2-0.5,r_p2+0.5])
    # plt.ylim([-0.5,0.5])

    plt.show()





def torq_cont(file,mach,f_rH):

    ## ***** try cutting out the smoothing length, since we
    # don't resovle the flow within that region. 
    # TIME-AVERAGE! 


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

    q = 1.e-3
    sigma_0 = 1.
    om2 = 1./r_p2**(3./2)
    T_0 =  r_p2**3 * om2**2 * sigma_0 * q**2 * mach**4

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
    # NORMALIZED
    torq = -dens*(f_phi2*r_p2) / T_0


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


    fig = plt.figure(figsize=(8,6))
    #contour_neg = -np.linspace(1,(np.max(torq)),50)[::-1]
    #contour_pos = np.linspace(1,(np.max(torq)),50)
    #   contour_levels = np.linspace(-0.995*np.max(abs(torq/np.max(torq))),0.995*np.max(abs(torq/np.max(torq))),60)
    contour_levels = np.linspace(-1.*np.max(abs(torq/np.max(torq))),1.*np.max(abs(torq/np.max(torq))),60)

    #contour_levels = np.append(contour_neg,contour_pos)
    cnt = plt.tricontourf(triang,torq/np.max(torq),contour_levels,\
        cmap=cmap_rdbu, rasterized=True)


    #cbar = fig.colorbar(cnt, ticks=[-0.6,-0.4,-0.2,0.,0.2,0.4,0.6],label=r'$T_{\rm code}$')
    #cbar = fig.colorbar(cnt,ticks=[-10000.,-5000., 0., 5000., 10000.],label=r'$T_{\rm code}$')
    cbar = fig.colorbar(cnt,label=r'$T_{\rm code}$',ticks=[-1.0,-0.5,0.,0.5,1.0])
    #cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'])
    #plt.clim(vmin=-2.1,vmax=2.1)

    #plt.xlim([r_p2-0.25,r_p2+0.25])
    #plt.ylim([-0.25,0.25])
    #plt.xlim([r_p2-0.05,r_p2+0.05])
    #plt.ylim([-0.05,0.05])
    plt.xlim([0.5,1.5])
    plt.ylim([-0.5,0.5])
    plt.xlabel(r'$x \, [r_0]$')
    plt.ylabel(r'$y \, [r_0]$')
    #plt.title(r'$\rm Torque \, density$')


    # rotated secondary position:
    plt.plot(x_plan2_rot,0.0,linestyle='none',marker='x',
            color='black',markersize=7,markeredgewidth=1)
    

    # Plot circles for the hill sphere, or planets
    fig = plt.gcf()
    ax = fig.gca()
    hillsph = plt.Circle((x_plan2_rot, y_plan2_rot), r_hill, \
         fill=False, color='k',linestyle='dashed')
    smooth = plt.Circle((x_plan2_rot, y_plan2_rot), eps2, \
         fill=False, color='orange',alpha=0.7)
    ax.add_artist(smooth)
    ax.add_artist(hillsph)

    cont_lines = np.linspace(-0.1*np.max(torq),0.1*np.max(torq),2)
    clines = plt.tricontour(triang,torq,cont_lines,alpha=0.7,colors='k',linewidths=0.75)
    cont_lines2 = np.linspace(-0.4*np.max(torq),0.4*np.max(torq),2)
    clines2 = plt.tricontour(triang,torq,cont_lines2,alpha=0.7,colors='k',linewidths=1.5)\

    #cont_lines = np.linspace(-0.14*np.max(torq),0.14*np.max(torq),2)
    #clines = plt.tricontour(triang,torq,cont_lines,alpha=0.7,colors='k',linewidths=0.75)
    #cont_lines2 = np.linspace(-0.37*np.max(torq),0.37*np.max(torq),2)
    #clines2 = plt.tricontour(triang,torq,cont_lines2,alpha=0.7,colors='k',linewidths=1.5)


    # This is the fix for the white lines between contour levels
    #for c in cnt.collections:
    #    c.set_edgecolor("face")

    #plt.tight_layout()
    plt.show()

    return triang, torq


def get_the_torq(file,f_rH):
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

    # Cut out a region around the planet
    # Make a data array that gives distance from gas to planet 
    dx = r*np.cos(phi) - r_p2*np.cos(phi_p2)
    dy = r*np.sin(phi) - r_p2*np.sin(phi_p2)
    # distance from fluid element to planet
    script_r = np.sqrt(dx**2 + dy**2)

    r_hill = (1e-3/3.)**(1./3)*r_p2

    cut_indx = np.where(script_r < f_rH*r_hill)
    torq[cut_indx] = 0.


    return r, phi, r_p2, phi_p2, torq


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



def data_avg(dir,f_rH):

    # this isn't working YET because data needs to be averaged with the 2D interpolation
    # and I want to specify quantities with the 1D array...

    files = []
    for f in os.listdir(dir):
      if os.path.isfile(os.path.join(dir,f)) and f.startswith('checkpoint'):
        files = np.append(files,dir+f)

    print 'averaging over: \n', files

    # Needs to be interpolated to 2D array to average
    # general grid info
    rmax = 2.0
    rmin = -2.0
    res = 1000
    # Making a new grid
    # this will be the shape of the interpolated 2d output data array
    xi = np.linspace(rmin,rmax,res)
    yi = np.linspace(rmin,rmax,res)
    #data_inner_sum = np.zeros((res,res))
    #data_outer_sum = np.zeros((res,res))
    data_sum = np.zeros((res,res))


    for i in range(0,len(files)):
        r, phi, r_p2, phi_p2, torq_1d = get_the_torq(files[i],f_rH)
        data_1d = torq_1d
        # FIRST let's do this with the density!
        #t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(files[i])
        #data_1d = prim[:,0]
        #r_p2 = plan[1,3]
        #phi_p2 = plan[1,4]
        # brake up torq from 1D array, then sum/average from 2D grid
        # Specify a region around the BH to define torq inner and torq outer
        # Make a data array that gives distance from gas to planet 
        #dx = r*np.cos(phi) - r_p2*np.cos(phi_p2)
        #dy = r*np.sin(phi) - r_p2*np.sin(phi_p2)
        # get distance from fluid element to planet
        #script_r = np.sqrt(dx**2 + dy**2)
        # we define regions by inside/outside the Hill radius 
        #r_hill = (1e-3/3.)**(1./3)*r_p2
        #torq_inner_1d = torq_1d
        #torq_outer_1d = torq_1d

        # lotsa ins and outs
        #cut_inner = np.where(script_r < r_hill)[0]
        #torq_outer_1d[cut_inner] = 0.
        #cut_outer = np.where(script_r > r_hill)[0]
        #torq_inner_1d[cut_outer] = 0.

        # interpolate the 1D array to a 2D array
        x_axis, y_axis, data_2d = xy_interp(r,phi,data_1d,phi_p2,res)
        #x_axis, y_axis, data_inner_2d = xy_interp(r,phi,torq_inner_1d,phi_p2,res)
        #x_axis, y_axis, data_outer_2d = xy_interp(r,phi,torq_outer_1d,phi_p2,res)
        #print np.max(data_inner_2d[np.where(data_inner_2d == data_inner_2d)])
        #print np.max(data_outer_2d[np.where(data_outer_2d == data_outer_2d)])

        data_sum += data_2d
        #data_inner_sum += data_inner_2d
        #data_outer_sum += data_outer_2d


    # divide by number of files
    data_avg = data_sum / len(files)
    #data_inner_avg = data_inner_sum / len(files)
    #data_outer_avg = data_outer_sum / len(files)
    #print np.max(data_inner_avg[np.where(data_inner_avg == data_inner_avg)])
    #print np.max(data_outer_avg[np.where(data_outer_avg == data_outer_avg)])

    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(files[-1])
    eps = plan[1,5]
    r_p2 = plan[1,3]
    r_hill = (1e-3/3)**(1./3) * r_p2

    plt.figure()
    contour_levels = np.linspace(-1.0*np.nanmax(abs(data_avg)),np.nanmax(abs(data_avg)),50)
    plt.contourf(x_axis,y_axis,data_avg,contour_levels,cmap='RdBu', rasterized=True)
    plt.colorbar()

    r0 = 1.
    plt.xlim([r_p2-r0/2.,r_p2+r0/2.])
    plt.ylim([-r0/2.,r0/2.])
    plt.xlabel(r'$x \, [r_{0}]$')
    plt.ylabel(r'$y \, [r_{0}]$')


    fig = plt.gcf()
    ax = fig.gca()
    hillsph = plt.Circle((r_p2, 0.), r_hill, \
         fill=False, color='gray',linestyle='dashed')
    smooth = plt.Circle((r_p2, 0.), eps, \
         fill=False, color='orange',alpha=0.7)
    ax.add_artist(hillsph)
    ax.add_artist(smooth)

    plt.show()

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





def zoom_cont(file):
    t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)


    dens = prim[:,0]
    pres = prim[:,1]
    velr = prim[:,2]
    vphi = prim[:,3]
    #velz = prim[:,4]
    orb = t/2./np.pi

    # get data for resolution
    r_cells = r[index[:,0]]
    # something for theta?
    theta = np.linspace(0,2*np.pi,300)
    radius_matrix, theta_matrix = np.meshgrid(r_cells,theta)
    x_grid = radius_matrix * np.cos(theta_matrix)
    y_grid = radius_matrix * np.sin(theta_matrix)

    # distance scale factor
    r0 = 1. # Rs


    # where is the planet?
    r2 = plan[1,3] * r0
    eps = plan[1,5] * r0
    phi2 = plan[1,4] # this is in radians
    x_plan2 = np.multiply(r2,np.cos(phi2))
    y_plan2 = np.multiply(r2,np.sin(phi2)) 

    x = np.multiply(r,np.cos(phi)) * r0
    y = np.multiply(r,np.sin(phi)) * r0
    velx = np.multiply(velr,np.cos(vphi))
    vely = np.multiply(velr,np.sin(vphi))

    # Hill radius
    r_hill = (1e-3/3)**(1./3)*r2 * r0

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
    min_radius = 1.01*np.min(r) * r0 
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
    triang.set_mask(mask)

    #plt.pcolormesh(x,y,np.log10(dens),cmap=cmaps.inferno)

    #contour_levels =np.exp(np.linspace(math.log(1e-3),math.log(5e2),200))

    fig=plt.figure()
    cnt = plt.tricontourf(triang,np.log10(dens),100,cmap='magma_r', rasterized=True)
    #plt.clim(vmin=-1.,vmax=1.)


    # plot radial grid boundaries
    plt.plot(x_grid,y_grid,'k',linewidth=0.5)
    # one row of horizontal lines just to approx cell position
    #plt.axhline(y=0.002, xmin=0, xmax=10, linewidth=0.5,color='k')
    #plt.axhline(y=-0.002, xmin=0, xmax=10, linewidth=0.5,color='k')


    cbar = fig.colorbar(cnt, ticks=[-1, 0, 1, 2, 3],label=r'$\Sigma$')
    cbar.ax.set_yticklabels([r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$'])

    # This is the fix for the white lines between contour levels
    for c in cnt.collections:
       c.set_edgecolor("face")

    #cont_lines = np.linspace(-0.7*np.max(prim[:,0]),0.7*np.max(prim[:,0]),4)
    #clines = plt.tricontour(triang,np.log10(dens),cont_lines,alpha=0.7,colors='white',linewidth=1)


    plt.xlim([x_plan2_rot-0.25,x_plan2_rot+0.25])
    plt.ylim([y_plan2_rot-0.25,y_plan2_rot+0.25])
    #plt.xlim([-2.8,2.8])
    #plt.ylim([-2.8,2.8])
    plt.xlabel(r'$x \, [r_{0}]$')
    plt.ylabel(r'$y \, [r_{0}]$')

    #plt.scatter([1.],[0.],color='k',marker='x',s=30)

    fig = plt.gcf()
    ax = fig.gca()
    hillsph = plt.Circle((x_plan2_rot, y_plan2_rot), r_hill, \
         fill=False, color='#FF3850',linestyle='--',alpha=0.9,linewidth=1)
    smooth = plt.Circle((x_plan2_rot, y_plan2_rot), eps, \
         fill=False, color='white',linestyle='dotted',alpha=0.7)

    print 'rhill boundaries in rS = ', 5.*(-r_hill), 5.*(+r_hill)
    print 'eps boundaries in rS = ', 5.*(-eps), 5.*(+eps)
    
    ax.add_artist(smooth)
    ax.add_artist(hillsph)

    plt.tight_layout()

    plt.show()
    return triang, dens





def torq_movie(dir):

    rep_file = dir+'report.dat'
    #check_file = dir + 'checkpoint_%s.h5'%chk_num
    #orb, torque, torque_cut, r2, p2, mdot2, jdot2 = loadreport(rep_file)

    t,smoothed,torque2 = np.genfromtxt('%s/smoothed.txt'%dir,unpack=True)
    orb = t/2./np.pi

    #window = int(orb[-1]-orb[0])/2
    #print 'window = ',window
    #poly_order = 3
    #smoothed = savitzky_golay(torque/T_0, window, poly_order)
    #smoothed = signal.medfilt(torque[cut:]/T_0,1115)
    #smoothed = pd.rolling_mean(torque,window)

    mach = 20.
    q = 1.e-3

    sigma_0 = 1.

    # type 1 torque:
    # T0 is an array of length len(total torque)
    # (dont have updated omega in repfile, but we have r_p)
    # Still in simulation units.
    # take last value
    #om2 = 1./r2**(3./2)
    #T_0 =  r2**4 * om2**2 * sigma_0 * q**2 * mach**2 
    T_0 =  sigma_0 * q**2 * mach**2 

    x = orb
    y = smoothed
    size = np.random.randint(150, size=2)
    colors = np.random.choice(["r", "g", "b"], size=2)

    fig = plt.figure()
    plt.xlabel(r'$t \, \rm [orb]$', fontsize=25, fontweight='bold')
    plt.ylabel(r'$ $', fontsize=25, fontweight='bold')
    plt.xlim([orb[120],orb[-1]])
    plt.ylim([-0.3,1.1*np.max(smoothed/T_0)])
    plt.axhline(y=0., xmin=0, xmax=1300, linewidth=0.5, linestyle='--',color='k')

    start =  np.where(smoothed>0.0)[0][0]

    for i in range(start-10,len(smoothed),100):
        print i
        plt.plot(orb[120:i],smoothed[120:i]/T_0,linewidth=3,color='darkred')
        plt.savefig('/Users/Shark/Dropbox/BHBH/IMRI_talk/TAC_seminar_Mar2018/torq_movie/torq_%i'%i)




def torqfit(dir, alpha):
    rep_file = dir+'report.dat'
    
    orb, torque, torque_halfeps, torque_eps, torque_hill, r2, p2, mdot2, jdot2, fr = loadreport(rep_file)

    # scale r to schwarszchild radii
    rscale = 5.

    # CONVERT TO PHYSICAL UNITS
    G = 6.67e-8
    c = 2.99e10
    Msun = 1.98e33
    M = 1.e6*Msun
    r0 = rscale*2*G*M/c**2

    # This changes with accretion rate and alpha
    #sigma_0 = 9.7e7
    sigma_0 = 1.

    # a is cgs distance (cm)
    a = r2*r0
    # omega in 1/s
    om2 = (G*M)**0.5 * a**(-1.5)

    mach = 20.
    q = 1.e-3  #!!! q will change!!
 

    # CUT DATA from beginning out for smoothing
    cut = np.where(orb>4500)[0][0]

    r2 = r2[cut:]
    acut = a[cut:]
    om2 = om2[cut:]
    torque = torque[cut:]
    torque_hill = torque_hill[cut:]
    jdot2 = jdot2[cut:]

    torque_inner = np.subtract(torque,torque_hill)

    #print len(np.where(np.isnan(torque_inner) == True)[0])

    # smooth inner torque

    #smoothed_inner = pd.rolling_mean(torque_inner,window=200) 
    smoothed_inner = savgol_filter(torque_inner, 1001, 3)
    smoothed_hill = pd.rolling_mean(torque_hill,window=300)

    print len(np.where(np.isnan(smoothed_inner) == True)[0])


    # trim ends from savitzky golay
    trim_in = 50
    trim_out = 50


    smoothed_inner = smoothed_inner[100:-100]
    smoothed_hill = smoothed_hill[100:-100]
    r2 = r2[100:-100]
    acut = acut[100:-100]
    om2 = om2[100:-100]
    torque = torque[100:-100]
    torque_hill = torque_hill[100:-100]


    # Scale torque to cgs units
    T_scale = G*M*sigma_0*r0


    # Density has radial dependence  
    sigma = sigma_0 * (r2)**(-0.5)

    # VISCOUS TORQUE
    torq_nu =  3*np.pi*acut**4 * om2**2 * sigma * alpha * 1./mach**2


    # GAS TORQUE to CGS UNITS
    torq_totalcgs = torque*T_scale
    #torq_outer = smoothed_hill# *T_scale
    # Dont smooth the outer torque
    torq_outercgs = torque_hill * T_scale
    torq_innercgs = smoothed_inner * T_scale

    # TORQUE IN CODE UNITS FOR FITTING 
    torq_total = torque
    #torq_outer = smoothed_hill#
    # Dont smooth the outer torque
    torq_outer = torque_hill
    torq_inner = smoothed_inner


    # FLATTEN BEFORE FIT
    #indx = np.where(rscale*r2 < 5.8)[0][0]
    #print indx
    #torq_inner[indx:] = torq_inner[indx]

    #
    # FIT lines/curves to the torques
    # Output is in the form y = mx + b  where fit[1] is b and fit[0] is m
    # 

    # The x axis is in cgs units for both fits.
    x = acut 

    #-----------

    fit1cgs = np.polyfit(x,torq_outercgs,0)
    torq_out_fitcgs = np.zeros_like(x)
    torq_out_fitcgs[:] = fit1cgs[0]
    print 'cgs: fit outer = ',fit1cgs

    fit2cgs = np.polyfit(x[:-1200],torq_innercgs[:-1200],2)
    torq_in_fitcgs = np.zeros_like(x)
    torq_in_fitcgs[:] = fit2cgs[0]*x*x + fit2cgs[1]*x + fit2cgs[2]
    print 'cgs: fit inner = ',fit2cgs

    cutfitcgs = np.argmin(torq_in_fitcgs)
    torq_in_fitcgs[cutfitcgs:] = torq_in_fitcgs[cutfitcgs]
    print 'cgs: break radius in rS = ',r2[cutfitcgs] * rscale
    print 'cgs: inner torque after break = ', torq_in_fitcgs[cutfitcgs]

    print ' '
    print 'cgs: Cfit+Dfit = ',fit2cgs[2]+fit1cgs

    print ' '

    #-----------

    fit1 = np.polyfit(x,torq_outer,0)
    torq_out_fit = np.zeros_like(x)
    torq_out_fit[:] = fit1[0]
    print 'fit outer = ',fit1

    # fit a polynomial to the inner torque, cut out the last bit **play with cut index**
    fit2 = np.polyfit(x[:-1200],torq_inner[:-1200],2)
    torq_in_fit = np.zeros_like(x)
    torq_in_fit[:] = fit2[0]*x*x + fit2[1]*x + fit2[2]
    print 'fit inner = ',fit2

    cutfit = np.argmin(torq_in_fit)
    torq_in_fit[cutfit:] = torq_in_fit[cutfit]
    print 'break radius in rS = ',r2[cutfit] * rscale
    print 'inner torque after break = ', torq_in_fit[cutfit]

    # Convert code fits to cgs??? not sure
    #print 'A cgs = ', fit2[0]*T_scale/r0/r0
    #print 'B cgs = ', fit2[1]*T_scale/r0
    #print 'C+D cgs = ',(fit2[2]+fit1)*T_scale
    #-----------
    print ' '
    print 'Cfit+Dfit = ',fit2[2]+fit1



    new_tick_locations = np.array([10.,9.,8.,7.,6.,5.])

    def tick_function(X):
        rX = X*2.*G*M/c**2
        ff = (G*M/rX**3)**(0.5) *1.e3 / np.pi  #obs f_gw at z=1 in mHz
        return [r"$%.1f$" % z for z in ff]

    # !!!!!!
    # REMEMBER that we want to plot in cgs, but we want the fit in code units
    # for the mathematica delta phi integral (it's scaled there)


    # Plot Torque in cgs for paper:

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    #ax1.plot(r2*rscale, torq_outer*T_scale/torq_nu, label=r'${\rm outside} \, r_{\rm H}$',color='#12A9A9',alpha=0.8,linewidth=3)
    #ax1.plot(r2*rscale, torq_out_fit*T_scale/torq_nu, linestyle='--',color='k',alpha=0.8,label=r'$\rm fit $',linewidth=2,dashes=(22, 8))
    #ax1.plot(r2*rscale, torq_inner*T_scale/torq_nu, label=r'${\rm inside} \, r_{\rm H}$',color='#A91212',alpha=0.8,linewidth=3)
    #ax1.plot(r2*rscale, torq_in_fit*T_scale/torq_nu, linestyle='--',color='k',alpha=0.8,linewidth=2,dashes=(22, 8))
    
    ax1.plot(r2*rscale, torq_outercgs/torq_nu, label=r'${\rm outside} \, r_{\rm H}$',color='#12A9A9',alpha=0.8,linewidth=3)
    ax1.plot(r2*rscale, torq_out_fitcgs/torq_nu, linestyle='--',color='k',alpha=0.8,label=r'$\rm fit $',linewidth=2,dashes=(22, 8))
    ax1.plot(r2*rscale, torq_innercgs/torq_nu, label=r'${\rm inside} \, r_{\rm H}$',color='#A91212',alpha=0.8,linewidth=3)
    ax1.plot(r2*rscale, torq_in_fitcgs/torq_nu, linestyle='--',color='k',alpha=0.8,linewidth=2,dashes=(22, 8))

    ax1.axhline(y=0., xmin=0, xmax=orb[-1], linewidth=2, linestyle='--',alpha=0.5,color='k',dashes=(5, 4))
    plt.legend()

    ax1.set_xlim([10.4,5.])
    ax1.set_xlabel(r'$r \, \rm [r_{\rm S}]$')

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"$f_{\rm obs} \, \rm [mHz]$")

    ax1.set_ylabel(r'$|T_g|/ T_{\nu}$')
    plt.tight_layout()



    # plot in code units:
    #ax1.plot(r2*rscale, torq_outer, label=r'${\rm outside} \, r_{\rm H}$',color='#12A9A9',alpha=0.8,linewidth=3)
    #ax1.plot(r2*rscale, torq_out_fit, linestyle='--',color='k',alpha=0.8,label=r'$\rm fit $',linewidth=2,dashes=(22, 8))
    #ax1.plot(r2*rscale, torq_inner, label=r'${\rm inside} \, r_{\rm H}$',color='#A91212',alpha=0.8,linewidth=3)
    #ax1.plot(r2*rscale, torq_in_fit, linestyle='--',color='k',alpha=0.8,linewidth=2,dashes=(22, 8))
    #ax1.axhline(y=0., xmin=0, xmax=orb[-1], linewidth=2, linestyle='--',alpha=0.5,color='k',dashes=(5, 4))


    #ax1.set_xlim([10.4,5.])
    #ax1.set_xlabel(r'$r \, \rm [r_{\rm S}]$')

    #ax1.set_ylabel(r'$T_{\rm code}$')
    #plt.tight_layout()

    return r2, torq_inner, torq_outer





def plottorque(window,dir,q,mach,w):
    rep_file = dir+'report.dat'

    #old files!   
    #orb, torque, torque_halfeps, torque_eps, torque_hill, r2, p2, mdot2, jdot2,fr = loadreport_old(rep_file)
    orb, torque, torque_hill, r2, p2, mdot2, jdot2, fr2 = loadreport(rep_file)

    #mach = 20.

    sigma_0 = 1.

    # scale r to schwarszchild radii
    # Get r0 scaling in Rs for the given migraiton rate w and mass ratio q
    rscale = (-1./w * 8./5 * 2**0.5 * 1./((1.+1./q)*(1.+q)))**(2./5)

    # type 1 torque:
    # T0 is an array of length len(total torque)
    # (dont have updated omega in repfile, but we have r_p)
    # Still in simulation units.
    om2 = 1./r2**(3./2)
    # Density has radial dependence too -- 

    sigma = sigma_0 * r2**(-0.5)
    T_0 =  r2**4 * om2**2 * sigma * q**2 * mach**2 

    # viscous torque:
    #T_nu =  3*np.pi*r2**4 * om2**2 * sigma * alpha * 1./mach**2

    # cut data from beginning out for smoothing
    # provide index to cut out till
    cut = np.where(orb > 1000)[0][0]
    torque = torque[cut:]
    torque_hill = torque_hill[cut:]
    T_0 = T_0[cut:]
    r2 = r2[cut:]
    om2 = om2[cut:]
    orb = orb[cut:]

    #window = int(orb[-1]-orb[0])/2
    #poly_order = 3
    #smoothed = savitzky_golay(torque/T_0, window, poly_order)
    #smoothed = pd.rolling_mean(torque,window) 
    #smoothed_hill = pd.rolling_mean(torque_hill,window)
    #jdot_s = pd.rolling_mean(jdot2/mdot2,window)
    smoothed = pd.Series(torque).rolling(window).mean()
    smoothed_hill = pd.Series(torque_hill).rolling(window).mean()
    jdot_s = pd.Series(jdot2/mdot2).rolling(window).mean()

    #cut_ind = np.where(orb>100)[0][0]


    #plot as a function of adot/(omega*a)
    adot_omegaa = w*r2**(-5./2.)

    # CONVERT TO PHYSICAL UNITS
    G = 6.67e-8
    c = 2.99e10
    Msun = 1.98e33
    M = 1.e6*Msun
    r0 = rscale*2*G*M/c**2
    

   
    C = 64./5. * G**3/c**5 * (M+q*M)**3/(1.+1./q)/(1.+q)

    # Get separation in cgs units for GW torque
    # Simulations with small q will correspond to inside ISCO separation
    # so GW torque is not directly meaningful
    a = r2*r0
    adot = C/(a)**3

    torq_gw = 0.5*q*M *om2*(r2*r0)* adot

    # Compare GW torque
    # torque total

    rax = r2*rscale

    #dfig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax2 = ax1.twiny()

    #ax1.plot(rax,smoothed/T_0)
    #ax1.plot(rax,smoothed_hill/T_0)
    #ax1.plot(rax,T_nu/T_0)
    #ax1.plot(rax,torq_gw/T_0)
    #ax1.plot(rax,)
    #ax2.plot(adot/a,smoothed/T_0,alpha=0.9)
    #ax2.set_xlim(ax1.get_xlim())
    #ax2.set_xticks(rax*2)
    #plt.axhline(y=0., xmin=0, xmax=orb[-1], linewidth=0.5, linestyle='--',color='k')

    #plt.plot(orb[cut_ind:],smoothed[cut_ind:]/T_0[cut_ind:],color='#B22C10',alpha=1.0,linewidth=2,label=r'$\rm total$')
    #plt.plot(orb[cut_ind:],smoothed_hill[cut_ind:]/T_0[cut_ind:],linewidth=2,label=r'$\rm cut \, r_{hill}$')
    #plt.legend()
    #plt.xlabel(r'$t \, \rm [orb]$')
    #plt.ylabel(r'$\rm T/T_0$')
    #plt.ylim([-1,1])

    #plt.annotate(r'$\mathcal{M} = %i$'%mach,xy=(500,0.07))
    #plt.annotate(r'$\dot{a}/\Omega a = %.1e$'%drift,xy=(500,0.045))
    #plt.annotate(r'$T/T_0 = %.2f$'%avg_torq,xy=(500,0.02))

    #plt.tight_layout()
    #plt.show()


    cutnans = -1
    smoothed = smoothed[:cutnans]
    smoothed_hill = smoothed_hill[:cutnans]
    T_0 = T_0[:cutnans]
    adot_omegaa = adot_omegaa[:cutnans]
    r2 = r2[:cutnans]
    adot = adot[:cutnans]
    om2 = om2[:cutnans]
    a = a[:cutnans]
    orb = orb[:cutnans]

    return orb, r2, adot_omegaa, smoothed, smoothed_hill, T_0
    #return r2, adot_omegaa, smoothed, smoothed_hill, T_0, adot, om2, a
    #return orb, adot_omegaa, smoothed, smoothed_eps, jdot_s, T_0, torq_gas, torq_gw, adot




def cf_torques(dir1, dir2):
    rep_file1 = dir1+'report.dat'
    rep_file2 = dir2+'report.dat'

    orb1, torque1, torque_halfeps1, torque_eps1, torque_hill1, r21, p21, mdot21, jdot21, fr1 = loadreport(rep_file1)
    orb2, torque2, torque_halfeps2, torque_eps2, torque_hill2, r22, p22, mdot22, jdot22, fr2 = loadreport(rep_file2)



    # scale r to schwarszchild radii
    rscale = 5.

    # CONVERT TO PHYSICAL UNITS
    G = 6.67e-8
    c = 2.99e10
    Msun = 1.98e33
    M = 1.e6*Msun
    r0 = rscale*G*M/c**2
    sigma_0 = 6.2e6

    # a is cgs distance (cm)
    a1 = r21*r0
    a2 = r22*r0
    # omega in 1/s
    om21 = (G*M)**0.5 * a1**(-1.5)
    om22 = (G*M)**0.5 * a2**(-1.5)

    mach = 20.
    q = 1.e-3
 
    alpha1 = 0.03
    alpha2 = 0.01


    # CUT DATA from beginning out for smoothing
    cut = np.where(orb1>2000)[0][0]

    r21 = r21[cut:]
    a1 = a1[cut:]
    om21 = om21[cut:]
    torque1 = torque1[cut:]
    torque_hill1 = torque_hill1[cut:]

    r22 = r22[cut:]
    a2 = a2[cut:]
    om22 = om22[cut:]
    torque2 = torque2[cut:]
    torque_hill2 = torque_hill2[cut:]


    torque_inner1 = np.subtract(torque1,torque_hill1)
    torque_inner2 = np.subtract(torque2,torque_hill2)

    smoothed_inner1 = pd.rolling_mean(torque_inner1,window=1000) 
    smoothed_inner2 = pd.rolling_mean(torque_inner2,window=1000) 

    smoothed_outer1 = pd.rolling_mean(torque_hill1,window=300) 
    smoothed_outer2 = pd.rolling_mean(torque_hill2,window=350) 


    # Scale torque to cgs units
    T_scale = G*M*sigma_0*r0


    # Density has radial dependence  
    sigma1 = sigma_0 * (r21)**(-0.5)
    sigma2 = sigma_0 * (r22)**(-0.5)

    # TYPE I TORQUE (SCALING)
    # compare for 1e-5 mass ratio
    #T_0 =  a1**4 * om21**2 * sigma1 * (1.e-5)**2 * mach**2 
    #WHY did I do that??:****

    T_01 =  a1**4 * om21**2 * sigma1 * (1.e-5)**2 * mach**2 
    T_02 =  a2**4 * om22**2 * sigma2 * (1.e-5)**2 * mach**2 

    # VISCOUS TORQUE
    torq_nu1 =  3*np.pi * a1**4 * om21**2 * sigma1 * alpha1 * 1./mach**2
    torq_nu2 =  3*np.pi * a2**4 * om22**2 * sigma2 * alpha2 * 1./mach**2


    # GAS TORQUE
    torq_inner1 = smoothed_inner1 
    torq_inner2 = smoothed_inner2 
    torq_hillcut1 = smoothed_outer1 
    torq_hillcut2 = smoothed_outer2 



    # plot by schwarzchild radii
    r_sch1 = r21*rscale
    r_sch2 = r22*rscale


    plt.figure()
    plt.plot(r_sch1,torq_inner1*T_scale/T_01, color='#B82B34',alpha=1.,label='alpha=0.03',linewidth=3)
    plt.plot(r_sch2,torq_inner2*T_scale/T_02, color='#B8692B',alpha=1.,label='alpha = 0.01',linewidth=3)
    plt.plot(r_sch1,torq_hillcut1*T_scale/T_01, color='#B82B34',alpha=1.,label='alpha=0.03',linewidth=3,linestyle='--',dashes=(8, 8))
    plt.plot(r_sch2,torq_hillcut2*T_scale/T_02, color='#B8692B',alpha=1.,label='alpha = 0.01',linewidth=3,linestyle='--',dashes=(8, 8))
    #plt.plot(r_sch1,torq_inner1/torq_inner2[:len(r_sch1)],color='grey',alpha=1.,label='alpha=0.03',linewidth=3)

    #plt.legend()

    plt.axhline(y=0., xmin=0, xmax=r_sch1[-1], linewidth=0.5, linestyle='--',color='k')
    plt.xlim([r_sch1[-1],r_sch1[0]])
    plt.gca().invert_xaxis()
    plt.xlabel(r'$r \, \rm [r_{\rm S}]$')
    #plt.ylabel(r'$T_{\rm g} / T_{\nu}$')
    #plt.yscale('log')
    plt.ylabel(r'$T_{\rm g}/ T_0$')
    plt.tight_layout()

    return torq_inner1, torq_inner2





def plottorq_cgs(dir,q):
    rep_file = dir+'report.dat'
    
    orb, torque, torque_halfeps, torque_eps, torque_hill, r2, p2, mdot2, jdot2, fr = loadreport(rep_file)

    # scale r to schwarszchild radii
    rscale = 5.

    # CONVERT TO PHYSICAL UNITS
    G = 6.67e-8
    c = 2.99e10
    Msun = 1.98e33
    M = 1.e6*Msun
    r0 = rscale*2*G*M/c**2
    #sigma_0 = 6.2e6

    # a is cgs distance (cm)
    a = r2*r0
    # omega in 1/s
    om2 = (G*M)**0.5 * a**(-1.5)

    mach = 20.
    # read in q
    #q = 1.e-3
 
    # this changes!!    ------------------REDO FOR NEW ALPHA
    alpha = 0.03



    # CUT DATA from beginning out for smoothing
    #cut = np.where(orb>5000)[0][0]
    cut =1

    r2 = r2[cut:]
    a = a[cut:]
    om2 = om2[cut:]
    torque = torque[cut:]
    torque_hill = torque_hill[cut:]
    jdot2 = jdot2[cut:]



    smoothed = pd.rolling_mean(torque,window=800) 
    #smoothed = savitzky_golay(torque, 500, 3)
    #smoothed_hill = pd.rolling_mean(torque_hill,window=500)
    jdot_s = pd.rolling_mean(jdot2,window=800)


    # Scale torque to cgs units
    T_scale = G*M*sigma_0*r0

    # Density has radial dependence  
    sigma = sigma_0 * (r2)**(-0.5)

    # TYPE I TORQUE (SCALING)
    T_0 =  a**4 * om2**2 * sigma * q**2 * mach**2 

    # VISCOUS TORQUE
    torq_nu =  3*np.pi*a**4 * om2**2 * sigma * alpha * 1./mach**2

    # GAS TORQUE
    torq_gas = smoothed *T_scale
    torq_hillcut_gas = torque_hill *T_scale

    # ACCRETION TORQUE
    torq_acc = jdot_s*T_scale

    # GW TORQUE
    C = 64./5. * G**3/c**5 * (M+q*M)**3/(1.+1./q)/(1.+q)
    adot = C*a**(-3.)

    torq_gw = 0.5*q*M * om2 * a * adot


    # for upper X axis

    new_tick_locations = np.array([10.,9.,8.,7.,6.,5.])

    def tick_function(X):
        rX = X*2.*G*M/c**2
        V = (G*M/rX**3)**(0.5) *2. *1.e3 / np.pi  #obs f_gw at z=1 in mHz
        return [r"$%.1f$" % z for z in V]


    # For plotting versus schwarzschild radii
    r_sch = r2*rscale


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    # ******** TO DO ******
    # Check that torques are RIGHT
    index_neg = np.where(torq_gas < 0.0)[0]
    torq_gas_pos = np.array(torq_gas)
    torq_gas_pos[index_neg] = np.nan
    #and plot that with dotted line 

    #plt.plot(r_sch,abs(torq_gas)/torq_nu,color='#28B8B8',alpha=1.,label=r'$\rm gas \, +/-$',linewidth=2)
    #plt.plot(r_sch,torq_gas_pos/torq_nu,color='#B82828',alpha=0.9,linestyle='-')
    ax1.plot(r_sch,1.e4*torq_gas/torq_nu,color='#A91212',alpha=0.7,label=r'$\rm gas \, +/-$',linewidth=4)
    ax1.plot(r_sch,abs(torq_gw)/torq_nu,color='#12A912',alpha=0.7,label=r'$\rm GW$',linewidth=4)
    ax1.plot(r_sch,1.e9*abs(torq_acc)/torq_nu,color='#1212A9',alpha=0.7,label=r'$\rm accretion$',linewidth=4)
    #plt.legend()


    #plt.plot(r_sch,(torq_gas-torq_hillcut_gas)/torq_nu,color='#973009',alpha=1.,label=r'$\rm gas inside rH$',linewidth=3)
    #plt.plot(r_sch,(torq_hillcut_gas)/torq_nu,color='#28B8B8',alpha=1.,label='gas outside rH',linewidth=3)


    plt.axhline(y=0., xmin=0, xmax=r_sch[-1], linewidth=0.5, linestyle='--',color='k')
    plt.xlim([r_sch[-1],r_sch[0]])
    #ax1 = plt.gca()
    #ax1.set_xlim(ax1.get_xlim()[::-1])
    ax1.set_xlim([10.4,5.])
    ax1.set_xlabel(r'$r \, \rm [r_{\rm S}]$')
    #plt.ylabel(r'$T_{\rm g} / T_{\nu}$')
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"$f_{\rm obs} \, \rm [mHz]$")

    ax1.set_yscale('log')
    ax1.set_ylabel(r'$|T_i|/ T_{\nu}$')
    plt.tight_layout()


    #plt.figure()
    #plt.plot(adot/om2/a,torq_gas/torq_gw,color='#097097',alpha=1.,label='gas')
    #plt.plot(adot/om2/a,torq_hillcut_gas/torq_gw,color='#973009',alpha=0.8,label='gas outside rH')
    #plt.axhline(y=0., xmin=0, xmax=r_sch[-1], linewidth=0.5, linestyle='--',color='k')
    #plt.legend()
    #ax = plt.gca()
    #ax.set_yscale('symlog')
    #plt.yscale('symlog')
    #plt.tick_params(axis='x', which='minor')
    #plt.tick_params(axis='x', which='minor')
    #plt.xscale('log')
    
    #plt.xlim([3.5e-5,1.1e-4])
    #plt.ylim([-2e-6,1.5e-6])
    #plt.xticks([5e-5,1e-4])
    #plt.yticks([-1e-6,-5e-7,0.0,5e-7,1e-6])
    #plt.xlabel(r'$\dot{a}/\Omega a$')
    #plt.ylabel(r'$T_{\rm g} / T_{\rm GW}$')
    #plt.gca().invert_xaxis()
    #plt.tight_layout()

    plt.show()

    return r_sch, torq_gas,torq_hillcut_gas, torq_nu, torq_gw, T_0



    # make this a function
def f_GW_s(M,a):
    G = 6.67e-8
    f_gw = 1./np.pi * (G*M)**0.5/a**1.5 
    return f_gw

def adot_GW_cm_per_s(M,q,a):
    G = 6.67e-8
    c = 2.99e10
    adot_GW =  -64./5 * G**3/c**5 * M**3/((1+1/q)*(1+q)) / a**3
    return adot_GW


def calc_snr(dir):
    #orb, tot_torq, tot_torqs = plottorque(dir)
    orb_data, torq_data, torq_halfeps_data, torq_eps_data, torq_hill_data,a_data, phi_data, mdot_data, jdot_data = loadreport(dir+'report.dat')

    # smooth the report data
    window = 200
    torque = pd.rolling_mean(torq_data,window) 
    torque_eps = pd.rolling_mean(torq_eps_data,window)

    # read in smoothed torque - different length array
    #t,torque,torque2 = np.genfromtxt('%s/smoothed.txt'%dir,unpack=True)
    #t,r_ax,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)

    # obtain the total accumulated phase with just GWs, and with GWs+gas
    # the gas torque produces a deviation

    # This is basically counting the number of orbits. If there is a sign change
    # in the torque, we can calculate the phase shift before the change and after
    # this would make a prediction that in the beginning of the inspiral we would
    # observe -X shift, and in the later stages we would observe +Y shift in phase. 

    # We are integrating over the radius of the simulation, 
    # or the position of the secondary. 

    # Let's count the number of orbits
    # Using the amount of times the BH crosses phi=0
    phi_diff = phi_data-np.roll(phi_data,-1)
    each_orbit = np.where(phi_diff > 0)[0]
    n_orbits = len(each_orbit)

    print 'N orbits = ', n_orbits # Something is wrong with this!
    # ------


    #orb = t/2./np.pi
    orb = orb_data

    G = 6.67e-8
    c = 2.99e10

    # choose M
    Msun = 1.e33
    M = 1.e6*Msun
    #M = 1.
    q = 1.e-3
    # scale distance
    # G = c = 1
    r0 = 5*(2.*G*M/c**2)

    # disk mass 
    # specify mdot for sigma
    #sigma_0 = 1.e6
    # sigma = mdot/(3 pi nu) 
    # mdot = 0.3 * medd
    sigma_0 = 1.

    a_range = a_data * r0

    # include gas torques in the integral
    # might want to cut out initial part when disk is settling
    # after 100 orbits or so
    #   *******
    #   PUT IN PHYSICAL UNITS!
    #   *****
    #ldot_gas = torque * sigma_0

    # interpolate the smoothed torque so you can get the value at any time
    #ldot_gas_interp = interp1d(orb,ldot_gas,kind='linear')
    # We want to integrate over separation, so need a function to convert
    # from time to separation
    # Check orbits versus time units!
    #a_interp = interp1d(orb_data,a_data,kind='linear')
    # This defines my a values to integrate over, same length as ldot_gas array
    #a_range = a_interp(orb) 
    #ldot_gas_interp_a = interp1d(a_range,ldot_gas,kind='linear')

    # get Fits from smoothed torque from simulation 
    orbcut, acut, torq_out, torq_in, fit, fit2 = torqfit(dir) #in units of code T
    ldot_gas_outer = fit[0] * G  # multiply by disk mass to get torque


    da = (np.roll(a_range,-1) - a_range)
    a = a_range 
    da[-1] = da[-2]


    # integrate using composite trapezoid rule
    #phi_gw = 2*np.pi * np.sum(np.multiply(f_GW_s(M,a)/adot_GW_cm_per_s(M,q,a),da))

    phi_gw = np.trapz(2*np.pi*f_GW_s(M,a)/adot_GW_cm_per_s(M,q,a), a, da)

    # just checking the above by actually doing the integral... it checks out. 
    #phi_gw_int = -1./16 * c**5 *(G*M)**(-5./2) * (1+1./q) * (1+q) * (np.min(a)**(5./2) - np.max(a)**(5./2))


    print 'phi_gw = ', phi_gw
    print 'phi_gw / (2 pi) / 2 = N_orbits = ',phi_gw/2./np.pi/2.


    phi_tot = np.pi * np.sum(f_GW_s(M,a)/(adot_GW_cm_per_s(M,q,a)/2. + (a/G/M)**0.5*ldot_gas_outer) * da)

    print 'phi_tot = ', phi_tot
    print 'phi_tot / (2 pi) / 2 = N_orbits = ',phi_tot/2./np.pi/2.


    phi_diff = abs(phi_tot - phi_gw)
    print 'phi difference = ', phi_diff


    return a, f_GW_s(M,a), phi_gw, phi_tot




def torq_prof(file,mach,f_rH):
    t,riph,r,phi,prim,plan,diag,index,nphi = loadcheckpoint(file)
    orb = t/2./np.pi
    print orb, ' orb'

    torq = diag[:,2]
    dens = diag[:,0]


    # where are the planets?
    r_p2 = plan[1,3]
    eps = plan[1,5]
    print 'eps = ',eps

    q = 1.e-3

    # denote hill sphere
    r_hill = (1e-3/3)**(1./3) * r_p2

    print 'r hill = ', r_hill

    # define scale height to scale axes
    # depends on disk modelheight = riph/mach, DIFFERENT FOR ALPHA
    height = 1./mach

    cut_indx = np.where((riph-r_p2 < f_rH*r_hill) & (riph-r_p2 > -f_rH*r_hill))
    torq[cut_indx] = 0.

    r_ax = (riph - r_p2)/height

    plan_indx = np.where(riph-r_p2 > 0.0)[0][0]

    torq_in = torq[:plan_indx+1]
    dens_in = dens[:plan_indx+1]
    r_in = r_ax[:plan_indx+1]
    torq_out = torq[plan_indx:]
    dens_out = dens[plan_indx:]
    r_out = r_ax[plan_indx:]
    torq_out_rev = np.fliplr([torq_out])[0]

    
    # type 1 torque:
    # (dont have updated omega in repfile, but we have r_p)
    # Still in simulation units.
    om2 = 1./r_p2**(3./2)
    #T_0 =  r_p2**3 * om2**2 * sigma_0 * q**2 * mach**4
    sigma_0 = 1.
    sigma = sigma_0 * r_p2**(-0.5)
    #T_0 =  r_p2**4 * om2**2 * sigma * q**2 * mach**2 
    #print T_0


    plt.figure()
    #plt.plot(riph, torq, color='#28B8B8')
    plt.plot(riph, torq, color='#973009')
    # insdie rH #973009
    # outside rH  #28B8B8
    #plt.plot(-r_in, dens_in/T_0, color='#0B4F9C',label=r'$\rm inner \, disk$')
    #plt.plot(r_out, dens_out/T_0, color='#9C580B',linestyle='--',label=r'$\rm outer \, disk$')
    #plt.plot(r_out, (torq_in/T_0-torq_out/T_0)*10., color='darkred',linestyle='-run ',label=r'$\rm amplified difference $')
    plt.axhline(y=0.,linestyle='--',color='grey',linewidth=1.0,alpha=0.7)
    #plt.axvline(r_p2+ eps,linestyle='--',color='lightblue',linewidth=1.0,alpha=0.8)
    #plt.axvline(r_p2-eps,linestyle='--',color='lightblue',linewidth=1.0,alpha=0.8)
    plt.axvline(r_p2+r_hill,linestyle='--',color='k',linewidth=1.0,alpha=0.7)
    plt.axvline(r_p2-r_hill,linestyle='--',color='k',linewidth=1.0,alpha=0.7)
    plt.xlabel(r'$r \, [r_0]$')
    plt.ylabel(r'$\frac{dT}{dr}$')
    #plt.legend(loc=1,fontsize=18)
    #plt.xlim([-.1,1.5])
    plt.tight_layout()
    plt.show()


    return r_ax, torq#, -r_in, torq_in, r_out, -torq_out/T_0


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
    data = dens


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
    # DIFFERENT FOR ALPHA DISK IF BH IS AT DIFF R??
    height = 1./mach

    cut_indx = np.where((r_array-r_p2 < f_rH*r_hill) & (r_array-r_p2 > -f_rH*r_hill))
    data_prof[cut_indx] = 0.

    #plt.axvline(x=0., linewidth=2,color='r')
    plt.plot((r_array-r_p2)/height,phi_avg,color='black',linewidth=2)
    plt.xlabel(r'$(r-r_p)/h$')
    #plt.yscale('log')
    #plt.ylim([1e-2,1e2])
    plt.ylabel(r'$\Sigma$')
    plt.tight_layout()

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


