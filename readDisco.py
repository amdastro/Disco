import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt


'''
prim[:,0] = rho
prim[:,1] = P
prim[:,2] = u_r
prim[:,3] = u_phi
prim[:,4] = u_z
'''

def load1D(filename):

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
    R = 0.5*(riph[1:] + riph[:-1])
    for i in xrange(index.shape[0]):
        for k in xrange(index.shape[1]):
            ind0 = index[i,k]
            ind1 = ind0 + nphi[i,k]
            r[ind0:ind1] = R[i]

    return t, r, prim




def load2D(filename):

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


    r = np.zeros(piph.shape)
    R = 0.5*(riph[1:] + riph[:-1])
    for i in xrange(index.shape[0]):
        for k in xrange(index.shape[1]):
            ind0 = index[i,k]
            ind1 = ind0 + nphi[i,k]
            r[ind0:ind1] = R[i]





    return t, r, piph, prim













