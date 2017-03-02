import sys
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import readDisco as read
import triangle
import triangle.plot
import colormaps as cmaps


plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


filename='../nodrift/r64/checkpoint_0031.h5'

t,r,phi,prim = read.loadcheckpoint(filename)


x = np.multiply(r,np.cos(phi)) 
y = np.multiply(r,np.sin(phi))

triang = tri.Triangulation(x, y)

# mask off unwanted traingles
min_radius = np.min(r)
xmid = x[triang.triangles].mean(axis=1)
ymid = y[triang.triangles].mean(axis=1)
mask = np.where(xmid*xmid + ymid*ymid < min_radius*min_radius, 1, 0)
triang.set_mask(mask)

fig,ax = plt.subplots()
cax = ax.tricontourf(triang, prim[:,0],200,cmap=plt.cm.gist_heat_r)
cbar = fig.colorbar(cax)
plt.show()
