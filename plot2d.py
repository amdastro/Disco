import sys
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import readDisco as read
import triangle
import triangle.plot

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


filename='../drift/checkpoint_0001.h5'

t,r,phi,prim = read.load2D(filename)


x = r*np.cos(phi)
y = r*np.sin(phi)


xy = [x,y]
xyt = np.transpose(xy)
A = dict(vertices=xyt)
B = triangle.triangulate(A)
triangles = B['triangles']

plt.tricontourf(x,y,triangles,(prim[:,0]),200,cmap='afmhot')
plt.colorbar()
plt.show()

#triang = tri.Triangulation(x,y)

#xmid = x[triang.triangles].mean(axis=1)
#ymid = y[triang.triangles].mean(axis=1)
#mask = 

# aD unique arrays for x and y axes
#x_ax = np.unique(xmid)
#y_ax = np.unique(ymid)


# Need to grid the 1d prim data
#prim2D = np.zeros([len(x_ax),len(y_ax)])


#for i in range(0,len(x_ax)):
#	for j in range(0,len(y_ax)):
#		points = np.where((x == x_ax[i]) & (y == y_ax[j]))


#plt.contourf(x,y,prim[:,0],20)
#plt.show()