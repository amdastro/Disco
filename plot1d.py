import sys
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import readDisco as read
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['lines.linewidth'] = 2
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


#var = 0 # dens
#var = 2 # u_r
var = 3 # u_phi
#var = 4 # u_z

# Array that points to all checkpoint files
files = []
path = 'kepler/' # not general
for f in os.listdir(path):
    if os.path.isfile(os.path.join(path,f)) and f.startswith('checkpoint_'):
    	files = np.append(files,f)
print 'reading %i files'%len(files)

t = dict()
r = dict()
prim = dict()

for i in range(0,len(files)):
	t[i],r[i],prim[i] = read.load1D(path+files[i])
	print prim[i][:,var]
	plt.plot(r[i],prim[i][:,var])

plt.show()

