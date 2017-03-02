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


path = '/Users/Shark/Desktop/Disco_tests/nodrift/r64/'
savedir = '/Users/Shark/Desktop/nodrift/r64/'

# Array that points to all checkpoint files
files = []
for f in os.listdir(path):
    if os.path.isfile(os.path.join(path,f)) and f.startswith('checkpoint_'):
    	files = np.append(files,f)
print 'reading %i files'%len(files)



for i in range(0,len(files)):
	read.plot2d(path+files[i],savedir)
	print 'plotted %s'%(path+files[i])


