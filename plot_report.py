import numpy as np
import matplotlib.pyplot as plt
import readDisco as read
import colormaps as cmaps


plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['lines.linewidth'] = 2
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


file='../drift/report.dat'

t, torque, power, f_r = read.loadreport(file)

orb = t/2/np.pi

plt.plot(orb,torque)
plt.xlabel(r'$t$')
plt.ylabel(r'$\rm Torque$')
plt.show()