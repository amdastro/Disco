import sys
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import readDisco as read


dir_m30 = '../drift/r512/m30/'
dir_m40 = '../drift/r512/m40/'

drift_m30 = ([6e-6,1e-5,2e-5,3e-5,1e-4])
drift_m40 = ([1e-5])

'''chk_files_m30 = ([dir_m30 +'6e-6/checkpoint_0003.h5',\
	dir_m30 +'1e-5/checkpoint_0018.h5', \
	dir_m30 +'2e-5/checkpoint_0018.h5', \
	dir_m30 +'3e-5/checkpoint_0009.h5', \
	dir_m30 +'1e-4/checkpoint_0014.h5'])
rep_files_m30 = np.array([dir_m30 +'6e-6/report.dat',\
	dir_m30 +'1e-5/report.dat', \
	dir_m30 +'2e-5/report.dat', \
	dir_m30 +'3e-5/report.dat', \
	dir_m30 +'1e-4/report.dat'])
'''
# no atmosphere:
chk_files_m30 = ([dir_m30 +'6e-6/checkpoint_0011.h5',\
	dir_m30 +'1e-5/checkpoint_0015.h5', \
	dir_m30 +'2e-5/checkpoint_0024.h5', \
	dir_m30 +'3e-5/checkpoint_0019.h5', \
	dir_m30 +'1e-4/checkpoint_0014.h5'])
rep_files_m30 = ([dir_m30 +'6e-6/report.dat',\
	dir_m30 +'1e-5/report.dat', \
	dir_m30 +'2e-5/report.dat', \
	dir_m30 +'3e-5/report.dat', \
	dir_m30 +'1e-4/report.dat'])
chk_files_m40 = ([dir_m40 +'1e-5/checkpoint_0018.h5'])
rep_files_m40 = ([dir_m40 +'1e-5/report.dat'])

T_m30 = np.zeros(len(drift_m30))
T_m40 = np.zeros(len(drift_m40))
for i in range(0,len(drift_m30)):
		T_m30[i] = read.readtorque(chk_files_m30[i],rep_files_m30[i],30)
for i in range(0,len(drift_m40)):
		T_m40[i] = read.readtorque(chk_files_m40[i],rep_files_m40[i],40)


print '6e-6 avg norm torq = ', T_m30[0]
print '1e-5 avg norm torq = ', T_m30[1]
print '2e-5 avg norm torq = ', T_m30[2]
print '3e-5 avg norm torq = ', T_m30[3]

print '1e-5 avg norm torq mach 30= ', T_m40[0]


plt.plot(drift_m30,T_m30,markersize=11,marker='+',color='k',linewidth=1,linestyle='--',\
	label='Mach 30')
plt.plot(drift_m30,T_m30,markersize=11,marker='+',color='k',linestyle='none',linewidth=1)

plt.plot(drift_m40,T_m40,markersize=9,marker='x',color='darkcyan',linewidth=1,linestyle='-',\
	label='Mach 40')
plt.plot(drift_m40,T_m40,markersize=9,marker='x',color='darkcyan',linestyle='none',linewidth=1)

plt.axhline(y=0., xmin=0, xmax=1000, linewidth=1, linestyle='-',color='k',alpha=0.6)

plt.xlabel(r'$w \, [|\dot{a}|/\Omega a]$')
plt.ylabel(r'$\Gamma/\Gamma_0$')
plt.xscale('log')
plt.xlim([5e-6,5e-4])
plt.ylim([-0.4,0.4])
plt.legend(loc=2,numpoints=1)

plt.tight_layout()

plt.show()

