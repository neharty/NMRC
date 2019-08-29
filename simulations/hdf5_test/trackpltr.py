import numpy as np
from matplotlib import pyplot as plt
import h5py
import argparse
import os

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list input')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation input')
args = parser.parse_args()

fin = h5py.File(args.inputfilename, 'r')
# plot vertex distribution

xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])

#fig1 = plt.subplot(2,1,1)
#fig2 = plt.subplot(2,1,2)
#cb = plt.subplot(2,2,2)

#fig1.hist2d(xx, yy,bins=[np.arange(-4000, 4000, 50), np.arange(-4000, 4000, 50)])
#plt.colorbar(fig1)
plt.hist2d(xx, zz, bins=[np.arange(-4000, 4000, 20), np.arange(-3000, 0, 20)])
plt.colorbar()
        
#fig1.set_title("")
"""
fig1.set_xlabel("x[m]")
fig1.set_ylabel("y[m]")
"""
plt.xlabel("x[m]")
plt.ylabel("z[m]")

#plt.savefig("1and2comp.pdf")

"""
fig, ax = plt.subplots(2,1)

im1 = ax[1,1]
"""
plt.tight_layout()
plt.show()

plt.clf()

