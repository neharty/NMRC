import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
from scipy.stats import lognorm, norm
import scipy.ndimage.filters as filters
from scipy.ndimage.morphology import generate_binary_structure

inFile = h5py.File(sys.argv[1], 'r')
print("Reading " + str(sys.argv[1]))
event_ids = np.array(inFile['event_ids'])
xx = np.array(inFile['xx'])
yy = np.array(inFile['yy'])
zz = np.array(inFile['zz'])
rr = (xx ** 2 + yy ** 2) ** 0.5
travel_times = np.array(inFile['station_101']['travel_times'])
receive_vectors = np.array(inFile['station_101']['receive_vectors'])
antenna_positions = inFile['station_101'].attrs['antenna_positions']
channelNum = 7
evtNum = len(event_ids)
c = 3e8
n = 1.75
step = 0.01
neighborhood_size = 5
neighborhood = generate_binary_structure(2,2)
layerR = 3000.0
#ture vertex
tt = np.arccos(zz / np.sqrt(np.square(xx) + np.square(yy) + np.square(zz)))
pp = np.arctan2(yy, xx)
pp = np.where(pp < 0, pp + 2 * np.pi, pp)
#true timing
dt = np.array([[[0.0 for x in range(evtNum)] for y in range(channelNum)] for z in range(2)])
travel_times = np.transpose(travel_times)
for i in range(channelNum):
    dt[0][i] = travel_times[0][i] - travel_times[0][-1]
    dt[1][i] = travel_times[1][i] - travel_times[1][-1]
#true receiving vectors
trueTheta = np.array([[0.0 for x in range(evtNum)] for y in range(2)])
trueTheta[0] = np.arccos(receive_vectors[:, -1, 0, 2] / np.sqrt(np.square(receive_vectors[:, -1, 0, 0]) + np.square(receive_vectors[:, -1, 0, 1]) + np.square(receive_vectors[:, -1, 0, 2])))
trueTheta[1] = np.arccos(receive_vectors[:, -1, 1, 2] / np.sqrt(np.square(receive_vectors[:, -1, 1, 0]) + np.square(receive_vectors[:, -1, 1, 1]) + np.square(receive_vectors[:, -1, 1, 2])))
truePhi = np.array([[0.0 for x in range(evtNum)] for y in range(2)])
truePhi[0] = np.arctan2(receive_vectors[:, -1, 0, 1], receive_vectors[:, -1, 0, 0])
truePhi[0] = np.where(truePhi[0] < 0, truePhi[0] + 2 * np.pi, truePhi[0])
truePhi[1] = np.arctan2(receive_vectors[:, -1, 1, 1], receive_vectors[:, -1, 1, 0])
truePhi[1] = np.where(truePhi[1] < 0, truePhi[1] + 2 * np.pi, truePhi[1])
#antenna positions
deltaPos = [[0.0, 0.0, 0.0] for i in range(channelNum)]
for i in range(channelNum):
    for j in range(3):
        deltaPos[i][j] = antenna_positions[i][j] - antenna_positions[6][j]
#target function
def targetFunc(THETA, PHI, DT):
    nVec = [np.sin(THETA) * np.cos(PHI), np.sin(THETA) * np.sin(PHI), np.cos(THETA)]
    target = 0.0
    for i in range(channelNum):
        target += np.square((nVec[0] * deltaPos[i][0] + nVec[1] * deltaPos[i][1] + nVec[2] * deltaPos[i][2]) / c * n * 1e9 - DT[i])
    target = np.sqrt(target)
    target /= (channelNum - 1.0)
    return target
#bining
phi = np.arange(0.0, 2.0 * np.pi, step)
theta = np.arange(np.pi, 0.0, -step)
Phi, Theta = np.meshgrid(phi, theta)
#for i in range(evtNum):
for i in range(10):
    #minimun finder
    zRec = targetFunc(Theta, Phi, -dt[0, :, i])
    minZRec = np.amin(zRec)
    [rowRec, colRec] = np.where(zRec == minZRec)
    if len(rowRec) != 0:
        #ray tracing from layerR to the station center
        reconTable = np.genfromtxt(sys.argv[2], skip_header = 1)
        phiRec = [item[0] for item in reconTable]
        thetaRec = [item[1] for item in reconTable]
        phiVertex = [item[2] for item in reconTable]
        thetaVertex = [item[3] for item in reconTable]
        zVertex = np.sqrt(np.square(phiRec - phi[colRec[0]]) + np.square(thetaRec - theta[rowRec[0]]))
        minZVertex = np.amin(zVertex)
        indexVertex = np.where(zVertex == minZVertex)
        #plotting receiving vector reconstruction
        im = plt.imshow(zRec, interpolation='none', extent=[0, 2 * np.pi, 0, np.pi])
        colorbar(im)
        plt.plot(phi[colRec], theta[rowRec], 'b+', label = 'Reconstructed receiving vector', linewidth = 0.5)
        plt.plot(truePhi[0][i], trueTheta[0][i], 'o', mfc = 'none', label = 'True receiving vector', linewidth = 0.5)
        plt.plot(pp[i], tt[i], 'x', label = 'True vertex', linewidth = 0.5)
        plt.plot(phiVertex[indexVertex[0][0]], thetaVertex[indexVertex[0][0]], '*', label = 'Reconstructed vertex', linewidth = 0.5)
        plt.xlabel("phi (rad)")
        plt.ylabel("theta (rad)")
        plt.title("Recon receiving = {:.3f} at ({:.3f}, {:.3f})\nTrue receiving at ({:.3f}, {:.3f})\nTrue vertex at ({:.3f}, {:.3f})\nRecon vertex found at ({:.3f}, {:.3f})\nStep: {} rad".format(minZRec, phi[colRec[0]], theta[rowRec[0]], truePhi[0][i], trueTheta[0][i], pp[i], tt[i], phiVertex[indexVertex[0][0]], thetaVertex[indexVertex[0][0]], step))
        plt.legend(fontsize = 'small', bbox_to_anchor = (1.2,1), loc = 'upper left')
        plt.savefig("./plots/dirReconRecEvt" + str(event_ids[i]) + ".pdf", bbox_inches="tight")
        plt.clf()