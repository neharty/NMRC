from __future__ import absolute_import, division, print_function
import logging
import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt
import scipy.signal as sp
from six import iteritems

from radiotools import helper as hp
from radiotools import plthelpers as php
from NuRadioMC.utilities import units
from NuRadioMC.utilities import medium
import json
import time
import os
import math


import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader

logging.basicConfig(level=logging.INFO)

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
parser.add_argument('detectordescription', type=str,
                    help='path to detectordescription')
parser.add_argument('simoutp', type=str,
                    help='path to NuRadioMC hdf5 simulation output')
args = parser.parse_args()

# read in detector positions (this is a dummy detector)
det = detector.Detector(json_filename=args.detectordescription)

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

sigtime = NuRadioReco.framework.parameters.channelParameters.signal_time
zenith = NuRadioReco.framework.parameters.stationParameters.zenith
azimuth = NuRadioReco.framework.parameters.stationParameters.azimuth


c = 3e8
n = 1.75

def cart2sphere(v):
    phi = np.arccos(v[2]/np.linalg.norm(v)) # angle from z-axis
    theta = np.arctan2(v[1], v[0]) # angle from x-axis
    return np.array([theta, phi])

def sphere2cart(zen, az):
    
    x = np.sin(zen) * np.cos(az)
    y = np.sin(zen) * np.sin(az)
    z = np.cos(zen) 
    
    return np.array([x,y,z])

fin = h5py.File(args.simoutp, 'r')
print('the following triggeres where simulated: {}'.format(fin.attrs['trigger_names']))

#for plotting
#fig = plt.figure()

eventCtr = 0
channelCtr = 0
goodpts = 0

eventarr= np.array([])

avpcterr = np.array([])
radialdist = np.array([])

coords = fin['station_101'].attrs['antenna_positions']

channelNum = 7


for event in eventReader.run():
    eventarr = np.append(eventarr, event)
    eventCtr += 1

maxes = np.array([[]])

eventCtr = 0

for event in eventReader.run():
    eventarr = np.append(eventarr, event)
    eventCtr += 1
          
pcterr = np.array([0.0 for x in range(eventCtr)])

rv1 = np.array([None for x in range(eventCtr)])
rv2 = np.array([None for x in range(eventCtr)])
rv = np.array([None for x in range(eventCtr)])

lv1 = np.array([None for x in range(eventCtr)])
lv2 = np.array([None for x in range(eventCtr)])
lv = np.array([None for x in range(eventCtr)])
launch_vectors = np.array(fin['station_101']['launch_vectors'])

sphererv1 = np.array([None for x in range(eventCtr)])
sphererv2 = np.array([None for x in range(eventCtr)])
spheremeasured = np.array([None for x in range(eventCtr)])

theta = np.array(fin['zeniths'])
phi = np.array(fin['azimuths'])

xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])

showerdir = np.array([None for x in range(eventCtr)])
for x in range(eventCtr):
    showerdir[x] = sphere2cart(theta[x], phi[x])

for key, st in iteritems(fin):
    if isinstance(st, h5py._hl.group.Group):
        for i in range(eventCtr):

            triggered = np.array(st['triggered'])
            trigger_name = 'all'
            events = triggered.size                        

            event = eventarr[i]
            stations = np.array([x for x in event.get_stations()])
            
            
            corrTime1 = [0.0 for x in range(channelNum)]
            corrTime2 = [0.0 for x in range(channelNum)]

            for j in range(channelNum):
                corrTime1[j] = stations[0].get_channel(j).get_parameter(sigtime) - stations[0].get_channel(1).get_parameter(sigtime)
                
            for j in range(channelNum):
                corrTime2[j] = stations[0].get_channel(j).get_parameter(sigtime) - stations[0].get_channel(5).get_parameter(sigtime)
                
                
            testcoords1 = np.array([(coords[0] - coords[1]), coords[2] - coords[1], (coords[6] - coords[1])])
            dists1 = c/n * 1e-9 * np.array([corrTime1[0], corrTime1[1], corrTime1[6]])
            dir1 = -1 * np.linalg.solve(testcoords1, dists1) / np.linalg.norm(np.linalg.solve(testcoords1, dists1))
            
            testcoords2 = np.array([(coords[3] - coords[5]), coords[4] - coords[5], (coords[6] - coords[5])])
            dists2 = c/n * 1e-9 * np.array([corrTime2[3], corrTime2[4], corrTime2[6]])
            dir2 = -1 * np.linalg.solve(testcoords2, dists2) / np.linalg.norm(np.linalg.solve(testcoords2, dists2))
            
            v = np.dot(dir1, dir2)
            
            coeffmatrix = [[-v, 1],
                           [1, -v]]
            
            righthandside = [np.dot(dir2, (coords[1] - coords[5])), np.dot(dir1, (coords[5] - coords[1]))]
            
            tssoln = np.linalg.solve(coeffmatrix, righthandside)
            
            vertex = ((tssoln[0] * dir1 + coords[1]) + (tssoln[1] * dir2 + coords[5]))
            
            actual = [xx[i], yy[i], zz[i]]
            
            pcterr[i] = np.abs(np.linalg.norm(vertex) - np.linalg.norm(actual)) * 100 /np.linalg.norm(actual)
            
            print(dir1, dir2)
            print(vertex, actual)
            print("======================")
            
plt.plot(np.sqrt(xx**2 + yy**2 + zz**2), pcterr, 'ro', ms = 2)
plt.title("first arrival comparison")
plt.xlabel("radial distance [m]")
plt.ylabel("angle between calculated and actual vec [rad]")
plt.show()
plt.clf()
