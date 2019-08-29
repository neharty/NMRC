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
n = 1.7

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
radialdistdirect = np.array([])
radialdistreflect = np.array([])
radialdistrefract = np.array([])

directdot = np.array([])
reflectdot = np.array([])
refractdot = np.array([])

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
"""
    for station in event.get_stations():
        for channel in station.iter_channels():
                timeS = channel.get_times()
                dt = timeS[1] - timeS[0]
                binNum = len(timeS)
                channelNum = channel.get_id() + 1
                channelCtr += 1

    trace = [[0 for x in range(binNum)] for y in range(channelNum)]
    
    for station in event.get_stations():
        station_id = station.get_id()
        for channel, num in zip(station.iter_channels(), range(channelCtr)):
            channel_id = channel.get_id()
            # get time trace and times of bins
            trace[channel_id] = channel.get_trace()
            maxes = np.append(maxes, sp.find_peaks(trace[channel_id], distance=20))

np.delete(maxes, 0, axis=0)
print(maxes)
"""
          
dots1 = np.array([0.0 for x in range(eventCtr)])
dots2 = np.array([0.0 for x in range(eventCtr)])
moredots = np.array([0.0 for x in range(eventCtr)])

rv1 = np.array([None for x in range(eventCtr)])
rv2 = np.array([None for x in range(eventCtr)])
rv = np.array([None for x in range(eventCtr)])

rvdirect = np.array([])
rvreflect = np.array([])
rvrefract = np.array([])

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

# calculate correct chereknov angle for ice density at vertex position
ice = medium.southpole_simple()
rho = np.array([0.0 for x in range(len(zz))])
for x in range(len(zz)):
    rho[x] = np.arccos(1. / ice.get_index_of_refraction([xx[x], yy[x], zz[x]]))

for key, st in iteritems(fin):
    if isinstance(st, h5py._hl.group.Group):
        for i in range(eventCtr):

            triggered = np.array(st['triggered'])
            trigger_name = 'all'
            events = triggered.size
            #skip all events outside radius
            #if(np.sqrt(fin['xx'][triggered][i]**2 + fin['yy'][triggered][i]**2) < 1000):
                #continue

            rvtemp = np.array(st['receive_vectors'])[triggered]
            solntypes = np.array(st['ray_tracing_solution_type'])[triggered]
            travtimes = np.array(st['travel_times'])[triggered]
            lvtemp = np.array(st['launch_vectors'])[triggered]
            
            
            
            rv1[i] = rvtemp[i][6][0] / np.linalg.norm(rvtemp[i][6][0])
            rv2[i] = rvtemp[i][6][1] / np.linalg.norm(rvtemp[i][6][1])
            
            lv1[i] = lvtemp[i][6][0] / np.linalg.norm(lvtemp[i][6][0])
            lv2[i] = lvtemp[i][6][1] / np.linalg.norm(lvtemp[i][6][1])
            
            """
            print(solntypes[i][6][0])
            print(solntypes[i][6][1])
            print("=================")
            
            if(solntypes[i][6][0] == 1.0):
                rv[i] = rvtemp[i][6][0]
            elif(solntypes[i][6][1] == 1.0):
                rv[i] = rvtemp[i][6][1]
            elif(solntypes[i][6][0] == 3.0):
                rv[i] = rvtemp[i][6][0]
            elif(solntypes[i][6][1] == 3.0):
                rv[i] = rvtemp[i][6][1]
            else:
                rv[i] = rvtemp[i][6][0]
            """
            cond = np.abs(np.arccos(np.dot(lv1[i], -1 * showerdir[i])) - rho[i]) < np.abs(np.arccos(np.dot(lv2[i], - 1 * showerdir[i])) - rho[i])
            #radialdist = np.append(radialdist, np.sqrt(xx[i]**2 + yy[i]**2))
            
            if(cond):
                rv[i] = rvtemp[i][6][0]
            else:
                rv[i] = rvtemp[i][6][1]
                
            sphererv1[i] = cart2sphere(rv1[i])
            sphererv2[i] = cart2sphere(rv2[i])

            event = eventarr[i]
            stations = np.array([x for x in event.get_stations()])
            
            """
            print(stations[0].has_parameter(zenith))
            zeniths[i] = stations[0].get_channel(6).has_parameter(zenith)
            azimuths[i] = stations[0].get_channel(6).get_parameter(azimuth)
            """
            
            corrTime = [0.0 for x in range(channelNum)]

            for j in range(channelNum):
                corrTime[j] = stations[0].get_channel(j).get_parameter(sigtime) - stations[0].get_channel(6).get_parameter(sigtime)
                
                
            testcoords = np.array([(coords[0] - coords[6]), coords[1] - coords[6], (coords[2] - coords[6])])
            dists = c/n * 1e-9 * np.array([corrTime[0], corrTime[1], corrTime[2]])
            soln = np.linalg.solve(testcoords, dists) / np.linalg.norm(np.linalg.solve(testcoords, dists))
            
            spheremeasured[i] = cart2sphere(-1 * soln)
            
            
            dots1[i] = np.dot(soln, rv1[i])
            dots2[i] = np.dot(soln, rv2[i])
            
            """
            if(np.arccos(-1 * dots1[i]) < 0.03 or np.arccos(-1 * dots2[i] < 0.03)):
                goodpts += 1
            
            print(dots1[i])
            print(dots2[i])
            print(goodpts)
            print("-----------")
            """
            
            recon = np.dot(soln, rv[i])
            
            if(cond and solntypes[i][6][0] == 1.0):
                rvdirect = np.append(rvdirect, recon)
                radialdistdirect = np.append(radialdistdirect, np.sqrt(xx[i]**2 + yy[i]**2))
            elif((~cond) and solntypes[i][6][1] == 1.0):
                rvdirect = np.append(rvdirect, recon)
                radialdistdirect = np.append(radialdistdirect, np.sqrt(xx[i]**2 + yy[i]**2))
            elif(cond and solntypes[i][6][0] == 2.0):
                rvreflect = np.append(rvreflect, recon)
                radialdistreflect = np.append(radialdistreflect, np.sqrt(xx[i]**2 + yy[i]**2))
            elif((~cond) and solntypes[i][6][1] == 2.0):
                rvreflect = np.append(rvreflect, recon)
                radialdistreflect = np.append(radialdistreflect, np.sqrt(xx[i]**2 + yy[i]**2))
            if(cond and solntypes[i][6][0] == 3.0):
                rvrefract = np.append(rvrefract, recon)
                radialdistrefract = np.append(radialdistrefract, np.sqrt(xx[i]**2 + yy[i]**2))
            elif((~cond) and solntypes[i][6][1] == 3.0):
                rvrefract = np.append(rvrefract, recon)
                radialdistrefract = np.append(radialdistrefract, np.sqrt(xx[i]**2 + yy[i]**2))
            
                  
"""
fig1 = plt.subplot(1,2,1)
fig2 = plt.subplot(1,2,2)

fig1.plot(radialdist, np.arccos(-1*dots1), 'ro', ms = 2)
fig2.plot(radialdist, np.arccos(-1*dots2), 'bo', ms = 2)

fig1.set_title("1st soln comparison")
fig2.set_title("2nd soln comparison")

fig1.set_xlabel("radial distance [m]")
fig1.set_ylabel("angle between calculated and actual vec [rad]")

fig2.set_xlabel("radial distance [m]")
fig2.set_ylabel("angle between calculated and actual vec [rad]")

plt.savefig("1and2comp.pdf")

plt.show()

plt.clf()
"""

plt.plot(radialdistdirect, np.arccos(-1*rvdirect), 'ro', ms = 2, label='direct')
plt.plot(radialdistreflect, np.arccos(-1*rvreflect), 'bo', ms = 2, label='reflect')
plt.plot(radialdistrefract, np.arccos(-1*rvrefract), 'go', ms = 2, label='refract')
plt.legend()
plt.title("first arrival comparison")
plt.xlabel("radial distance [m]")
plt.ylabel("angle between calculated and actual vec [rad]")
plt.savefig("closer2cherenkov.pdf")
plt.show()
plt.clf()
"""
for i in range(5):
    x = np.random.randint(0, eventCtr)
    plt.plot(spheremeasured[x][0], spheremeasured[x][1], 'b*', label="recon direction")
    plt.plot(sphererv1[x][0], sphererv1[x][1], 'gx', label="1st true direction")
    plt.plot(sphererv2[x][0], sphererv2[x][1], 'r+', label ="2nd true direction")
    plt.xlabel("angle from x axis (rad)")
    plt.ylabel("angle from z axis (rad)")
    plt.ylim(0, np.pi)
    plt.xlim(0, 2*np.pi)
    plt.legend()
    plt.show()
    plt.clf()


plt.clf()
"""
