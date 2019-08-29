from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.utilities import units
from NuRadioMC.utilities import medium
import h5py
import argparse
import json
import time
import os
import math

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str, nargs = '+', help='path to NuRadioMC hdf5 simulation output')
# parser.add_argument('outputfilename', type=str,
#                     help='name of output file storing the electric field traces at detector positions')
args = parser.parse_args()
hdf5filename = os.path.splitext(os.path.basename(args.inputfilename[0]))[0]
txtfilename = os.path.splitext(os.path.basename(args.inputfilename[1]))[0]
dirname = os.path.dirname(args.inputfilename[0])
plot_folder = os.path.join(dirname, 'plots', hdf5filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

fin = h5py.File(args.inputfilename[0], 'r')
print("Reading " + str(args.inputfilename[0]))
weights = np.array(fin['weights'])
triggered = np.array(fin['triggered'])
xx = np.array(fin['xx'])
yy = np.array(fin['yy'])
zz = np.array(fin['zz'])
maxAmp = np.array(fin['station_101']['maximum_amplitudes'])
receive_vectors = np.array(fin['station_101']['receive_vectors'])
theta = np.array(fin['zeniths'])
phi = np.array(fin['azimuths'])
polarization = np.array(fin['station_101']['polarization']).flatten()
launch_vectors = np.array(fin['station_101']['launch_vectors'])
multiple_triggers = np.array(fin['station_101']['multiple_triggers'])
travel_times = np.array(fin['station_101']['travel_times'])
n_events = fin.attrs['n_events']
vrms = fin.attrs['Vrms']
antenna_positions = fin['station_101'].attrs['antenna_positions']
number_of_triggers = len(multiple_triggers[0])
'''
for i in range(len(args.inputfilename) - 1):
    fin = h5py.File(args.inputfilename[i + 1], 'r')
    weights = np.append(weights, np.array(fin['weights']))
    triggered = np.append(triggered, np.array(fin['triggered']))
    xx = np.append(xx, np.array(fin['xx']))
    yy = np.append(yy, np.array(fin['yy']))
    zz = np.append(zz, np.array(fin['zz']))
    maxAmp = np.append(maxAmp, np.array(fin['maximum_amplitudes']), axis = 0)
    receive_vectors = np.append(receive_vectors, np.array(fin['receive_vectors']), axis = 0)
    theta = np.append(theta, np.array(fin['zeniths']))
    phi = np.append(phi, np.array(fin['azimuths']))
    polarization = np.append(polarization, np.array(fin['polarization']).flatten())
    launch_vectors = np.append(launch_vectors, np.array(fin['launch_vectors']), axis = 0)
    multiple_triggers = np.append(multiple_triggers, fin['multiple_triggers'], axis = 0)
    travel_times = np.append(travel_times, fin['travel_times'], axis = 0)
    n_events += fin.attrs['n_events']
'''
travel_times = np.transpose(travel_times)
trigNum = len(travel_times[0][0])
channelNum = len(travel_times[0])
for i in range(channelNum):
    travel_times[0][i] = travel_times[0][i] - travel_times[0][channelNum - 1]
    travel_times[1][i] = travel_times[1][i] - travel_times[1][channelNum - 1]
    antenna_positions[i] -= antenna_positions[channelNum - 1]

#receive_vectors = np.transpose(receive_vectors)
#print("len(receive_vectors) = {}; len(receive_vectors[0]) = {}; len(receive_vectors[0][0]) = {}; len(receive_vectors[0][0][0]) = {}; \n". format(len(receive_vectors), len(receive_vectors[0]), len(receive_vectors[0][0]), len(receive_vectors[0][0][0])))

txtFile = np.genfromtxt(args.inputfilename[1], skip_header = 0)
dt = [[0 for x in range(trigNum)] for y in range(channelNum + 1)]
err = [[0 for x in range(trigNum)] for y in range(channelNum - 1)]
mask = [[0 for x in range(trigNum)] for y in range(channelNum - 1)]
sumErr = np.zeros(trigNum)
sumMask = np.zeros(trigNum)
#for i in range(channelNum + 1):
    #dt[i] = [item[i] for item in txtFile]
for i in range(channelNum - 1):
    #err[i] = (dt[i + 1] - travel_times[0][i]) / travel_times[0][i]
    mask[i] = np.logical_not(np.isnan(travel_times[0][i]))
    sumErr += np.square(np.nan_to_num(err[i]))
    sumMask += mask[i].astype(float)
threshold = np.sqrt(sumErr / sumMask)
#print(threshold[np.where((threshold <= 0.1) & (threshold >= 0.0))].astype(bool))

density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 1000 * units.kg / units.m ** 3
#print("len(weights) = {}; len(threshold) = {}; ".format(len(weights), len(threshold)))
V = None
if('xmax' in fin.attrs):
    dX = fin.attrs['xmax'] - fin.attrs['xmin']
    dY = fin.attrs['ymax'] - fin.attrs['ymin']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = dX * dY * dZ
elif('rmin' in fin.attrs):
    rmin = fin.attrs['rmin']
    rmax = fin.attrs['rmax']
    dZ = fin.attrs['zmax'] - fin.attrs['zmin']
    V = np.pi * (rmax**2 - rmin**2) * dZ
multi_trigger = np.zeros(len(weights))
multiple_triggers = np.transpose(multiple_triggers)
for i in range(len(multiple_triggers)):
    Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[multiple_triggers[i].astype(bool)]) / n_events
    VeffReco = V * density_ice / density_water * 4 * np.pi * np.sum(weights[np.logical_and(np.logical_and(threshold <= 0.5, threshold >= 0.0), multiple_triggers[i].astype(bool))]) / n_events
    print("Veff{} = {:.3f} km^3 sr".format(i, Veff / units.km ** 3))
    print("VeffReco{} = {:.3f} km^3 sr".format(i, VeffReco / units.km ** 3))
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights[np.logical_or(multiple_triggers[0].astype(bool), multiple_triggers[1].astype(bool))]) / n_events
VeffReco = V * density_ice / density_water * 4 * np.pi * np.sum(weights[np.logical_and(np.logical_or(multiple_triggers[0].astype(bool), multiple_triggers[1].astype(bool)), np.logical_and(threshold <= 0.5, threshold >= 0.0))]) / n_events
print("VeffTot = {:.3f} km^3 sr".format(Veff / units.km ** 3))
print("VeffTotReco = {:.3f} km^3 sr".format(VeffReco / units.km ** 3))
print("\n")

'''
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - travel_times[0][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-300, 300, 10))
    plt.title("dt_corr" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrTrueDirect')
plt.savefig(os.path.join(plot_folder, 'dtCorrTrueDirect.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - travel_times[0][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-30, 30, 2))
    plt.title("dt_corr" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrTrueDirectZoom')
plt.savefig(os.path.join(plot_folder, 'dtCorrTrueDirectZoom.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - travel_times[1][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-300, 300, 10))
    plt.title("dt_corr" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrTrueRefract')
plt.savefig(os.path.join(plot_folder, 'dtCorrTrueRefract.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - travel_times[1][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-30, 30, 2))
    plt.title("dt_corr" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrTrueRefractZoom')
plt.savefig(os.path.join(plot_folder, 'dtCorrTrueRefractZoom.pdf'), bbox_inches="tight")
plt.clf()

plane_times = [[[0 for x in range(trigNum)] for y in range(channelNum - 1)] for z in range(2)]
#print("len(plane_times) = {}; len(plane_times[0]) = {}; len(plane_times[0][0]) = {}; \n".format(len(plane_times), len(plane_times[0]), len(plane_times[0][0])))
c = 3.0e8 / 1.75
for i in range(channelNum - 1):
    plane_times[0][i] = - (receive_vectors[:, channelNum - 1, 0, 0] * antenna_positions[i][0] + receive_vectors[:, channelNum - 1, 0, 1] * antenna_positions[i][1] + receive_vectors[:, channelNum - 1, 0, 2] * antenna_positions[i][2]) / c * 1.0e9
    plane_times[1][i] = - (receive_vectors[:, channelNum - 1, 1, 0] * antenna_positions[i][0] + receive_vectors[:, channelNum - 1, 1, 1] * antenna_positions[i][1] + receive_vectors[:, channelNum - 1, 1, 2] * antenna_positions[i][2]) / c * 1.0e9
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - plane_times[0][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-300, 300, 10))
    plt.title("dt_corr" + str(i) + "6 - dt_plan" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrPlanDirect')
plt.savefig(os.path.join(plot_folder, 'dtCorrPlanDirect.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - plane_times[0][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-30, 30, 2))
    plt.title("dt_corr" + str(i) + "6 - dt_plan" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrPlanDirectZoom')
plt.savefig(os.path.join(plot_folder, 'dtCorrPlanDirectZoom.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - plane_times[1][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-300, 300, 10))
    plt.title("dt_corr" + str(i) + "6 - dt_plan" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrPlanRefract')
plt.savefig(os.path.join(plot_folder, 'dtCorrPlanRefract.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = dt[i + 1] - plane_times[1][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-30, 30, 2))
    plt.title("dt_corr" + str(i) + "6 - dt_plan" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'CorrPlanRefractZoom')
plt.savefig(os.path.join(plot_folder, 'dtCorrPlanRefractZoom.pdf'), bbox_inches="tight")
plt.clf()

for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = plane_times[0][i] - travel_times[0][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-300, 300, 10))
    plt.title("dt_plan" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'PlanTrueDirect')
plt.savefig(os.path.join(plot_folder, 'dtPlanTrueDirect.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = plane_times[0][i] - travel_times[0][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-30, 30, 2))
    plt.title("dt_plan" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'PlanTrueDirectZoom')
plt.savefig(os.path.join(plot_folder, 'dtPlanTrueDirectZoom.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = plane_times[1][i] - travel_times[1][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-300, 300, 10))
    plt.title("dt_plan" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'PlanTrueRefract')
plt.savefig(os.path.join(plot_folder, 'dtPlanTrueRefract.pdf'), bbox_inches="tight")
plt.clf()
for i in range(channelNum - 1):
    plt.subplot(3, 2, i + 1)
    err[i] = plane_times[1][i] - travel_times[1][i]
    err[i] = np.ma.array(err[i], mask=np.isnan(err[i]))
    plt.hist(np.nan_to_num(err[i]), bins = np.arange(-30, 30, 2))
    plt.title("dt_plan" + str(i) + "6 - dt_true" + str(i) + "6 [ns]")
    plt.suptitle(hdf5filename + 'PlanTrueRefractZoom')
plt.savefig(os.path.join(plot_folder, 'dtPlanTrueRefractZoom.pdf'), bbox_inches="tight")
plt.clf()
'''

