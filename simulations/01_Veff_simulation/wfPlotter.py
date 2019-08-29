import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader
from NuRadioReco.framework.parameters import stationParameters as stnp
from scipy.signal import hilbert
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
import h5py

logging.basicConfig(level=logging.INFO)

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
parser.add_argument('detectordescription', type=str,
                    help='path to detectordescription')
parser.add_argument('hdf5', type=str,
                    help='path to hdf5')
args = parser.parse_args()

#read in hdf5
fin = h5py.File(args.hdf5, 'r')
travel_times = np.array(fin['station_101']['travel_times'])
receive_vectors = np.array(fin['station_101']['receive_vectors'])
antenna_positions = fin['station_101'].attrs['antenna_positions']
trigger_names = fin.attrs['trigger_names']
travel_times = np.transpose(travel_times)
trigNum = len(travel_times[0][0])
channelNum = len(travel_times[0])
minTravelTimes = np.nanmin(travel_times[0], axis = 0)
for i in range(channelNum):
    travel_times[0][i] = travel_times[0][i] - minTravelTimes + 260
    travel_times[1][i] = travel_times[1][i] - minTravelTimes + 260
    antenna_positions[i] -= antenna_positions[channelNum - 1]
plane_times = [[[0 for x in range(trigNum)] for y in range(channelNum)] for z in range(2)]
c = 3.0e8 / 1.75
for i in range(channelNum - 1):
    plane_times[0][i] = - (receive_vectors[:, channelNum - 1, 0, 0] * antenna_positions[i][0] + receive_vectors[:, channelNum - 1, 0, 1] * antenna_positions[i][1] + receive_vectors[:, channelNum - 1, 0, 2] * antenna_positions[i][2]) / c * 1.0e9
    plane_times[1][i] = - (receive_vectors[:, channelNum - 1, 1, 0] * antenna_positions[i][0] + receive_vectors[:, channelNum - 1, 1, 1] * antenna_positions[i][1] + receive_vectors[:, channelNum - 1, 1, 2] * antenna_positions[i][2]) / c * 1.0e9

# read in detector positions (this is a dummy detector)
det = detector.Detector(json_filename=args.detectordescription)

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

filename = os.path.splitext(os.path.basename(args.inputfilename))[0]
#print("filename = " + filename)
dirname = os.path.dirname(args.inputfilename)
#print("dirname = " +str(dirname))
outfilename = os.path.join(dirname, 'correlation', filename)
#print("outfilename = " + str(outfilename) + ".txt")
plotDir = "/data/user/ypan/bin/simulations/width/wf/" + str(filename) + "/"
if (not os.path.exists(plotDir)):
    os.makedirs(plotDir)
outFile = open(str(outfilename) + ".txt", "w")

pulserTime = np.zeros((100000, 2))
triggerTime = np.zeros(100000)
evtNum = 0

for event in eventReader.run():
    event_id = event.get_id()
    #if event_id < 100005:
    if event_id < 1000:
        binNum = 0
        channelNum = 0
        dt = 0.0
        for station in event.get_stations():
            for channel in station.iter_channels():
                timeS = channel.get_times()
                dt = timeS[1] - timeS[0]
                binNum = len(timeS)
                channelNum = channel.get_id() + 1
                #print("binNum = {}; channelNum = {}; dt = {}; ".format(binNum, channelNum, dt))

        trace = [[0 for x in range(binNum)] for y in range(channelNum)]
        times = [[0 for x in range(binNum)] for y in range(channelNum)]
        correlationV = [[0 for x in range(channelNum)] for y in range(channelNum)]
        correlationT = [[0 for x in range(channelNum)] for y in range(channelNum)]
        for station in event.get_stations():
            station_id = station.get_id()
            for channel in station.iter_channels():
                channel_id = channel.get_id()
                # get time trace and times of bins
                trace[channel_id] = channel.get_trace()
                times[channel_id] = channel.get_times() - channel.get_trace_start_time()
                # or get the frequency spetrum instead
                #spectrum = channel.get_frequency_spectrum()
                #frequencies = channel.get_frequencies()
            #xcorrelarion
            for i in range(channelNum):
                for j in range(channelNum):
                    correlation = np.correlate(trace[i], trace[j], "full")
                    #print("len(correlation) = {}\n".format(len(correlation)))
                    correlationV[i][j] = np.amax(correlation)
                    correlationT[i][j] = (np.where(correlation == correlationV[i][j])[0][0] - len(correlation) / 2) * dt
            #writing to a file
            outFile.write("{}\t".format(event_id))
            text = " "
            for i in range(channelNum):
                text = text + "dt" + str(i) + "6 = {:.1f}".format(correlationT[i][-1]) + "ns\n"
                outFile.write("{:.3f} ".format(correlationT[i][-1]))
            outFile.write("\n")
            #plotting wfs
            for channel in station.iter_channels():
                if event_id < 1000:
                    #calculations
                    channel_id = channel.get_id()
                    avgV = np.average(trace[channel_id][0:1000])
                    varV = np.average((trace[channel_id][0:1000] - avgV) ** 2)
                    stdV = np.ones(len(times[channel_id])) * np.sqrt(varV)
                    Vrms = np.ones(len(times[channel_id])) * 9.87323e-6
                    peakV = np.amax(trace[channel_id])
                    peakT = times[channel_id][np.where(trace[channel_id] == peakV)[0][0]]
                    analytical_signal = hilbert(trace[channel_id])
                    amplitude_envelope = np.abs(analytical_signal)
                    pulserBin, pulserProp = find_peaks(amplitude_envelope, distance = 200, height = 150 * np.sqrt(varV), width = 10)
                    analytical_signalPA = hilbert(trace[6])
                    amplitude_envelopePA = np.abs(analytical_signalPA)
                    pulserBinPA, pulserPropPA = find_peaks(amplitude_envelopePA, distance = 200, height = 150 * np.sqrt(varV), width = 10)
                    #print(pulserBin)
                    #whole wfs
                    plt.subplot(channelNum, 3, 2 + channel_id * 3)
                    plt.plot(times[channel_id], amplitude_envelope, linewidth = 0.5, color = 'r')
                    plt.plot(times[channel_id], trace[channel_id], linewidth = 0.5)
                    plt.plot(times[channel_id], 4.2 * Vrms, linewidth = 0.5, color = 'y')
                    plt.plot(times[channel_id], -4.2 * Vrms, linewidth = 0.5, color = 'y')
                    #plt.plot(times[channel_id], 150.0 * stdV, linewidth = 0.5, color = 'black')
                    #plt.plot(times[channel_id], -150.0 * stdV, linewidth = 0.5, color = 'black')
                    plt.plot(times[channel_id][pulserBin], np.zeros(len(times[channel_id]))[pulserBin], "o", linewidth = 0.5, markerfacecolor='None')
                    plt.xlim(0, 2000)
                    plt.ylim(-0.0001, 0.0001)
                    plt.tick_params(labelsize = 5)
                    plt.grid(b=True, which='major', color='#666666', linestyle=':', linewidth = 0.5)
                    plt.minorticks_on()
                    plt.grid(b=True, which='minor', color='#999999', linestyle=':', alpha=0.2, linewidth = 0.5)
                    #print("channel.get_parameter('signal_time') = {}".format(channel.get_parameter('signal_time')))
                    if station.get_trigger(trigger_names[0]).has_triggered():
                        triggerTime[evtNum] = station.get_trigger(trigger_names[0]).get_trigger_time()
                    elif station.get_trigger(trigger_names[1]).has_triggered():
                        triggerTime[evtNum] = station.get_trigger(trigger_names[1]).get_trigger_time()
                    #plt.axvline(x = triggerTime[evtNum], color = 'g', linewidth = 0.5, alpha = 0.5)
                    if pulserPropPA["peak_heights"][0] < pulserPropPA["peak_heights"][-1]:
                        plt.axvline(x = travel_times[1][6][evtNum] + correlationT[channel_id][-1], color = 'g', linewidth = 0.5, alpha = 0.5)
                    else:
                        plt.axvline(x = travel_times[0][6][evtNum] + correlationT[channel_id][-1], color = 'g', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[0][channel_id][evtNum], color = 'r', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[1][channel_id][evtNum], color = 'r', linewidth = 0.5, alpha = 0.5)
                    plt.axvline(x = travel_times[0][6][evtNum] + plane_times[0][channel_id][evtNum], color = 'k', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[1][6][evtNum] + plane_times[1][channel_id][evtNum], color = 'k', linewidth = 0.5, alpha = 0.5)
                    #pulsers1
                    if len(pulserBin) > 0:
                        plt.subplot(channelNum, 3, 1 + channel_id * 3)
                        analytical_signal = hilbert(trace[channel_id])
                        amplitude_envelope = np.abs(analytical_signal)
                        plt.plot(times[channel_id][(pulserBin[0] - 300) : (pulserBin[0] + 300)], amplitude_envelope[(pulserBin[0] - 300) : (pulserBin[0] + 300)], linewidth = 0.5, color = 'r')
                        plt.plot(times[channel_id][(pulserBin[0] - 300) : (pulserBin[0] + 300)], trace[channel_id][(pulserBin[0] - 300) : (pulserBin[0] + 300)], linewidth = 0.5)
                        plt.xlim(times[channel_id][(pulserBin[0] - 300)], times[channel_id][(pulserBin[0] + 300)])
                        plt.tick_params(labelsize = 5)
                        plt.grid(b=True, which='major', color='#666666', linestyle=':', linewidth = 0.5)
                        plt.minorticks_on()
                        plt.grid(b=True, which='minor', color='#999999', linestyle=':', alpha=0.2, linewidth = 0.5)
                        #plt.axvline(x = triggerTime[evtNum], color = 'g', linewidth = 0.5, alpha = 0.5)
                    if pulserPropPA["peak_heights"][0] < pulserPropPA["peak_heights"][-1]:
                        plt.axvline(x = travel_times[1][6][evtNum] + correlationT[channel_id][-1], color = 'g', linewidth = 0.5, alpha = 0.5)
                    else:
                        plt.axvline(x = travel_times[0][6][evtNum] + correlationT[channel_id][-1], color = 'g', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[0][channel_id][evtNum], color = 'r', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[1][channel_id][evtNum], color = 'r', linewidth = 0.5, alpha = 0.5)
                    plt.axvline(x = travel_times[0][6][evtNum] + plane_times[0][channel_id][evtNum], color = 'k', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[1][6][evtNum] + plane_times[1][channel_id][evtNum], color = 'k', linewidth = 0.5, alpha = 0.5)
                    #pulsers2
                    if len(pulserBin) > 1:
                        plt.subplot(channelNum, 3, 3 + channel_id * 3)
                        analytical_signal = hilbert(trace[channel_id])
                        amplitude_envelope = np.abs(analytical_signal)
                        plt.plot(times[channel_id][(pulserBin[1] - 300) : (pulserBin[1] + 300)], amplitude_envelope[(pulserBin[1] - 300) : (pulserBin[1] + 300)], linewidth = 0.5, color = 'r')
                        plt.plot(times[channel_id][(pulserBin[1] - 300) : (pulserBin[1] + 300)], trace[channel_id][(pulserBin[1] - 300) : (pulserBin[1] + 300)], linewidth = 0.5)
                        plt.xlim(times[channel_id][(pulserBin[1] - 300)], times[channel_id][(pulserBin[1] + 300)])
                        plt.tick_params(labelsize = 5)
                        plt.grid(b=True, which='major', color='#666666', linestyle=':', linewidth = 0.5)
                        plt.minorticks_on()
                        plt.grid(b=True, which='minor', color='#999999', linestyle=':', alpha=0.2, linewidth = 0.5)
                        #plt.axvline(x = triggerTime[evtNum], color = 'g', linewidth = 0.5, alpha = 0.5)
                    if pulserPropPA["peak_heights"][0] < pulserPropPA["peak_heights"][-1]:
                        plt.axvline(x = travel_times[1][6][evtNum] + correlationT[channel_id][-1], color = 'g', linewidth = 0.5, alpha = 0.5)
                    else:
                        plt.axvline(x = travel_times[0][6][evtNum] + correlationT[channel_id][-1], color = 'g', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[0][channel_id][evtNum], color = 'r', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[1][channel_id][evtNum], color = 'r', linewidth = 0.5, alpha = 0.5)
                    plt.axvline(x = travel_times[0][6][evtNum] + plane_times[0][channel_id][evtNum], color = 'k', linewidth = 0.5, alpha = 0.5)
                    #plt.axvline(x = travel_times[1][6][evtNum] + plane_times[1][channel_id][evtNum], color = 'k', linewidth = 0.5, alpha = 0.5)
        #saving wfs
        if event_id < 1000:
            plt.suptitle("ev" + str(event_id) + " " + str(filename))
            plt.figtext(1.0, 0.5, text)
            plt.tight_layout()
            plt.savefig(str(plotDir) + "/ev" + str(event_id) + ".pdf", bbox_inches="tight")
            plt.clf()

        evtNum += 1


outFile.close()


