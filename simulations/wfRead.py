import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from NuRadioReco.utilities import units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.eventReader

from NuRadioReco.framework.parameters import stationParameters as stnp

logging.basicConfig(level=logging.INFO)

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='NuRadioSim file')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC simulation result')
parser.add_argument('detectordescription', type=str,
                    help='path to detectordescription')
args = parser.parse_args()

# read in detector positions (this is a dummy detector)
det = detector.Detector(json_filename=args.detectordescription)

# initialize modules
eventReader = NuRadioReco.modules.io.eventReader.eventReader()
eventReader.begin(args.inputfilename)

plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

#for plotting
fig = plt.figure()

channelCtr = 0
eventNum = 0
stationNum = 0

for event in eventReader.run():
    binNum = 0
    channelNum = 0
    dt = 0.0

    for station in event.get_stations():
        for channel in station.iter_channels():
                timeS = channel.get_times()
                dt = timeS[1] - timeS[0]
                binNum = len(timeS)
                channelNum = channel.get_id() + 1
                channelCtr += 1

    trace = [[0 for x in range(binNum)] for y in range(channelNum)]
    times = [[0 for x in range(binNum)] for y in range(channelNum)]
    correlationV = [[0 for x in range(channelNum)] for y in range(channelNum)]
    correlationT = [[0 for x in range(channelNum)] for y in range(channelNum)]
    for station in event.get_stations():
        station_id = station.get_id()
        for channel, num in zip(station.iter_channels(), range(channelCtr)):
            channel_id = channel.get_id()
            # get time trace and times of bins
            trace[channel_id] = channel.get_trace()
            times[channel_id] = channel.get_times() - channel.get_trace_start_time()

            # or get the frequency spetrum instead
            spectrum = channel.get_frequency_spectrum()
            frequencies = channel.get_frequencies()
            
            plt.title("waveform, channel: " + str(channel_id) + ", event no.: " + str(eventNum) + ", station: " + str(stationNum))
            plt.xlabel("time (ns)")
            plt.ylabel("trace")
            plt.plot(times[channel_id], trace[channel_id])
            plt.show()

        for i in range(channelNum):
                for j in range(channelNum):
                    correlation = np.correlate(trace[i], trace[j], "full")
                    #print("len(correlation) = {}\n".format(len(correlation)))
                    correlationV[i][j] = np.amax(correlation)
                    correlationT[i][j] = (np.where(correlation == correlationV[i][j])[0][0] - len(correlation) / 2) * dt
        stationNum += 1
    eventNum += 1
            #fig.savefig(os.path.join(plot_folder, "waveforms_test.pdf"))
