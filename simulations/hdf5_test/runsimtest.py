from __future__ import absolute_import, division, print_function
import argparse
# import detector simulation modules
#import NuRadioReco.modules.efieldToVoltageConverterPerChannel
import NuRadioReco.modules.efieldToVoltageConverter
#import NuRadioReco.modules.ARIANNA.triggerSimulator
#import NuRadioReco.modules.triggerSimulator
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelBandPassFilter
import NuRadioReco.modules.channelGenericNoiseAdder
import NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution
from NuRadioReco.utilities import units
from NuRadioMC.simulation import simulation2 as simulation
from NuRadioReco.framework.parameters import channelParameters as chp
import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("runstrawman")
# initialize detector sim modules
#efieldToVoltageConverterPerChannel = NuRadioReco.modules.efieldToVoltageConverterPerChannel.efieldToVoltageConverterPerChannel()
efieldToVoltageConverter = NuRadioReco.modules.efieldToVoltageConverter.efieldToVoltageConverter()
#efieldToVoltageConverterPerChannel.begin(debug=False, time_resolution=1*units.ns)
efieldToVoltageConverter.begin(debug=False, time_resolution=1*units.ns)
triggerSimulator = NuRadioReco.modules.triggerSimulator.triggerSimulator()
triggerSimulatorARIANNA = NuRadioReco.modules.ARIANNA.triggerSimulator.triggerSimulator()
channelResampler = NuRadioReco.modules.channelResampler.channelResampler()
channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
channelGenericNoiseAdder = NuRadioReco.modules.channelGenericNoiseAdder.channelGenericNoiseAdder()
calculateAmplitudePerRaySolution = NuRadioReco.modules.custom.deltaT.calculateAmplitudePerRaySolution.calculateAmplitudePerRaySolution()

class mySimulation(simulation.simulation):

    def _detector_simulation(self):
        calculateAmplitudePerRaySolution.run(self._evt, self._station, self._det)
        # save the amplitudes to output hdf5 file
        # save amplitudes per ray tracing solution to hdf5 data output
        if('max_amp_ray_solution' not in self._mout):
            self._mout['max_amp_ray_solution'] = np.zeros((self._n_events, self._n_antennas, 2)) * np.nan
#        for sim_channel in self._station.get_sim_station().iter_channels():
        for sim_channel in self._station.iter_channels():
            for iCh2, sim_channel2 in enumerate(sim_channel):
                channel_id = sim_channel2.get_id()
                self._mout['max_amp_ray_solution'][self._iE, channel_id, iCh2] = sim_channel2.get_parameter(
                    chp.maximum_amplitude_envelope)
        self._increase_signal(0, 8**0.5)
        self._increase_signal(1, 8**0.5)
        self._increase_signal(2, 8**0.5)

        # start detector simulation
#        efieldToVoltageConverterPerChannel.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern
        efieldToVoltageConverter.run(self._evt, self._station, self._det)  # convolve efield with antenna pattern

        if(bool(self._cfg['signal']['zerosignal'])):
            self._increase_signal(None, 0)

        if bool(self._cfg['noise']):
            Vrms = self._Vrms / self._bandwidth * 2 * units.GHz  # normalize noise level to the bandwidth its generated for
            channelGenericNoiseAdder.run(self._evt, self._station, self._det, amplitude=Vrms, min_freq=0 * units.MHz,
                            max_freq=2 * units.GHz, type='rayleigh')

        channelBandPassFilter.run(self._evt, self._station, self._det, filter_type='NTU+cheb')

        triggerSimulator.run(self._evt, self._station, self._det,
                             threshold=4.24 * self._Vrms,
                             triggered_channels=[0, 1, 2, 3, 4, 5],  # run trigger on all channels
                             number_concidences=3,
                             trigger_name='simple_threshold')  # the name of the trigger

parser = argparse.ArgumentParser(description='Run NuRadioMC simulation')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC input event list')
parser.add_argument('detectordescription', type=str,
                    help='path to file containing the detector description')
parser.add_argument('config', type=str,
                    help='NuRadioMC yaml config file')
parser.add_argument('outputfilename', type=str,
                    help='hdf5 output filename')
parser.add_argument('outputfilenameNuRadioReco', type=str, nargs='?', default=None,
                    help='outputfilename of NuRadioReco detector sim file')
args = parser.parse_args()

sim = mySimulation(eventlist=args.inputfilename,
                            outputfilename=args.outputfilename,
                            detectorfile=args.detectordescription,
                            station_id=101,
                            outputfilenameNuRadioReco=args.outputfilenameNuRadioReco,
                            config_file=args.config)
sim.run()
