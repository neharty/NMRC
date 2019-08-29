from __future__ import absolute_import, division, print_function
import numpy as np
from radiotools import helper as hp
from radiotools import plthelpers as php
from matplotlib import pyplot as plt
from NuRadioMC.utilities import units
from NuRadioMC.utilities import medium
from six import iteritems
import h5py
import argparse
import json
import time
import os
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

parser = argparse.ArgumentParser(description='Plot NuRadioMC event list output.')
parser.add_argument('inputfilename', type=str,
                    help='path to NuRadioMC hdf5 simulation output')
parser.add_argument('--trigger_name', type=str, default=None, nargs='+',
                    help='the name of the trigger that should be used for the plots')
parser.add_argument('--Veff', type=str,
                    help='specify json file where effective volume is saved as a function of energy')
args = parser.parse_args()

filename = os.path.splitext(os.path.basename(args.inputfilename))[0]
dirname = os.path.dirname(args.inputfilename)
plot_folder = os.path.join(dirname, 'plots', filename)
if(not os.path.exists(plot_folder)):
    os.makedirs(plot_folder)

fin = h5py.File(args.inputfilename, 'r')
print('the following triggeres where simulated: {}'.format(fin.attrs['trigger_names']))
if(args.trigger_name is None):
    triggered = np.array(fin['triggered'])
    print("you selected any trigger")
    trigger_name = 'all'
else:
    if(len(args.trigger_name) > 1):
        print("trigger {} selected which is a combination of {}".format(args.trigger_name[0], args.trigger_name[1:]))
        trigger_name = args.trigger_name[0]
        plot_folder = os.path.join(dirname, 'plots', filename, args.trigger_name[0])
        if(not os.path.exists(plot_folder)):
            os.makedirs(plot_folder)
        triggered = np.zeros(len(fin['multiple_triggers'][:, 0]), dtype=np.bool)
        for trigger in args.trigger_name[1:]:
            iTrigger = np.squeeze(np.argwhere(fin.attrs['trigger_names'] == trigger))
            triggered = triggered | np.array(fin['multiple_triggers'][:, iTrigger], dtype=np.bool)
    else:
        trigger_name = args.trigger_name[0]
        iTrigger = np.argwhere(fin.attrs['trigger_names'] == trigger_name)
        triggered = np.array(fin['multiple_triggers'][:, iTrigger], dtype=np.bool)
        print("\tyou selected '{}'".format(trigger_name))
        plot_folder = os.path.join(dirname, 'plots', filename, trigger_name)
        if(not os.path.exists(plot_folder)):
            os.makedirs(plot_folder)

weights = np.array(fin['weights'])[triggered]
n_events = fin.attrs['n_events']

###########################
# calculate effective volume
###########################
density_ice = 0.9167 * units.g / units.cm ** 3
density_water = 997 * units.kg / units.m ** 3

n_triggered = np.sum(weights)
print('fraction of triggered events = {:.0f}/{:.0f} = {:.3f}'.format(n_triggered, n_events, n_triggered / n_events))

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
Veff = V * density_ice / density_water * 4 * np.pi * np.sum(weights) / n_events
print("Veff = {:.6g} km^3 sr".format(Veff / units.km ** 3))


###########################
# plot neutrino direction
###########################
for typ in [12, 14, 16]:
    flav = np.array(fin['flavors'][triggered] == typ)
    zentemp = np.array(fin['zeniths'])[triggered]
    
    zeniths = np.array([zentemp[0]])
    weightstemp = np.array([])

    
    for i in range(flav.size):
        if(flav[i]):
            zeniths = np.append(zeniths, [zentemp[i]], axis=0)
            weightstemp = np.append(weightstemp, weights[i])

    
    if typ == 12:
        title = 'electron'
    elif typ == 14:
        title = 'muon'
    elif typ == 16:
        title = 'tau'
        
    zeniths = np.delete(zeniths, 0, axis=0)
    
    fig, ax = php.get_histogram(zeniths / units.deg, weights=weightstemp,
                                ylabel='weighted entries', xlabel='zenith angle [deg]',
                                bins=np.arange(0, 181, 5), figsize=(6, 6))
    ax.set_xticks(np.arange(0, 181, 45))
    ax.set_title(title)
    fig.tight_layout()
    
    fig.savefig(os.path.join(plot_folder, title + '_neutrino_direction.pdf'))

###########################
# calculate sky coverage of 90% quantile
###########################
from radiotools import stats
q2 =stats.quantile_1d(np.array(fin['zeniths'])[triggered], weights, 0.95)
q1 =stats.quantile_1d(np.array(fin['zeniths'])[triggered], weights, 0.05)
from scipy import integrate
def a(theta):
    return np.sin(theta)
b = integrate.quad(a, q1, q2)
print("90% quantile sky coverage {:.2f} sr".format(b[0] * 2 * np.pi))

###########################
# plot vertex distribution
###########################

for typ in [12, 14, 16]:
    fig, ax = plt.subplots(1, 1)
    
    flav = np.array(fin['flavors'][triggered] == typ)
    #antiflav = np.array(fin['flavors'][triggered] == -1*typ)
    #flav = np.append(flav, antiflav)
    #print(flav.size)
    xtemp = np.array(fin['xx'][triggered])
    ytemp = np.array(fin['yy'][triggered])
    ztemp = np.array(fin['zz'][triggered])
    #print(xtemp.size)

    xx = np.array([])
    yy = np.array([])
    zz = np.array([])
    weightstemp = np.array([])
    
    imgarr = np.array([])

    for i in range(flav.size):
        if (flav[i]):
            xx = np.append(xx, xtemp[i])
            yy = np.append(yy, ytemp[i])
            zz = np.append(zz, ztemp[i])
            weightstemp = np.append(weightstemp, weights[i])
    
    #print(xx.size)
    rr = (xx ** 2 + yy ** 2) ** 0.5
    mask_weight = weightstemp > 1e-2
    max_r = rr[mask_weight].max()
    max_z = np.abs(zz[mask_weight]).max()
    h = ax.hist2d(rr / units.m, zz / units.m, bins=[np.linspace(0, max_r, 50), np.linspace(-max_z, 0, 50)],
              cmap=plt.get_cmap('Blues'), weights=weightstemp)
    cb = plt.colorbar(h[3], ax=ax)
    cb.set_label("weighted number of events")
    ax.set_aspect('equal')
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_xlim(0, rmax)
    ax.set_ylim(fin.attrs['zmin'], 0)
    if typ == 12:
        ax.set_title("Electron Neutrino Triggers")
        fig.savefig(os.path.join(plot_folder, 'elec_vertex_distribution.pdf'), bbox='tight')
    elif typ == 14:
        ax.set_title("Muon Neutrino Triggers")
        fig.savefig(os.path.join(plot_folder, 'muon_vertex_distribution.pdf'), bbox='tight')
    elif typ == 16:
        ax.set_title("Tau Neutrino Triggers")
        fig.savefig(os.path.join(plot_folder, 'tau_vertex_distribution.pdf'), bbox='tight')

    
    #pdf.add_page()
    #pdf.image(ax, 0, 0)

#pdf.output("triggers.pdf", "F")
    
    #fig.savefig(os.path.join(plot_folder, 'vertex_distribution.pdf'), bbox='tight')
#with matplotlib.backends.backend_pdf.PdfPages('flavor triggers.pdf') as pdf:
    # As many times as you like, create a figure fig and save it:
    #fig = plt.tight_layout()
    #pdf.savefig(fig)
    # When no figure is specified the current figure is saved
    #pdf.savefig()

###########################
# loop over all stations and produce station specific plots
###########################

for key, station in iteritems(fin):
    if isinstance(station, h5py._hl.group.Group):
        ###########################
        # recalculate triggers per station
        ###########################
        
        if(args.trigger_name is None):
            triggered = np.array(station['triggered'])
            print("you selected any trigger")
            trigger_name = 'all'
        else:
            if(len(args.trigger_name) > 1):
                print("trigger {} selected which is a combination of {}".format(args.trigger_name[0], args.trigger_name[1:]))
                trigger_name = args.trigger_name[0]
                triggered = np.zeros(len(station['multiple_triggers'][:, 0]), dtype=np.bool)
                for trigger in args.trigger_name[1:]:
                    iTrigger = np.squeeze(np.argwhere(fin.attrs['trigger_names'] == trigger))
                    triggered = triggered | np.array(station['multiple_triggers'][:, iTrigger], dtype=np.bool)
            else:
                trigger_name = args.trigger_name[0]
                iTrigger = np.argwhere(fin.attrs['trigger_names'] == trigger_name)
                triggered = np.array(station['multiple_triggers'][:, iTrigger], dtype=np.bool)
                print("\tyou selected '{}'".format(trigger_name))
            
        ###########################
        # plot incoming direction
        ###########################
        for typ in [12, 14, 16]:
            flav = np.array(fin['flavors'][triggered] == typ)
            rvtemp = np.array(station['receive_vectors'])[triggered]
            weightstemp = np.array([])
            
            #print(rvtemp.shape)
            
            receive_vectors = np.array([rvtemp[0]])
            #print(receive_vectors[0])
            #print(rvtemp[0])
            
            for i in range(flav.size):
                if (flav[i]):
                    receive_vectors = np.append(receive_vectors, [rvtemp[i]], axis=0)
                    weightstemp = np.append(weightstemp, weights[i])
#                    print("--------------------------")
#                    print(rvtemp[0])
#                    print("----------")
#                    print(receive_vectors[0])
#                    print("--------------------------")
            
            receive_vectors = np.delete(receive_vectors, 0, axis=0)
                        
            # receive_vectors = np.array(station['receive_vectors'])[triggered]
            # for all events, antennas and ray tracing solutions
            zeniths, azimuths = hp.cartesian_to_spherical(receive_vectors[:, :, :, 0].flatten(),
                                                          receive_vectors[:, :, :, 1].flatten(),
                                                          receive_vectors[:, :, :, 2].flatten())
            
            for i in range(len(azimuths)):
                azimuths[i] = hp.get_normalized_angle(azimuths[i])
                weights_matrix = np.outer(weightstemp, np.ones(np.prod(receive_vectors.shape[1:-1]))).flatten()
                mask = ~np.isnan(azimuths)  # exclude antennas with not ray tracing solution (or with just one ray tracing solution)
            
            fig, axs = php.get_histograms([zeniths[mask] / units.deg, azimuths[mask] / units.deg],
                                        bins=[np.arange(0, 181, 5), np.arange(0, 361, 45)],
                                        xlabels=['zenith [deg]', 'azimuth [deg]'],
                                        weights=weights_matrix[mask], stats=False)
       # axs[0].xaxis.set_ticks(np.arange(0, 181, 45))
            majorLocator = MultipleLocator(45)
            majorFormatter = FormatStrFormatter('%d')
            minorLocator = MultipleLocator(5)
            axs[0].xaxis.set_major_locator(majorLocator)
            axs[0].xaxis.set_major_formatter(majorFormatter)
            axs[0].xaxis.set_minor_locator(minorLocator)
            
            if typ == 12:
                fig.suptitle('incoming signal direction (electron neutrinos)')
                fig.savefig(os.path.join(plot_folder, 'elec_{}_incoming_signal.pdf'.format(key)))
            elif typ == 14:
                fig.suptitle('incoming signal direction (muon neutrinos)')
                fig.savefig(os.path.join(plot_folder, 'muon_{}_incoming_signal.pdf'.format(key)))
            elif typ == 16:
                fig.suptitle('incoming signal direction (tau neutrinos)')
                fig.savefig(os.path.join(plot_folder, 'tau_{}_incoming_signal.pdf'.format(key)))

        
        ###########################
        # plot polarization
        ###########################
            ptemp = np.array(station['polarization'])[triggered]

            p = np.array([ptemp[0]])
            
            
            for i in range(flav.size):
                if (flav[i]):
                    p = np.append(p, [ptemp[i]], axis=0)
                    
            p = np.delete(p, 0, axis = 0)
            
            p_H = (p[:,:,:,0]**2 + p[:,:,:,1]**2)**0.5

            p_V = np.abs(p[:,:,:,2])
            weights_matrix = np.outer(weightstemp, np.ones(np.prod(p_V.shape[1:]))).flatten()
            p_ratio = (p_V/p_H).flatten()
            bins = np.linspace(0, 7, 150)
            
            if typ == 12:
                title = 'electron'
            elif typ == 14:
                title = 'muon'
            elif typ == 16:
                title = 'tau'
            
            plt.locator_params(numticks=10)
            
            mask = zeniths > 90 * units.deg  # select rays coming from below
            #for all events, antennas and ray tracing solutions
            fig, ax = php.get_histogram(p_ratio,
                                        bins=bins,
                                        xlabel='vertical/horizonal polarization ratio ' + title,
                                        weights=weights_matrix, stats=False,
                                        kwargs={'facecolor':'0.7', 'alpha':1, 'edgecolor':"k", 'label': 'all'},
                                        figsize=(6, 6))
            maxy = ax.get_ylim()
            php.get_histogram(p_ratio[mask],
                              bins=bins,
                              weights=weights_matrix[mask], stats=False,
                              xlabel='vertical/horizonal polarization ratio ' + title,
                              ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k", 'label': 'direct rays'})
            ax.set_xticks(bins)
            ax.locator_params(axis = 'x', nbins = 7)
            ax.legend()
            ax.set_ylim(maxy)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_folder, title + '_{}_polarization.pdf'.format(key)))
           
            mask = zeniths > 90 * units.deg  # select rays coming from below
            fig, ax = php.get_histogram(p_ratio,
                                        bins=bins,
                                        stats=False,
                                        xlabel='vertical/horizonal polarization ratio, ' + title,
                                        kwargs={'facecolor':'0.7', 'alpha':1, 'edgecolor':"k", 'label': 'all'},
                                        figsize=(6, 6))
            maxy = ax.get_ylim()
            php.get_histogram(p_ratio[mask],
                              bins=bins,
                              xlabel='vertical/horizonal polarization ratio, unweighted, ' + title,
                              stats=False,
                              ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k", 'label': 'direct rays'})
            ax.set_xticks(bins)
            ax.locator_params(axis = 'x', nbins = 7)
            ax.legend()
            ax.set_ylim(maxy)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_folder, title + '_{}_polarization_unweighted.pdf'.format(key)))
            
            #plot vs V/H angle
            p_ratio = np.arctan2(p_V, p_H).flatten()
            bins = np.linspace(-np.pi, np.pi, 50)
            
            fig, ax = php.get_histogram(p_ratio,
                                        bins=bins,
                                        xlabel='vertical/horizonal polarization ratio angle,' + title,
                                        weights=weights_matrix, stats=False,
                                        kwargs={'facecolor':'0.7', 'alpha':1, 'edgecolor':"k", 'label': 'all'},
                                        figsize=(6, 6))
            maxy = ax.get_ylim()
            php.get_histogram(p_ratio[mask],
                              bins=bins,
                              weights=weights_matrix[mask], stats=False,
                              xlabel='vertical/horizonal polarization ratio angle,' + title,
                              ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k", 'label': 'direct rays'})
            ax.set_xticks(bins)
            ax.locator_params(axis = 'x', nbins = 7)
            ax.legend()
            ax.set_ylim(maxy)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_folder, title + '_{}_polarization_angles.pdf'.format(key)))
           
            mask = zeniths > 90 * units.deg  # select rays coming from below
            fig, ax = php.get_histogram(p_ratio,
                                        bins=bins,
                                        stats=False,
                                        xlabel='vertical/horizonal polarization ratio angle, ' + title,
                                        kwargs={'facecolor':'0.7', 'alpha':1, 'edgecolor':"k", 'label': 'all'},
                                        figsize=(6, 6))
            maxy = ax.get_ylim()
            php.get_histogram(p_ratio[mask],
                              bins=bins,
                              xlabel='vertical/horizonal polarization ratio angle, unweighted, ' + title,
                              stats=False,
                              ax=ax, kwargs={'facecolor': 'C0', 'alpha': 1, 'edgecolor': "k", 'label': 'direct rays'})
            ax.set_xticks(bins)
            ax.locator_params(axis = 'x', nbins = 7)
            ax.legend()
            ax.set_ylim(maxy)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_folder, title + '_{}_polarization_unweighte_angles.pdf'.format(key)))
        ##########################
         #plot viewing angle
        ##########################
        
            zeniths = np.array(fin['zeniths'])[triggered]
            azimuths = np.array(fin['azimuths'][triggered])
            lvtemp = np.array(station['launch_vectors'])[triggered]
            
            xtemp = np.array(fin['xx'])[triggered]
            ytemp = np.array(fin['yy'])[triggered]
            ztemp = np.array(fin['zz'])[triggered]
            
            launch_vectors = np.array([lvtemp[0]])
            zenith_inp = np.array([zeniths[0]])
            azimuth_inp = np.array([azimuths[0]])
            xx = np.array([])
            yy = np.array([])
            zz = np.array([])
            
            for i in range(flav.size):
                if (flav[i]):
                    launch_vectors = np.append(launch_vectors, [lvtemp[i]], axis=0)
                    zenith_inp = np.append(zenith_inp, [zeniths[i]], axis=0)
                    azimuth_inp = np.append(azimuth_inp, [azimuths[i]], axis=0)
                    xx = np.append(xx, xtemp[i])
                    yy = np.append(yy, ytemp[i])
                    zz = np.append(zz, ztemp[i])
                    
            launch_vectors = np.delete(launch_vectors, 0, axis=0)
            zenith_inp = np.delete(zenith_inp, 0, axis=0)
            azimuth_inp = np.delete(azimuth_inp, 0, axis=0)
            
            shower_axis = -1 * hp.spherical_to_cartesian(zenith_inp, azimuth_inp)
            viewing_angles = np.array([hp.get_angle(x, y) for x, y in zip(shower_axis, launch_vectors[:, 0, 0])])
            
            # calculate correct chereknov angle for ice density at vertex position
            ice = medium.southpole_simple()
            n_indexs = np.array([ice.get_index_of_refraction(x) for x in np.array([xx, yy, zz]).T])
            rho = np.arccos(1. / n_indexs)
            
            mask = ~np.isnan(viewing_angles)
            fig, ax = php.get_histogram((viewing_angles[mask] - rho[mask]) / units.deg, weights=weightstemp[mask],
                                        bins=np.arange(-30, 30, 1), xlabel='viewing - cherenkov angle [deg] - ' + title, figsize=(6, 6))
            fig.savefig(os.path.join(plot_folder, title + '_{}_dCherenkov.pdf'.format(key)))
        
        ###########################
        # plot flavor ratios
        ###########################
        
        flavor_labels = ['e cc', r'$\bar{e}$ cc', 'e nc', r'$\bar{e}$ nc',
                   '$\mu$ cc', r'$\bar{\mu}$ cc', '$\mu$ nc', r'$\bar{\mu}$ nc',
                   r'$\tau$ cc', r'$\bar{\tau}$ cc', r'$\tau$ nc', r'$\bar{\tau}$ nc']
        yy = np.zeros(len(flavor_labels))
        yy[0] = np.sum(weights[(fin['flavors'][triggered] == 12) & (fin['interaction_type'][triggered] == 'cc')])
        yy[1] = np.sum(weights[(fin['flavors'][triggered] == -12) & (fin['interaction_type'][triggered] == 'cc')])
        yy[2] = np.sum(weights[(fin['flavors'][triggered] == 12) & (fin['interaction_type'][triggered] == 'nc')])
        yy[3] = np.sum(weights[(fin['flavors'][triggered] == -12) & (fin['interaction_type'][triggered] == 'nc')])
        
        yy[4] = np.sum(weights[(fin['flavors'][triggered] == 14) & (fin['interaction_type'][triggered] == 'cc')])
        yy[5] = np.sum(weights[(fin['flavors'][triggered] == -14) & (fin['interaction_type'][triggered] == 'cc')])
        yy[6] = np.sum(weights[(fin['flavors'][triggered] == 14) & (fin['interaction_type'][triggered] == 'nc')])
        yy[7] = np.sum(weights[(fin['flavors'][triggered] == -14) & (fin['interaction_type'][triggered] == 'nc')])
        
        yy[8] = np.sum(weights[(fin['flavors'][triggered] == 16) & (fin['interaction_type'][triggered] == 'cc')])
        yy[9] = np.sum(weights[(fin['flavors'][triggered] == -16) & (fin['interaction_type'][triggered] == 'cc')])
        yy[10] = np.sum(weights[(fin['flavors'][triggered] == 16) & (fin['interaction_type'][triggered] == 'nc')])
        yy[11] = np.sum(weights[(fin['flavors'][triggered] == -16) & (fin['interaction_type'][triggered] == 'nc')])
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.bar(range(len(flavor_labels)), yy)
        ax.set_xticks(range(len(flavor_labels)))
        ax.set_xticklabels(flavor_labels, fontsize='large', rotation=45)
        ax.set_title("trigger: {}".format(trigger_name))
        ax.set_ylabel('weighted number of triggers', fontsize='large')
        fig.tight_layout()
        fig.savefig(os.path.join(plot_folder, '{}_flavor.pdf'.format(key)))
        plt.show()
