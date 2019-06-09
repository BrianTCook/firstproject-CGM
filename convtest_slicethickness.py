#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 20:18:03 2019

@author: BrianTCook
"""

from __future__ import division
import numpy as np
import glob
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter

npix_x = int(2**13)

'''
all of the possible contingencies for radial profiles
https://matplotlib.org/users/dflt_style_changes.html for color scheme
'''

ions = [['hneutralssh', 'H I'], ['si4', 'Si IV'], ['c4', 'C IV']]
masses = [ ['1.00e+09','$9< \log(M/M_{\odot}) < 9.5$', 'C1'], ['3.16e+09', '$9.5< \log(M/M_{\odot}) < 10$', 'C8'], ['1.00e+10', '$10 < \log(M/M_{\odot}) < 10.5$', 'C2'], ['3.16e+10', '$10.5 < \log(M/M_{\odot}) < 11$', 'C4'], ['1.00e+11', '$11 < \log(M/M_{\odot}) < 11.5$', 'C5'] ]
options = [['WholeBox', r'$\Delta Z$ = 25 cMpc'], ['2Slice', r'$\Delta Z$ = 12.5 cMpc'], ['4Slice', r'$\Delta Z$ = 6.25 cMpc'], ['8Slice', r'$\Delta Z$ = 3.125 cMpc']]
sfr_answers = ['yes']
#cent_or_sat_options = ['central', 'satellite']
profile_types = ['coldens', 'covfrac']
binning_methods = ['virial']

box_size, sim = 25., 'L0025N0752'

#for L0025N0752
snaps = [[11, 3.53]]
sim_str = 'L0025N0752RECALIBRATED'
recal_ref = 'RECALIBRATED'

def figure_maker(ions, masses):
	
    '''
	initializes figure
	'''

    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')

    fig = plt.figure()
    axes = []

    for mlow, mass_label, mass_color in masses:

        for ion, ion_label in ions:

            eps, divider_x, divider_y = 0.09, 3.3, 6.25
            eps_legend = 0.12

            y0 = eps_legend + eps

            if ion == 'hneutralssh':
                x0 = eps
            if ion == 'c4':
                x0 = eps + (1-eps)/divider_x
            if ion == 'si4':
                x0= eps + 2*(1-eps)/divider_x

            if mlow == '1.00e+11':
                y0 = eps + eps_legend
            if mlow == '3.16e+10':
                y0 = eps + eps_legend + (1-eps)/divider_y
            if mlow == '1.00e+10':
                y0 = eps + eps_legend + 2*(1-eps)/divider_y
            if mlow == '3.16e+09':
                y0 = eps + eps_legend + 3*(1-eps)/divider_y
            if mlow == '1.00e+09':
                y0 = eps + eps_legend + 4*(1-eps)/divider_y

            ax = fig.add_axes([x0, y0, (1-eps)/divider_x, (1-eps)/divider_y])
            ax.tick_params(left=True, bottom=True, right=False, top=False, labelsize='small')

            ticks = [0.25, 0.5, 0.75, 1.0, 1.25]
            ticks_str = [str(x) for x in ticks] 
            
            ax.set_xticks(ticks)			
            ax.set_xticklabels(ticks_str)
            ax.set_xlim(0, np.sqrt(2))

            ax.set_yscale('log')
            ax.set_ylim(1e-5, 1e1)
            ax.set_yticks([1e-4, 1e-2, 1e0])	
            
            if ion == 'si4':
                ax.annotate(mass_label, xy = (0.25, 0.7), xycoords = 'axes fraction', fontsize=6)

            if ion != 'hneutralssh':

                ax.yaxis.set_major_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
            
            if mlow != '1.00e+11':

                ax.xaxis.set_major_formatter(NullFormatter())
                ax.xaxis.set_minor_formatter(NullFormatter())	

            if mlow == '1.00e+10': 			
				
                if ion == 'hneutralssh':
						
                    ax.set_ylabel(r'$\log$(frac. error)', fontsize=12)

            if mlow == '1.00e+11':
                if ion == 'c4':
                    ax.set_xlabel('$r/R_{vir}$', fontsize=12)

            if mlow == '1.00e+09':

                ax.set_title(ion_label, fontsize=12)

            axes.append([ax, ion, mlow])

    return fig, axes

for snap, redshift in snaps:
    fig, axes = figure_maker(ions, masses)
    
    '''
    gets slices, saves them in a list
    combine i, j --> ij (8 to 4)
    combine ij, kl --> ijkl (4 to 2)
    combine ijkl, mnop --> ijklmnop (2 to 1)
    '''
    
    for mlow, mass_label, mass_color in masses:
    
        '''
        #want star forming galaxies for this convergence test
        centrals, satellites
        central_samples = random.sample()
        satellite_samples = random.sample()
        
        they should all probably come from the same 3.125 slice right? 
        Would only need one 8 slice, one 4 slice, one 2 slice, and one 1 slice
        
        '''
        
        for ion, ion_label in ions:
            
            '''
            gets similar outputs for outputsfromslices
            number of bins dependent on total mass
            |current thickness - 3.125|/|3.125|
            will only have three plots for each panel (6.25, 12.5, 25)
            '''
            
            for axis in axes:
                axis_a, ion_a, mass_a = axis
                if ion == ion_a and mlow == mass_a:
                    ax = axis_a
                       
            X = np.linspace(0, np.sqrt(2),20)
            Y1 = [1e-5 + (1e-1)*random.random()*x for x in X]
            Y2 = [1e-5 + (1e-1)*random.random()*x for x in X]
            Y3 = [1e-5 + (1e-1)*random.random()*x for x in X]
                    
            ax.semilogy(X, Y1, label=r'$\Delta Z = 6.25$ cMpc')
            ax.semilogy(X, Y2, label=r'$\Delta Z = 12.5$ cMpc')
            ax.semilogy(X, Y3, label=r'$\Delta Z = 25$ cMpc')
            
            if mlow == '1.00e+11':		

                if ion == 'hneutralssh':
                    ax.legend(bbox_to_anchor=(2.0, -1.1, 1., 0.35), framealpha=1, ncol=3, fontsize=10)	

                                     	
plt.savefig('ahh.png')
plt.close()