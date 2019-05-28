from __future__ import division
import numpy as np
import pandas as pd
import itertools
import csv
import shutil
import glob
import math
#from read_eagle_files import *
from random import randint, shuffle

def halopicker(sim, snap, mlow, mhigh, is_it_star_forming):#, central_or_satellite): #num_of_parameters, #num_of_galaxies

	column_names = ['Galaxy_ID','sub_Mass_Total','sub_Mass_Star','sub_SFR','sub_gcom_x','sub_gcom_y','sub_gcom_z','fof_critR200','sub_sgn','fof_snap']
	data = pd.read_csv('EAGLE_queries/aperturesizechosen/halos_snap=%i_apsize=30_%s.csv'%(snap,sim), header=None)
	data.columns = column_names

	print "The parameters available are sub_Mass_Total (M_{Sun}), sub_Mass_Star (M_{Sun}), sub_SFR (M_{Sun}/yr), sub_gcom_x (cMpc), sub_gcom_y (cMpc), sub_gcom_z (cMpc), fof_critR200 (pkpc)"	

	#filters out the rows of the data frame that meet the desired requirements
	data = data.loc[data['sub_Mass_Star'] > mlow]
	data = data.loc[data['sub_Mass_Star'] < mhigh]

	sfr_threshold_val = 1e-11 #solar masses per year

	if is_it_star_forming == 'yes':
		data = data.loc[data['sub_SFR'] > sfr_threshold_val]
	if is_it_star_forming == 'no':
		data = data.loc[data['sub_SFR'] < sfr_threshold_val]

	'''
	if central_or_satellite == 'central':
		data = data.loc[data['sub_sgn'] == 0]
	if central_or_satellite == 'satellite':
		data = data.loc[data['sub_sgn'] != 0]
	'''

	#saves the galaxies with the desired requirements
	galaxy_IDs = data['Galaxy_ID'].tolist()
	galaxies = data.values.tolist()

	return galaxy_IDs, galaxies

# basic box and projection region parameters (splitting into (sub)slices is done later)
simulations = ['L0025N0752']
snaps = [11, 19, 28]
#masses = [ [10**9., 10**9.5], [10**9.5, 10**10.], [10**10., 10**10.5], [10**10.5, 10**11.], [10**11., 10**11.5] ]
masses = [ [10**7., 10**7.5], [10**7.5, 10**8.], [10**8., 10**8.5], [10**8.5, 10**9.], [10**9., 10**9.5] ]
sfr_answers = ['yes', 'no']
#central_or_satellite_answers = ['central', 'satellite']

for sim in simulations:
	#dir='/disks/eagle/%s/REFERENCE'%sim #in database it's the difference between ref and recal
	
	dir_str = '%s_RECALIBRATED'%sim

	for snap in snaps:
		for mlow, mhigh in masses:
			for sfr_answer in sfr_answers:
				#for central_or_satellite_answer in central_or_satellite_answers:

				print 'sim = %s, snap = %i, mlow=%.02e, sfr=%s'%(sim,snap,mlow,sfr_answer)#, , galaxy type = %s central_or_satellite_answer)

				gal_IDs, gals = halopicker(sim, snap, mlow, mhigh, sfr_answer)#, central_or_satellite_answer)

				ListToSave = []
				for gal_ID in gal_IDs:
					ListToSave.append(gal_ID)

				with open('IDs_%s_apsize=30_snap=%i_mlow=%.02e_sfr=%s.txt'%(dir_str, snap, mlow, sfr_answer), 'a') as f: #, central_or_satellite_answer
				    for item in ListToSave:
					f.write("%.0f\n" % item)

				#empty space in .txt file to delineate between directories
				with open('IDs_%s_apsize=30_snap=%i_mlow=%.02e_sfr=%s.txt'%(dir_str, snap, mlow, sfr_answer), 'a') as f: #, central_or_satellite_answer
					f.write("      \n")

				with open('gals_%s_apsize=30_snap=%i_mlow=%.02e_sfr=%s.txt'%(dir_str, snap, mlow, sfr_answer), 'a') as f: # , central_or_satellite_answer
				    for item in gals:
					f.write("%i %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.0f %.0f\n" % tuple(item)) #10 elements in gals row
