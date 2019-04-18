from __future__ import division
import make_maps_v3_master as mmap
import numpy as np
import pandas as pd
import itertools
import csv
import shutil
import glob
import math
#from read_eagle_files import *
from random import randint, shuffle

'''
mlow and mhigh as function arguments for getting array of galaxies
'''

def halopicker(sim, snap, mlow, mhigh, is_it_star_forming): #num_of_parameters, #num_of_galaxies

	'''
	snap can either be 12, 14, or 28
	N is number of galaxies requested
	num_of_parameters can be 1 or 2. need percentile brackets for each parameter requested
	'''	

	column_names = ['Galaxy_ID','sub_Mass_Total','sub_Mass_Star','sub_SFR','sub_gcom_x','sub_gcom_y','sub_gcom_z','fof_critR200','sub_hmr_Star','fof_snap']
	data = pd.read_csv('EAGLE_queries/halos_snap=%i_%s.csv'%(snap,sim), header=None)
	data.columns = column_names

	if num_of_parameters == 1:
		print "The parameters available are sub_Mass_Total (M_{Sun}), sub_Mass_Star (M_{Sun}), sub_SFR (M_{Sun}/yr), sub_gcom_x (cMpc), sub_gcom_y (cMpc), sub_gcom_z (cMpc), fof_critR200 (pkpc)"
		parameter1 = str(raw_input("What is the parameter you want to incorporate? "))

		if parameter1 == 'sub_Mass_Total' or parameter1 == 'sub_Mass_Star' or parameter1 == 'fof_critR200' or parameter1 == 'sub_SFR':

			value1_low = float(raw_input("Minimum value for this parameter? "))
			value1_high = float(raw_input("Maximum value for this parameter? "))

		else:

			percentile1_low = float(raw_input("Minimum percentile for this parameter [0,1]? "))
			percentile1_high = float(raw_input("Maximum percentile for this parameter [0,1]? "))
			value1_low = data.loc[:,"%s"%parameter1].quantile(percentile1_low)
			value1_high = data.loc[:,"%s"%parameter1].quantile(percentile1_high)

		#filters out the rows of the data frame that meet the desired requirements
		data = data.loc[data["%s"%parameter1] > value1_low]
		data = data.loc[data["%s"%parameter1] < value1_high]

	if num_of_parameters == 2:
		print "The parameters available are sub_Mass_Total, sub_Mass_Star, sub_SFR, sub_gcom_x, sub_gcom_y, sub_gcom_z, fof_critR200"
		parameter1 = str(raw_input("What is the parameter you want to incorporate? "))

		if parameter1 == 'sub_Mass_Total' or parameter1 == 'sub_Mass_Star' or parameter1 == 'fof_critR200' or parameter1 == 'sub_SFR':

			value1_low = float(raw_input("Minimum value for this parameter? "))
			value1_high = float(raw_input("Maximum value for this parameter? "))

		else:

			percentile1_low = float(raw_input("Minimum percentile for this parameter [0,1]? "))
			percentile1_high = float(raw_input("Maximum percentile for this parameter [0,1]? "))
			value1_low = data.loc[:,"%s"%parameter1].quantile(percentile1_low)
			value1_high = data.loc[:,"%s"%parameter1].quantile(percentile1_high)

		parameter2 = str(raw_input("What is the next parameter you want to incorporate? "))
		
		if parameter2 == 'sub_Mass_Total' or parameter2 == 'sub_Mass_Star' or parameter2 == 'fof_critR200'or parameter2 == 'sub_SFR':

			value2_low = float(raw_input("Minimum value for this parameter? "))
			value2_high = float(raw_input("Maximum value for this parameter? "))

		else:

			percentile2_low = float(raw_input("Minimum percentile for this parameter [0,1]? "))
			percentile2_high = float(raw_input("Maximum percentile for this parameter [0,1]? "))
			value2_low = data.loc[:,"%s"%parameter1].quantile(percentile1_low)
			value2_high = data.loc[:,"%s"%parameter1].quantile(percentile1_high)

		#filters out the rows of the data frame that meet the desired requirements
		data = data.loc[data["%s"%parameter1] > value1_low]
		data = data.loc[data["%s"%parameter1] < value1_high]
		data = data.loc[data["%s"%parameter2] > value2_low]
		data = data.loc[data["%s"%parameter2] < value2_high]

	galaxy_IDs = []
	galaxies = []
	
	data_x, data_y = data.shape
	print 'number of relevant galaxies is', data_x

	flag = 0
	misses = 0

	while misses < 5*data_x:

		samp = data.sample(n=1)		
		element = samp.values.tolist()[0] ##not sure what the other things are that are saved in the samp object as defined above		
		element[0] = int(element[0]) #saves galaxy ID as an int
		g_ID = element[0]

		element[1] = float(element[1]) #ensuring large values are not saved as strings

		if g_ID not in galaxy_IDs:
			galaxy_IDs.append(g_ID)
			galaxies.append(element)
		else:
			misses += 1

	#galaxy_IDs = [int(x) for x in galaxy_IDs]
	return galaxy_IDs, galaxies

# basic box and projection region parameters (splitting into (sub)slices is done later)
simulations = ['L0025N0376']

for sim in simulations:
	#dir='/disks/eagle/%s/REFERENCE'%sim #in database it's the difference between ref and recal
	
	dir_str = '%s_REFERENCE'%sim

	snaps = [12, 19, 28]
	masses = [[10**8, 10**8.5], [10**8.5, 10**9], [10**9, 10**9.5]]
	sfr_answers = ['yes', 'no']

	for snap in snaps:
		for mlow, mhigh in masses:
			for sfr_answer in sfr_answers:

				print 'sim = %s, snap = %i, mlow=%.02e, sfr=%s'%(sim,snap,mlow,sfr_answer)

				gal_IDs, gals = halopicker(sim, snap, mlow, mhigh, sfr_answer)

				ListToSave = []
				ListToSave.append(snap) #saves the snapshot
				for gal_ID in gal_IDs:
					ListToSave.append(gal_ID)

				with open('IDs_%s_snap=%i_mlow=%.02e_sfr=%s.txt'%(dir_str, snap, mlow, sfr_answer), 'a') as f:
				    for item in ListToSave:
					f.write("%.0f\n" % item)

				#empty space in .txt file to delineate between directories
				with open('IDs_%s_snap=%i_mlow=%.02e_sfr=%s.txt'%(dir_str, snap, mlow, sfr_answer), 'a') as f:
					f.write("      \n")

				with open('gals_%s_snap=%i_mlow=%.02e_sfr=%s.txt'%(dir_str, snap, mlow, sfr_answer), 'a') as f:
				    for item in gals:
					item = tuple(item)
					f.write("%i %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f %.05f\n" % item) #10 elements in gals row
