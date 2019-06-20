import numpy as np
import glob
from matplotlib.colors import LogNorm
import itertools
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

box_size, N_pix = 25., 2**13

def ColdensImpactParameterPlotter(input_arrays, num_bins, covfracvalue, physical_or_virial, ion_or_other): #boxsize in Mpc, array must be square

	pixs_in_bins = [[] for i in range(num_bins-1)] #initialize binning process
	output_to_display = np.zeros((num_bins-1,7))

	flag = 0 #for getting radial distances right the first time
	
	PercentileList1 = np.linspace(0,100,num_bins)

	for radius_and_input in input_arrays:

		r_halo_proper, input_array = radius_and_input

		pix = input_array.shape[0] #number of pixels
		pixel_array = np.zeros((pix,pix))

		'''
		output_array will have the input_array as the first slice
		and the distance from the center of the array as the second
		'''
		output_array = 10**input_array

		ranges = [range(pix), range(pix)]
		x_center, y_center = float(pix/2.), float(pix/2.)

		for j,k in itertools.product(*ranges):
			x_pixel, y_pixel = j+0.5, k+0.5

			if physical_or_virial == 'virial':

				pixel_array[j,k] = np.sqrt((x_pixel-x_center)**2 + (y_pixel-y_center)**2)

			if physical_or_virial == 'physical':

				pixel_array[j,k] = r_halo_proper * np.sqrt((x_pixel-x_center)**2 + (y_pixel-y_center)**2)

		'''
		radial distance in r/Rvir or r (pkpc) is the zeroth slice
		and the number of pixels that fit into that bin as the first slice,
		xth, yth, and zth percentile column densities saved as the second through fourth slices
		average column densities as fifth slice
		covering fractions (percentage of pixels in that bin whose value is higher than a defined threshold value)
		'''

		if physical_or_virial == 'virial':

			#tells you the impact parameters (pixel units) that fit the desired percentiles
			PercentileList2 = [np.percentile(pixel_array, PercentileList1[i]) for i in range(num_bins)] 

		if physical_or_virial == 'physical':

			PercentileList2 = np.linspace(0.,300.,num_bins) #in pkpc

		for i in range(num_bins-1):
			r_min, r_max = PercentileList2[i], PercentileList2[i+1]

			if physical_or_virial == 'virial':

				output_to_display[i,0] = (r_min + float(r_max-r_min)/2.)/float(pix/2.) #converts from pixel to r/R_virial

			if physical_or_virial == 'physical':

				output_to_display[i,0] = r_min + float(r_max-r_min)/2.#converts from pixel to r/R_virial

		for i in range(num_bins-1):
			r_min, r_max = PercentileList2[i], PercentileList2[i+1]
			for j,k in itertools.product(*ranges):

				if physical_or_virial == 'virial':

					if r_min < pixel_array[j,k] < r_max:
						output_to_display[i,1] += 1

				if physical_or_virial == 'physical':

					if r_min < pixel_array[j,k] < r_max:
						output_to_display[i,1] += 1


		'''
		populates the total output which will be stored in pixs and pixs_in_bins, which are binned by percentile of pixel distance not physical distance
		'''

		for i in range(num_bins-1):
			r_min, r_max = PercentileList2[i], PercentileList2[i+1]
			pixs = pixs_in_bins[i]
			
			if ion_or_other == 'ion':

				vals_into_bin = [output_array[j,k] for j,k in itertools.product(*ranges) if PercentileList2[i] < pixel_array[j,k] and PercentileList2[i+1] > pixel_array[j,k]]

			if ion_or_other == 'other':

				vals_into_bin = [output_array[j,k] for j,k in itertools.product(*ranges) if PercentileList2[i] < pixel_array[j,k] and PercentileList2[i+1] > pixel_array[j,k] and output_array[j,k] != 0]

			for val in vals_into_bin:
				pixs.append(val)			
		
	'''
	get statistics
	'''

	for i in range(num_bins-1):

		pixs = pixs_in_bins[i]
		len_pixs = len(pixs_in_bins[i])

		if len(pixs) != 0:

			output_to_display[i,2] = np.percentile(np.asarray(pixs), 25)
			output_to_display[i,3] = np.percentile(np.asarray(pixs), 50)
			output_to_display[i,4] = np.percentile(np.asarray(pixs), 75)
			output_to_display[i,5] = np.mean(np.asarray(pixs))

			num_above_covfracvalue = np.zeros(num_bins-1)

			for pix in pixs:
				if pix > covfracvalue:
					num_above_covfracvalue[i] += 1

			output_to_display[i,6] = num_above_covfracvalue[i]/len(pixs)

		if len(pixs) == 0:

			#2 won't shop up on coldens or covfrac plots
			output_to_display[i,2] = np.NaN
			output_to_display[i,3] = np.NaN
			output_to_display[i,4] = np.NaN
			output_to_display[i,5] = np.NaN
			output_to_display[i,6] = np.NaN

	return 1, output_to_display # 1 is placeholder for output_array
