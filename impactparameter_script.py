import numpy as np
import glob
from matplotlib.colors import LogNorm
import itertools
import random

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def ColdensImpactParameterPlotter(input_arrays, num_bins, covfracvalue): #boxsize in Mpc, array must be square

	pixs_in_bins = [[] for i in range(num_bins -1)] #initialize binning process
	output_to_display = np.zeros((num_bins-1,7))

	#combined_outputs = []

	for input_array in input_arrays:

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
			pixel_array[j,k] = np.sqrt((x_pixel-x_center)**2 + (y_pixel-y_center)**2)
			#combined_outputs.append(output_array[i,j])

		'''
		output_to_display has the impact parameter of the bin's center in pixel units as the first slice
		and the number of pixels that fit into that bin as the second slice,
		xth, yth, and zth percentile column densities saved as the third through fifth slices
		average column densities as sixth slice
		covering fractions (percentage of pixels in that bin whose value is higher than the median)
		'''

		PercentileList1 = np.linspace(0,100,num_bins)

		'''
		this part only needs to be done once
		'''

		PercentileList2 = [] #tells you the impact parameters (pixel units) that fit the desired percentiles
		for i in range(num_bins):        
			PercentileList2.append(np.percentile(pixel_array, PercentileList1[i]))

		for i in range(num_bins-1):
			r_min, r_max = PercentileList2[i], PercentileList2[i+1]
			output_to_display[i,0] = r_min + float(r_max-r_min)/2.
			output_to_display[i,0] = output_to_display[i,0]/float(pix/2.) #converts from pixel to r/R_virial

			for j,k in itertools.product(*ranges):
				if r_min < pixel_array[j,k] < r_max:
					output_to_display[i,1] += 1


		'''
		populates the total output which will be stored in pixs and pixs_in_bins, which are binned by percentile of pixel distance not physical distance
		'''

		for i in range(num_bins-1):
			r_min, r_max = PercentileList2[i], PercentileList2[i+1]
			pixs_in_bins[i] = [output_array[j,k] for j,k in itertools.product(*ranges) if PercentileList2[i] < pixel_array[j,k] and PercentileList2[i+1] > pixel_array[j,k]]

		'''
		for j,k in itertools.product(*ranges):
		if r_min < pixel_array[j,k] < r_max:
		pixs.append(output_array[j,k])
		'''

		'''
		median_output_array = np.percentile(combined_outputs, 50)
		ninezero_output_array = np.percentile(combined_outputs, 90)
		'''

	for i in range(len(pixs_in_bins)):

		output_to_display[i,2] = np.percentile(np.asarray(pixs_in_bins[i]), 25)
		output_to_display[i,3] = np.percentile(np.asarray(pixs_in_bins[i]), 50)
		output_to_display[i,4] = np.percentile(np.asarray(pixs_in_bins[i]), 75)
		output_to_display[i,5] = np.mean(np.asarray(pixs_in_bins[i]))

		num_above_covfracvalue = np.zeros(num_bins-1)

		for pix in pixs_in_bins[i]:
			if pix > covfracvalue:
				num_above_covfracvalue[i] += 1

		output_to_display[i,6] = num_above_covfracvalue[i]/len(pixs_in_bins[i])

	return 1, output_to_display # 1 is placeholder for output_array

'''
simnum = 'L0025N0376'
snaps = [12, 28]
masses = [['1.00e+08', '$10^{8} M_{\odot} < M_{\star} < 10^{8.5} M_{\odot}$'], ['3.16e+08', '$10^{8.5} M_{\odot} < M_{\star} < 10^{9} M_{\odot}$'], ['1.00e+09','$10^{9} M_{\odot} < M_{\star} < 10^{9.5} M_{\odot}$']]
ions = [['hneutralssh', 'H I'], ['si4', 'Si IV'], ['c4', 'C IV']]

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')	

mass_label = r'$10^{8} M_{\odot} < M_{\star} < 10^{8.5} M_{\odot}$'

for snap in snaps:
	if snap == 12:
		redshift = '3.02'
	if snap == 28:
		redshift = '0'

	for ion, ion_label in ions:

		print 'snap, ion are', snap, ion

		if ion == 'hneutralssh':
			covfracvalue = 1e18
		else:
			covfracvalue = 1e12

		gal = np.loadtxt('/home/cook/Desktop/CGM-Research/package_HsmlAndProject_pywrapper/python_wrapper/SampleGalaxies/SampleGalaxies_TxtFiles/gals_%s_REFERENCE_n=1_snap=%i_mlow=1.00e+08.txt'%(simnum,snap), ndmin=1,unpack=True)
		projections = glob.glob('/net/quasar/data2/cook/temp/SliceThicknessConvergenceTests/coldens_%s_*_%s_*.npz'%(ion, snap))
		projections = sorted(projections)

		Y4_thinnest = []
	
		for projection in projections:

			print 'projection is', projection

			nn = 40

			Input = np.load(projection)['arr_0']
			Output = ColdensImpactParameterPlotter(Input, nn, covfracvalue)

			X, Y1, Y2, Y3, Y4 = Output[1][:,0], Output[1][:,2], Output[1][:,3], Output[1][:,4], Output[1][:,6]

			split_projection = projection.split('_')
			for element in split_projection:
				if element.endswith('slice'):
					thickness = element[:-5]

			if float(thickness) < 1.:
		
				Y4_thinnest = Y4
				#print Y4[0]
				#plt.plot(X, Y4, label=r'$\Delta z = %s$ cMpc'%str(thickness))

			if float(thickness) > 1.:

				if len(Y4_thinnest) > 0: 
	
					Y = [abs(Y4[i]-Y4_thinnest[i]) for i in range(len(Y4))]
					plt.semilogy(X, Y, linestyle = '--', label=r'$\Delta z = %s$ cMpc'%str(thickness))
				
		plt.text(0.9, 0.5, ion_label, fontsize = 12)
		plt.text(0.9, 0.4, mass_label, fontsize = 12)
		plt.text(0.9, 0.3, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)			

		plt.xlabel(r'$r/R_{crit, 200}$', fontsize=16)
		plt.legend(loc='upper right',fontsize=12)	
		plt.tight_layout()
		plt.savefig('CoveringFractionDifferences_%s_snap=%i_mlow=1.00e+08.pdf'%(ion,snap))
		plt.close()	

n_bins = 3
bins = ['bin'+str(i) for i in range(1,n_bins+1)]

masses = ['$10^{8} M_{\odot} < M_{\star} < 10^{8.5} M_{\odot}$', '$10^{8.5} M_{\odot} < M_{\star} < 10^{9} M_{\odot}$', '$10^{9} M_{\odot} < M_{\star} < 10^{9.5} M_{\odot}$']
cov_frac_list = [[] for i in range(len(bins))]
ionList = ['si4', 'hneutralssh']

snaps = [12, 28]

gals = np.loadtxt("/data1/cook/Snap=%i_Results/gals_forslicethickness_convergence_snap=%i.txt"%(snap, snap))

pixels = [[] for i in range(n_bins)]
for i in range(len(bins)):
	conv_bin = bins[i]
	myprojections = glob.glob("/data1/cook/Snap=14_Results/12March_SliceThickness_ConvergenceTests/%s/coldens_%s_*.npz"%(conv_bin, 'si4'))
	for projection in myprojections:
		split_projection = projection.split('_')
		for element in split_projection:
			if element.endswith('pix'):
	    			pixs = element[:-3]
				pixelselement = pixels[i]
				if pixs not in pixelselement:
					pixelselement.append(pixs)
			

for snap in snaps:

	if snap == 28:
		redshift = '0'
	if snap == 12:
		redshift = '3.02'

	for ion in ionList:

		if ion == 'hneutralssh':
			ion_label = 'H I'
		if ion == 'si4':
			ion_label = 'Si IV'

		for i in range(len(bins)):

			conv_bin = bins[i]
			mass_label = masses[i]
			pixs = pixels[i]

			pixs_ints = [int(pix) for pix in pixs]
		

			for pix in pixs:

				pixstr = str(pix) + 'pix'

				myprojections = glob.glob("/net/quasar/data2/cook/temp/SliceThicknessConvergenceTests/coldens_%s_*_%s_*.npz"%(conv_bin, ion, pixstr))

				gal = gals[i]
				ID, ap_Mass_Star, sub_Mass_Star, ap_SFR, fof_gcom_x, fof_gcom_y, fof_gcom_z, fof_critR200, sub_hmr_Star, fof_snap = gal

				#finding median of single halo projection

				thicknesses = []

				for projection in myprojections:

					split_projection = projection.split('_')

					for element in split_projection:
						if element.endswith('slice'):
			    				thickness = element[:-5]
							thicknesses.append(float(thickness))

				thinnest_slice_index = np.argmin(thicknesses)
				thinnest_slice = myprojections[thinnest_slice_index]
				thinnest_slice_array = np.load(projection)['arr_0']
			
				if ion == 'hneutralssh':
					
					covfracvalue = 1e18#np.percentile(np.asarray(10**thinnest_slice_array), 90)

				if ion == 'si4':
					
					covfracvalue = 1e12#np.percentile(np.asarray(10**thinnest_slice_array), 50)

				
				#original portion of the script

				Inputs = []

				plt.rc('text', usetex = True)
				plt.rc('font', family = 'serif')	
				plt.figure()

				label_counter = 0

				for projection in myprojections:

					split_projection = projection.split('_')

					for element in split_projection:
						if element.endswith('slice'):
			    				thickness = element[:-5]

					nn = 40

					Input = np.load(projection)['arr_0']
					Output = ColdensImpactParameterPlotter(Input, nn, covfracvalue)

					X, Y1, Y2, Y3, Y4 = Output[1][:,0], Output[1][:,2], Output[1][:,3], Output[1][:,4], Output[1][:,6]		

					plt.figure()
					plt.title(r'Si IV column density, %s, %s'%(s, conv_bin), fontsize = 14)
					plt.semilogy(X_si4, Y1_si4, 'k--', label=r'Si IV, 50th percentile')
					plt.semilogy(X_si4, Y2_si4, 'k', label=r'Si IV, 70th percentile')
					plt.semilogy(X_si4, Y3_si4, 'k--', label=r'Si IV, 90th percentile')

					plt.xlabel(r'$r/R_{crit,200}$',fontsize = 14)
					plt.legend(loc='best',fontsize=8)
					plt.savefig('ImpactParameterPlot_si4_%s_%s.jpg'%(s, conv_bin))
					plt.close()

					#plt.title('%s Covering Fractions, %s, $z = %s$'%(ion_label, mass_label, redshift) , fontsize = 10)
					#subplot_count += 1

					if float(thickness) > 1.:

						plt.plot(X, Y4, linestyle = '--', label=r'$\Delta z = %s$ cMpc'%thickness)
						
						if int(pix) == min(pixs_ints):	

							plt.plot(X, Y4, linestyle = '--', label=r'$\Delta z = %s$ cMpc'%thickness)
				
						
						if int(pix) == max(pixs_ints):			

							plt.plot(2*X, Y4, linestyle = '--', label=r'$\Delta z = %s$ cMpc'%thickness)

					if float(thickness) < 1.:
			
						plt.plot(X, Y4, label=r'$\Delta z = %s$ cMpc'%thickness)

						if int(pix) == min(pixs_ints):

							plt.plot(X, Y4, label=r'$\Delta z = %s$ cMpc'%thickness)
				
						if int(pix) == max(pixs_ints):			

							plt.plot(2*X, Y4, label=r'$\Delta z = %s$ cMpc'%thickness)						

					if int(pix) == min(pixs_ints):

						c=np.random.rand(3,)

						plt.semilogy(X, Y1, color = c, linestyle='--')
						plt.semilogy(X, Y2, color = c, label=r'$\Delta z = %s$ cMpc'%thickness)
						plt.semilogy(X, Y3, color = c, linestyle='--')	
			
					if int(pix) == max(pixs_ints):			

						c=np.random.rand(3,)

						plt.semilogy(2*X, Y1, color = c, linestyle='--')
						plt.semilogy(2*X, Y2, color = c, label=r'$\Delta z = %s$ cMpc'%thickness)
						plt.semilogy(2*X, Y3, color = c, linestyle='--')

					if label_counter == 0:

						if ion == 'hneutralssh':

								plt.text(0.9, 1000*covfracvalue, ion_label, fontsize = 12)
								plt.text(0.9, 3.162*100*covfracvalue, mass_label, fontsize = 12)
								plt.text(0.9, 100*covfracvalue, r'$\langle R_{crit, 200} \rangle = %.03f$  pkpc'%fof_critR200, fontsize = 12)
								plt.text(0.9, 3.162*10*covfracvalue, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)
					
								label_counter = 1

						if ion == 'si4':

							plt.text(0.9, 10000*covfracvalue, ion_label, fontsize = 12)
							plt.text(0.9, 1000*covfracvalue, mass_label, fontsize = 12)
							plt.text(0.9, 100*covfracvalue, r'$\langle R_{crit, 200} \rangle = %.03f$  pkpc'%fof_critR200, fontsize = 12)
							plt.text(0.9, 10*covfracvalue, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)

						if int(pix) == min(pixs_ints):

							if ion == 'hneutralssh':

								plt.text(0.9, 1000*covfracvalue, ion_label, fontsize = 12)
								plt.text(0.9, 3.162*100*covfracvalue, mass_label, fontsize = 12)
								plt.text(0.9, 100*covfracvalue, r'$\langle R_{crit, 200} \rangle = %.03f$  pkpc'%fof_critR200, fontsize = 12)
								plt.text(0.9, 3.162*10*covfracvalue, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)
					
								label_counter = 1

							if ion == 'si4':

								plt.text(0.9, 10000*covfracvalue, ion_label, fontsize = 12)
								plt.text(0.9, 1000*covfracvalue, mass_label, fontsize = 12)
								plt.text(0.9, 100*covfracvalue, r'$\langle R_{crit, 200} \rangle = %.03f$  pkpc'%fof_critR200, fontsize = 12)
								plt.text(0.9, 10*covfracvalue, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)
					
								label_counter = 1

						if int(pix) == max(pixs_ints):			

							if ion == 'hneutralssh':

								plt.text(1.8, 1000*covfracvalue, ion_label, fontsize = 12)
								plt.text(1.8, 3.162*100*covfracvalue, mass_label, fontsize = 12)
								plt.text(1.8, 100*covfracvalue, r'$\langle R_{crit, 200} \rangle = %.03f$  pkpc'%fof_critR200, fontsize = 12)
								plt.text(1.8, 3.162*10*covfracvalue, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)
					
								label_counter = 1

							if ion == 'si4':

								plt.text(1.8, 10000*covfracvalue, ion_label, fontsize = 12)
								plt.text(1.8, 1000*covfracvalue, mass_label, fontsize = 12)
								plt.text(1.8, 100*covfracvalue, r'$\langle R_{crit, 200} \rangle = %.03f$  pkpc'%fof_critR200, fontsize = 12)
								plt.text(1.8, 10*covfracvalue, r'$c^{\star} = %e$ '%covfracvalue, fontsize = 12)
					
								label_counter = 1

				if ion == 'hneutralssh':
					
					plt.ylim(1e14,1e22)#np.percentile(np.asarray(10**thinnest_slice_array), 90)

				if ion == 'si4':
					
					plt.ylim(1e4,1e18)#np.percentile(np.asarray(10**thinnest_slice_array), 50)

				plt.xlabel(r'$r/R_{crit, 200}$', fontsize=16)
				plt.legend(loc='upper left',fontsize=12)	
				plt.tight_layout()
				plt.savefig('ImpactParameter_%s_snap=14_%s_%s.pdf'%(ion,conv_bin,pixstr))
				plt.close()	

		plt.figure()
		plt.title('Covering Fractions, %s'%conv_bin , fontsize = 14)

		for i in range(len(bins)):

			bb = bins[i]
			xxs, yys = cov_frac_list[i]
			plt.plot(xxs, yys, label=r'%s'%bb)

'''
