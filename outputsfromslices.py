from __future__ import division
import numpy as np
import math
import glob
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
from impactparameter_script import ColdensImpactParameterPlotter
import os

npix_x = int(2**13)

'''
all of the possible contingencies for radial profiles
https://matplotlib.org/users/dflt_style_changes.html for color scheme
'''

ions = [['hneutralssh', 'H I'], ['si4', 'Si IV'], ['c4', 'C IV']]
#things = [['Density', r'$\rho$'], ['Temperature', 'T'], ['SmoothedMetallicity', 'Z']]
ion_or_other = 'other' #ion

masses = [ ['1.00e+09','$9< \log(M/M_{\odot}) < 9.5$', 'C3'], ['3.16e+09', '$9.5< \log(M/M_{\odot}) < 10$', 'C2'], ['1.00e+10', '$10 < \log(M/M_{\odot}) < 10.5$', 'C1'], ['3.16e+10', '$10.5 < \log(M/M_{\odot}) < 11$', 'C4'], ['1.00e+11', '$11 < \log(M/M_{\odot}) < 11.5$', 'C9'] ]

options = [ ['WholeBox', r'$\Delta Z$ = 25 cMpc'], ['8Slice', r'$\Delta Z$ = 3.125 cMpc'] ]
sfr_answers = ['yes', 'no']
cent_or_sat_options = ['central', 'satellite']
profile_types = ['covfrac'] #'coldens', 
binning_methods = ['virial', 'physical'] 

box_size, sim = 25., 'L0025N0752'

#for L0025N0752
snaps = [[11, 3.53], [19, 1.], [28, 0.]]
sim_str = 'L0025N0752RECALIBRATED'
recal_ref = 'RECALIBRATED'

def figure_maker_ions(ions, snaps, coldens_or_covfrac, physical_or_virial):
	
	'''
	initializes figure
	'''

	plt.rc('text', usetex = True)
	plt.rc('font', family = 'serif')

	fig = plt.figure()
	axes = []

	for ion, ion_label in ions:

		for snap, redshift in snaps:

			eps, divider_x, divider_y = 0.09, 3.3, 3.7
			eps_legend = 0.12

			y0 = eps_legend + eps

			if snap == 11:
				x0 = eps
			if snap == 19:
				x0 = eps + (1-eps)/divider_x
			if snap == 28:
				x0 = eps + 2*(1-eps)/divider_x

			if ion == 'hneutralssh':
				y0 = eps + eps_legend
			if ion == 'c4':
				y0 = eps + eps_legend + (1-eps)/divider_y
			if ion  == 'si4':
				y0 = eps + eps_legend + 2*(1-eps)/divider_y

			ax = fig.add_axes([x0, y0, (1-eps)/divider_x, (1-eps)/divider_y])
		        ax.tick_params(left=True, bottom=True, right=False, top=False, labelsize='small')

			if physical_or_virial == 'virial':

				ticks = [0.25, 0.5, 0.75, 1.0, 1.25]
				ticks_str = [str(x) for x in ticks]

				ax.set_xlim(0, np.sqrt(2))

			if physical_or_virial == 'physical':

				ticks = [50, 100, 150, 200, 250]
				ticks_str = [str(x) for x in ticks]

				ax.set_xlim(0, 300.)

			ax.set_xticks(ticks)			
			ax.set_xticklabels(ticks_str)

			if coldens_or_covfrac == 'coldens':

				ax.set_yscale('log')
			
				if ion == 'c4':
					ax.set_ylim(1e9,1e16)
				if ion == 'si4':
					ax.set_ylim(1e6,1e15)
				if ion == 'hneutralssh':
					ax.set_ylim(1e11,1e21)


			if coldens_or_covfrac == 'covfrac':

				ax.set_ylim(0, 1)
			        ax.set_yticks([0.25, 0.5, 0.75])
		        	ax.set_yticklabels(['0.25', '0.5', '0.75'])

				if ion == 'hneutralssh' or ion == 'h1':
					
					cfv_label = '10^{17}'

				if ion == 'si4':
					
					cfv_label = '10^{13}'

				if ion == 'c4':
					
					cfv_label = '10^{14}'

			if snap != 11:

				ax.yaxis.set_major_formatter(NullFormatter())
				ax.yaxis.set_minor_formatter(NullFormatter())

			if ion != 'hneutralssh':

				ax.xaxis.set_major_formatter(NullFormatter())
				ax.xaxis.set_minor_formatter(NullFormatter())

			if snap == 11:
				redshift_label = r'$z=3.53$'

			if snap == 19:
				redshift_label = r'$z=1$'

			if snap == 28:
				redshift_label = r'$z=0$'	


			if snap == 11:				
				
		                if ion == 'c4':
	
					if coldens_or_covfrac == 'coldens':
						ax.set_ylabel(r'$\log(N$[cm$^{-2}$]$)$', fontsize=12)
					if coldens_or_covfrac == 'covfrac':
						ax.set_ylabel(r'$F(N_{th},r)$', fontsize=12)

				if ion == 'hneutralssh':
					ax.annotate(option_label, xy = (0.1, 0.05), xycoords = 'axes fraction', fontsize=8)
					ax.annotate('star-forming: ' + sfr_answer, xy = (0.1, 0.15), xycoords = 'axes fraction', fontsize=8)
					ax.annotate('galaxy type: ' + cent_or_sat, xy = (0.1, 0.25), xycoords = 'axes fraction', fontsize=8)

			if snap == 19:		

		                if ion == 'hneutralssh':
					#ax.legend(bbox_to_anchor=(1.2, -0.65, 1., 0.35), framealpha=1, ncol=3, fontsize=10)	

					if physical_or_virial == 'virial':
					
						ax.set_xlabel(r'$r/R_{vir}$', fontsize=12)

					if physical_or_virial == 'physical':
					
						ax.set_xlabel(r'$r$ (pkpc)', fontsize=12)

			if snap == 28:
				
				ax.annotate(ion_label, xy = (0.5, 0.75), xycoords = 'axes fraction', fontsize = 10)

				if coldens_or_covfrac == 'covfrac':

					ax.annotate(r'$N_{th} = %s$'%cfv_label, xy = (0.5, 0.6), xycoords = 'axes fraction', fontsize = 10)

			if ion == 'si4':

				ax.set_title(redshift_label, fontsize=12)

			axes.append([ax, ion, snap])

	return fig, axes

def figure_maker_other(things, snaps, coldens_or_covfrac, physical_or_virial):
	
	'''
	initializes figure
	'''

	plt.rc('text', usetex = True)
	plt.rc('font', family = 'serif')

	fig = plt.figure()
	axes = []

	for thing, thing_label in things:

		for snap, redshift in snaps:

			eps, divider_x, divider_y = 0.09, 3.3, 3.7
			eps_legend = 0.12

			y0 = eps_legend + eps

			if snap == 11:
				x0 = eps
			if snap == 19:
				x0 = eps + 1*(1-eps)/divider_x
			if snap == 28:
				x0 = eps + 2*(1-eps)/divider_x

			if thing == 'Density':
				y0 = eps + eps_legend
			if thing == 'Temperature':
				y0 = eps + eps_legend + (1-eps)/divider_y
			if thing  == 'SmoothedMetallicity':
				y0 = eps + eps_legend + 2*(1-eps)/divider_y

			ax = fig.add_axes([x0, y0, (1-eps)/divider_x, (1-eps)/divider_y])
		        ax.tick_params(left=True, bottom=True, right=False, top=False, labelsize='small')

			if physical_or_virial == 'virial':

				ticks = [0.25, 0.5, 0.75, 1.0, 1.25]
				ticks_str = [str(x) for x in ticks]

				ax.set_xlim(0, np.sqrt(2))

			if physical_or_virial == 'physical':

				ticks = [50, 100, 150, 200, 250]
				ticks_str = [str(x) for x in ticks]

				ax.set_xlim(0, 300.)

			ax.set_xticks(ticks)			
			ax.set_xticklabels(ticks_str)

			if coldens_or_covfrac == 'coldens':

				ax.set_yscale('log')
			
				if thing == 'Density':
					ax.set_ylim(1e-6,1e1)
				if thing == 'SmoothedMetallicity':
					ax.set_ylim(1e-2, 1e1)
				if thing == 'Temperature':
					ax.set_ylim(1e3,10**7.5)

			if snap != 11:

				ax.yaxis.set_major_formatter(NullFormatter())
				ax.yaxis.set_minor_formatter(NullFormatter())

			if thing != 'Density':

				ax.xaxis.set_major_formatter(NullFormatter())
				ax.xaxis.set_minor_formatter(NullFormatter())

			if snap == 11:
				redshift_label = r'$z=3.53$'

			if snap == 19:
				redshift_label = r'$z=1$'

			if snap == 28:
				redshift_label = r'$z=0$'	


			if snap == 11:				
				
		                if thing == 'Temperature':

					ax.set_ylabel(r'$\log(T$ [K] $)$', fontsize=12)

				if thing == 'Density':

					ax.set_ylabel(r'$\log(n_{H}$ cm$^{-3}$)', fontsize=12)

					ax.annotate(option_label, xy = (0.1, 0.05), xycoords = 'axes fraction', fontsize=8)
					ax.annotate('star-forming: ' + sfr_answer, xy = (0.1, 0.15), xycoords = 'axes fraction', fontsize=8)
					ax.annotate('galaxy type: ' + cent_or_sat, xy = (0.1, 0.25), xycoords = 'axes fraction', fontsize=8)

				if thing == 'SmoothedMetallicity':

					ax.set_ylabel(r'$\log(Z$ [$Z_{\odot}$]$)$', fontsize=12)


			if snap == 19:		

		                if thing == 'Density':
					#ax.legend(bbox_to_anchor=(1.2, -0.65, 1., 0.35), framealpha=1, ncol=3, fontsize=10)	

					if physical_or_virial == 'virial':
					
						ax.set_xlabel(r'$r/R_{vir}$', fontsize=12)

					if physical_or_virial == 'physical':
					
						ax.set_xlabel(r'$r$ (pkpc)', fontsize=12)

			if snap == 28:
				
				ax.annotate(thing_label, xy = (0.75, 0.75), xycoords = 'axes fraction', fontsize = 10)

				if coldens_or_covfrac == 'covfrac':

					if thing == 'Temperature':

						ax.annotate(r'$T_{th} = %s$ K'%cfv_label, xy = (0.5, 0.6), xycoords = 'axes fraction', fontsize = 10)

					if thing == 'SmoothedMetallicity':

						ax.annotate(r'$Z_{th} = %s$'%cfv_label, xy = (0.5, 0.6), xycoords = 'axes fraction', fontsize = 10)

					if thing == 'Density':

						ax.annotate(r'$\rho_{th} = %s$ '%cfv_label, xy = (0.5, 0.6), xycoords = 'axes fraction', fontsize = 10)	

			if thing == 'SmoothedMetallicity':

				ax.set_title(redshift_label, fontsize=12)

			if thing == 'Temperature':

				#ion_levels = [7.94e4, 1.26e5, 3.16e6, 6.31e6]

				ax.axhline(y=7.94e4, linewidth=0.25, linestyle='--', color='k')
				ax.axhline(y=1.26e5, linewidth=0.25, linestyle='--', color='k')
				ax.axhline(y=3.16e6, linewidth=0.25, linestyle='--', color='k')
				ax.axhline(y=6.31e6, linewidth=0.25, linestyle='--', color='k')

			axes.append([ax, thing, snap])

	return fig, axes

def get_inputs(slices, gals, option):

	'''
	allocates galaxy into appropriate slice
	for ColdensImpactParameterPlotter
	saves galaxy class data (e.g., how many galaxies are too close to xy boundary)
	'''

	rho = (box_size*1000.)/(1+redshift) / npix_x
	one_over_rho = 1/rho

	Inputs = []
	fof_critR200_vals = gals[:,7]
	r_pixs = [ one_over_rho * fof_critR200 for fof_critR200 in fof_critR200_vals ]

	for sl in slices:

		slice_array = np.load(sl)['arr_0']
		split_sl = sl.split('_')
		
		zSlicecen = 12.5 #default

		#picks out center z value of each slice
		for piece in split_sl:
			if 'zcen' in piece:
				zSlicecen = float(piece.replace('zcen',''))

		for gal in gals:

			Galaxy_ID, sub_Mass_Total, sub_Mass_Star, sub_SFR, sub_gcom_x, sub_gcom_y, sub_gcom_z, fof_critR200, sub_sgn, fof_snap = gal
			
			x_halo, y_halo, z_halo = sub_gcom_x, sub_gcom_y, sub_gcom_z #in cMpc, from EAGLE database
			r_halo = fof_critR200 #in pkpc, from EAGLE database
			snap = int(fof_snap)

			r_halo = r_halo/(1000.)*(1+redshift) #converts from pkpc to cMpc

			#makes sure to catch galaxies only in that slice and not too close to the boundary
		
			if option == 'WholeBox':

				x_pix, y_pix = int(math.floor((x_halo/box_size)*npix_x)), int(math.floor((y_halo/box_size)*npix_x))

				rval = one_over_rho * fof_critR200
				r_pix = int(math.ceil(rval))

				if r_pix %2 == 0:
					r_pix += 1

				x_min = int(x_pix - int(math.floor(r_pix/2)))
				x_max = int(x_pix + int(math.floor(r_pix/2)))
				y_min = int(y_pix - int(math.floor(r_pix/2)))
				y_max = int(y_pix + int(math.floor(r_pix/2)))

				n_shift = 100

				if x_min < 0:
		
					slice_array = np.roll(slice_array, n_shift, axis=1)
					x_min += n_shift
					x_max += n_shift

				if x_max > npix_x:
		
					slice_array = np.roll(slice_array, -n_shift, axis=1)
					x_min -= n_shift
					x_max -= n_shift

				if y_min < 0:
		
					slice_array = np.roll(slice_array, n_shift, axis=0)
					y_min += n_shift
					y_max += n_shift

				if y_max > npix_x:
		
					slice_array = np.roll(slice_array, -n_shift, axis=0)
					y_min -= n_shift
					y_max -= n_shift

				Input = slice_array[x_min:x_max, y_min:y_max]				

				I_x, I_y = Input.shape

				if I_x == I_y and I_x > 0:
					Inputs.append([fof_critR200, Input])

				else:
					print 'got one! offender is %s, %s, %i, %s, %i'%(sim, recal_ref, snap, mlow, Galaxy_ID)
					print 'xmin xmax ymin ymax are', x_min, x_max, y_min, y_max
					print ''

			if option != 'WholeBox':					
				
				if abs(z_halo - zSlicecen) < abs(slice_thickness/2. - r_halo):

					x_pix, y_pix = int(math.floor((x_halo/box_size)*npix_x)), int(math.floor((y_halo/box_size)*npix_x))

					rval = one_over_rho * fof_critR200
					r_pix = int(math.ceil(rval))

					if r_pix %2 == 0:
						r_pix += 1

					x_min = int(x_pix - int(math.floor(r_pix/2)))
					x_max = int(x_pix + int(math.floor(r_pix/2)))
					y_min = int(y_pix - int(math.floor(r_pix/2)))
					y_max = int(y_pix + int(math.floor(r_pix/2)))

					n_shift = 100

					if x_min < 0:
			
						slice_array = np.roll(slice_array, n_shift, axis=1)
						x_min += n_shift
						x_max += n_shift

					if x_max > npix_x:
			
						slice_array = np.roll(slice_array, -n_shift, axis=1)
						x_min -= n_shift
						x_max -= n_shift

					if y_min < 0:
			
						slice_array = np.roll(slice_array, n_shift, axis=0)
						y_min += n_shift
						y_max += n_shift

					if y_max > npix_x:
			
						slice_array = np.roll(slice_array, -n_shift, axis=0)
						y_min -= n_shift
						y_max -= n_shift

					Input = slice_array[x_min:x_max, y_min:y_max]				

					I_x, I_y = Input.shape

					if I_x == I_y and I_x > 0:
						Inputs.append([fof_critR200, Input])
					else:
						print 'got one! offender is %s, %s, %i, %s, %i'%(sim, recal_ref, snap, mlow, Galaxy_ID)
						print 'xmin xmax ymin ymax are', x_min, x_max, y_min, y_max
						print ''		

	

	return Inputs, int(math.ceil(np.sqrt(2.)/4.*np.mean(r_pixs)))


def galaxy_list_finder(sim, recal_ref, snap, mlow, sfr_answer, cent_or_sat):
	
	'''
	finds appropriate text file containing galaxy data
	'''

	return '30MayGalaxyLists_byTotalMass/gals_%s_%s_apsize=30_snap=%i_mlow=%s_sfr=%s_%s.txt'%(sim, recal_ref, snap, mlow, sfr_answer, cent_or_sat)	

def slices_finder(ion, sim_str, snap, option_str):
	
	'''
	finds appropriate slice data
	'''

	return glob.glob("/net/quasar/data2/cook/temp/L0025N0752_8192pix_Projections/L0025N0752_Projectionscoldens_%s_%s_%i_*_%s_*.npz"%(ion, sim_str, snap, option_str))	

def slices_finder_other(thing, sim_str, snap, option_str):
	
	'''
	finds appropriate slice data
	'''

	return glob.glob("/net/quasar/data2/cook/temp/L0025N0752_Projections/%s_*_%s_%i_*_%s_*.npz"%(thing, sim_str, int(snap), option_str))	

Ngalaxies_table = []

plot_things = [ [option, option_label, sfr_answer, cent_or_sat, type_of_profile, binning_method] for option, option_label in options for sfr_answer in sfr_answers for cent_or_sat in cent_or_sat_options for type_of_profile in profile_types for binning_method in binning_methods ]

for plot_things_vals in plot_things:

	option, option_label, sfr_answer, cent_or_sat, type_of_profile, binning_method = plot_things_vals

	if option == 'WholeBox':
		num_slices = 1
		option_str = '25.0slice'

	if option == '8Slice':
		num_slices = 8
		option_str = '3.125slice'

	slice_thickness = box_size/num_slices

	fig, axes = figure_maker_ions(ions, snaps, type_of_profile, binning_method)
	#fig, axes = figure_maker_other(things, snaps, type_of_profile, binning_method)

	print ''
	print option
	print 'sfr: %s'%(sfr_answer)
	print 'galaxy type: %s'%(cent_or_sat)
	print 'profile type: %s'%(type_of_profile)
	print 'binning method: %s'%(binning_method)

	for snap, redshift in snaps:

		print ''
		print 'doing redshift = %.02f'%redshift

		for j in range(len(masses)):

			mlow, mass_label, color = masses[j]			
			print 'doing mass = %s'%mass_label

			gals_str = galaxy_list_finder(sim, recal_ref, snap, mlow, sfr_answer, cent_or_sat)					
			gals_file_size = os.stat(gals_str).st_size

			if gals_file_size > 100:

				gals = np.loadtxt(gals_str)

			if gals_file_size <= 100:
			
				print 'gals type: <= 1 galaxy, move on!'			


			flag = 0

			#for thing, thing_label in things:
			for ion, ion_label in ions:

				'''
				if len(gals) == 0:

					while flag == 0:
						Ngalaxies_table.append([sim,snap,mlow,sfr_answer,cent_or_sat, 0, 0])
						flag = 1
					continue

				if gals_file_size < 100:

					while flag == 0:
						Ngalaxies_table.append([sim,snap,mlow,sfr_answer,cent_or_sat, 1, 1])
						flag = 1
					continue
				'''

				if gals_file_size > 100:				
					slices = slices_finder(ion, sim_str, snap, option_str)
					#slices = slices_finder_other(thing, sim_str, snap, option_str)
					print slices

				if gals_file_size > 100:
					Inputs, radial_bins = get_inputs(slices, gals, option)

				'''
				if gals_file_size >= 100:
					while flag == 0:
						Ngalaxies_table.append([sim,snap,mlow,sfr_answer,cent_or_sat,len(Inputs),len(gals)])
						flag = 1
				'''

				if ion == 'hneutralssh' or ion == 'h1':
		
					covfracvalue = 1e17

				if ion == 'si4':
					
					covfracvalue = 1e13

				if ion == 'c4':
					
					covfracvalue = 1e14
				if gals_file_size > 100:
				
					Output = ColdensImpactParameterPlotter(Inputs, radial_bins, covfracvalue, binning_method, ion_or_other)
					X, Y0, Y1, Y2, Y3, Y4 = Output[1][:,0], Output[1][:,1], Output[1][:,2], Output[1][:,3], Output[1][:,4], Output[1][:,6]

				if gals_file_size <= 100:

					X, Y0, Y1, Y2, Y3, Y4 = [-1., -0.5], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.]

				for axis in axes:
					
					axis_a, ion_a, snap_a = axis
					#axis_a, thing_a, snap_a = axis

					if ion == ion_a and snap == snap_a:
						#if thing == thing_a and snap == snap_a:
						ax = axis_a											

				#25/50/75th percentile column density for radial bin

				'''
				mH = 1.674e-24

				if thing == 'Density':

					Y1 = [y/mH for y in Y1]
					Y2 = [y/mH for y in Y2]
					Y3 = [y/mH for y in Y3]

				Zsun = 0.012			

				if thing == 'SmoothedMetallicity':
	
					Y1 = [y/Zsun for y in Y1]
					Y2 = [y/Zsun for y in Y2]
					Y3 = [y/Zsun for y in Y3]
				'''

				if type_of_profile == 'coldens':

					ax.plot(X, Y1, color, linewidth=0.5, linestyle=':')	    
					ax.plot(X, Y2, color, linewidth=1, label=r'%s'%(mass_label))
					ax.plot(X, Y3, color, linewidth=0.5, linestyle=':')

				if type_of_profile == 'covfrac':

					#covering fraction
					ax.plot(X, Y4, color, linewidth=1, label=r'%s'%(mass_label))

				if snap == 19:		

					#if thing == 'Density':
					if ion == 'hneutralssh':

						ax.legend(bbox_to_anchor=(1.2, -0.65, 1., 0.35), framealpha=1, ncol=3, fontsize=10)

	if type_of_profile == 'coldens':

		plt.savefig('%s_%s_%s_sfr=%s_%s_%s.pdf'%(type_of_profile, sim, option, sfr_answer, cent_or_sat, binning_method), bbox_inches='tight')

	if type_of_profile == 'covfrac':

		plt.savefig('%s_%s_%s_sfr=%s_%s_%s.pdf'%(type_of_profile, sim, option, sfr_answer, cent_or_sat, binning_method), bbox_inches='tight')
	
	plt.close()

'''
with open('Ngalaxies_table_8Slice_L0025N0752.txt', 'w+') as f:
	f.write('sim,snap,mlow,sfr_answer,len(Inputs),len(gals)\n')
	for item in Ngalaxies_table:
		f.write('%s %i %s %s %s %i %i\n'%(tuple(item)))
'''
