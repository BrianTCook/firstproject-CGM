from __future__ import division
import numpy as np
import math
import glob
#import rdists_sl_faster
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import NullFormatter
from impactparameter_script import ColdensImpactParameterPlotter

npix_x = int(2**13)
#snaps = [14]#, 28]
ions = ['hneutralssh', 'si4', 'c4']
ion_labels = ['H I', 'Si IV', 'C IV'] 

masses = [['1.00e+08', '$8< \log(M_{\star}/M_{\odot}) < 8.5$', 'r'], ['3.16e+08', '$8.5< \log(M_{\star}/M_{\odot}) < 9$', 'g'], ['1.00e+09','$9< \log(M_{\star}/M_{\odot}) < 9.5$', 'b']]


boxsize_simulations = [[25., 'L0025N0376']]

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

Ngalaxies_table = []

for box_size, sim in boxsize_simulations:	

        if sim == 'L0100N1504' or sim == 'L0025N0376':
		snaps = [12, 19, 28]	
		sim_str = sim
		recal_ref = 'REFERENCE'

	for i in range(len(ions)):
		
		fig = plt.figure()
				
		ion = ions[i]
		ion_label = ion_labels[i]

		for snap in snaps:

			slices = glob.glob("/net/quasar/data2/cook/temp/%s_8Slice_Projections/coldens_%s_%s_%i_*.npz"%(sim, ion, sim, snap))	

			eps, divider_x, divider_y = 0.09, 3.3, 3.5
			eps_legend = 0.06

			y0 = eps_legend + eps

			if snap == 12:
				x0 = eps
			if snap == 19:
				x0 = eps + (1-eps)/divider_x
			if snap == 28:
				x0 = eps + 2*(1-eps)/divider_x

			ax = fig.add_axes([x0, y0, (1-eps)/divider_x, (1-eps)/divider_y])
			ax.set_yscale('log')
			ax.tick_params(left=True, bottom=True, right=False, top=False, labelsize='small')

			ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
			ticks_str = [str(x) for x in ticks]

			#ax.set_ylim(0,1)
			#ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])

			if snap != 12:

				ax.yaxis.set_major_formatter(NullFormatter())
				ax.yaxis.set_minor_formatter(NullFormatter())

			ax.set_xticks(ticks)

			if ion != 'hneutralssh':
				ax.set_xticklabels([])
			if ion == 'hneutralssh':			
				ax.set_xticklabels(ticks_str)

			
			if ion == 'c4':
				ax.set_ylim(1e11/8,1e16/8)
			if ion == 'si4':
				ax.set_ylim(1e8/8,1e15/8)
			if ion == 'hneutralssh':
				ax.set_ylim(1e13/8,1e21/8)

			for j in range(len(masses)):
				
				mlow, mass_label, color = masses[j]

				if sim == 'L0025N0752RECALIBRATED':
					sim_str = 'L0025N0752'
				
				gals_str = '/data1/cook/14AprilGalaxyLists/gals_%s_%s_snap=%i_mlow=%s_sfr=yes.txt'%(sim_str, recal_ref, snap, mlow)

				gals = glob.glob(gals_str)				
				gals = np.loadtxt(gals[0]) #mitigates n = ? issue

				L_x = box_size #box length
				num_slices = 1
				slice_thickness = L_x/num_slices

				zList = [(i/num_slices)*L_x for i in range(num_slices+1)]
				zcenList = []

				for i in range(len(slices)): #len slices != 4 because there are multiple ions to consider
					
					sl = slices[i]
					split_sl = sl.split('_')

					#picks out center z value of each slice
					for piece in split_sl:
						if 'zcen' in piece:
							piece = piece.replace('zcen','')
							if piece not in zcenList:
								zcenList.append(piece)

				zcenList = np.sort(map(float,zcenList))			

				Inputs = []
				counter = 0

				for sl in slices:

					slice_array = np.load(sl)['arr_0']
					split_sl = sl.split('_')
					
					#picks out center z value of each slice
					for piece in split_sl:
						if 'zcen' in piece:
							zSlicecen = float(piece.replace('zcen',''))

					for gal in gals:

						ID, ap_Mass_Star, sub_Mass_Star, ap_SFR, fof_gcom_x, fof_gcom_y, fof_gcom_z, fof_critR200, sub_hmr_Star, fof_snap = gal
						
						ap_M, sub_M, ap_SFR = ap_Mass_Star, sub_Mass_Star, ap_SFR
						x_halo, y_halo, z_halo = fof_gcom_x, fof_gcom_y, fof_gcom_z #in cMpc, from EAGLE database
						r_halo = fof_critR200 #in pkpc, from EAGLE database
						snap = int(fof_snap)

						if snap == 12:
							redshift = 3.02
						if snap == 14:
							redshift = 2.24
						if snap == 19:
							redshift = 1.
						if snap == 28:
							redshift = 0.

						r_halo = r_halo/(1000.)*(1+redshift) #converts from pkpc to cMpc

						'''
						makes sure to catch galaxies only in that slice and not too close to the boundary
						'''

						if abs(fof_gcom_z - zSlicecen) < (slice_thickness/2 - r_halo):

							x_pix, y_pix = int(math.floor((x_halo/L_x)*npix_x)), int(math.floor((y_halo/L_x)*npix_x))
							r_pix = int(math.ceil((r_halo/L_x)*npix_x))

							if r_pix %2 == 0:
								r_pix += 1

							x_min = int(x_pix - int(math.floor(r_pix/2)))
							x_max = int(x_pix + int(math.floor(r_pix/2)))
							y_min = int(y_pix - int(math.floor(r_pix/2)))
							y_max = int(y_pix + int(math.floor(r_pix/2)))

							#print 'xmin, xmax, ymin, ymax are', x_min, x_max, y_min, y_max

							Input = slice_array[x_min:x_max, y_min:y_max]	
							I_x, I_y = Input.shape

							if I_x == I_y and I_x > 0:
								Inputs.append(Input)

								'''
								if counter < 3:

									fig_sh, ax_sh = plt.subplots()
									cax = ax_sh.imshow(10**Input, norm=LogNorm())#, vmin=vvmin, vmax=vvmax)		
									cbar = fig_sh.colorbar(cax)

									savefig_str = 'projection_ID=%i_ion=%s_snap=%s_%s.pdf'%(ID,ion,snap,sim)

									plt.savefig(savefig_str)
									plt.close()

									counter += 1
								'''	

				if ion == 'hneutralssh' or ion == 'h1':
					
					covfracvalue = 1e18#np.percentile(np.asarray(10**thinnest_slice_array), 90)
					cfv_label = '10^{18}'

				if ion == 'si4':
					
					covfracvalue = 1e13#np.percentile(np.asarray(10**thinnest_slice_array), 50)
					cfv_label = '10^{13}'

				if ion == 'c4':
					
					covfracvalue = 1e14#np.percentile(np.asarray(10**thinnest_slice_array), 50)
					cfv_label = '10^{14}'


				bins = 20

				print ''
				print 'sim = %s, snap = %i, ion = %s, mlow = %s are'%(sim, snap, ion, mlow)				
				print 'CIPP inputs are', len(Inputs), bins, covfracvalue
				print ''
				Ngalaxies_table.append([sim,snap,mlow,len(Inputs)])

				Output = ColdensImpactParameterPlotter(Inputs, bins, covfracvalue)
				X, Y1, Y2, Y3, Y4 = Output[1][:,0], Output[1][:,2], Output[1][:,3], Output[1][:,4], Output[1][:,6]

				#gals_label = r'$N_{galaxies} = %i$'%(len(Inputs))
				ax.plot(X, Y1, color, linewidth=1, linestyle=':')#, label=r'25th percentile')
				ax.plot(X, Y2, color, linewidth=1, label=r'%s'%(mass_label))
			        ax.plot(X, Y3, color, linewidth=1, linestyle=':')#, label=r'25, 75th percentile')
				#plt.fill_between(X, Y3, Y1, color = color, interpolate=True, alpha = 0.2)

				#ax.plot(X, Y4, color, linewidth=1, label=r'%s'%(mass_label))

			if snap == 28:
				redshift_label = r'$z=0$'
			if snap == 19:
				redshift_label = r'$z=1$'

			if snap == 14:
				redshift_label = r'$z=2.24$'
			if snap == 12:
				redshift_label = r'$z=3.02$'

			#ax.annotate(redshift_label, xy = (0.1, 0.45), xycoords = 'axes fraction', fontsize = 6)

			#if color == 'g':
			
				#ax.annotate(r'50 $\pm$ 25th percentile', xy = (0.4, 0.75), xycoords = 'axes fraction', fontsize = 7)
				#ax.annotate(ion_label, xy = (0.85, 0.85), xycoords = 'axes fraction', fontsize = 8)
				#ax.annotate(r'$N_{th} = %.02e$'%covfracvalue, xy = (0.1, 0.25), xycoords = 'axes fraction', fontsize = 6)

			if snap == 19:

				ax.legend(bbox_to_anchor=(1.2, -0.65, 1., 0.35), framealpha=1, ncol=3, fontsize=10)	

			if snap == 12:		

				ax.set_ylabel(r'$\log(N$[cm$^{-2}$]$)$', fontsize=12)
				#ax.set_ylabel(r'$F(N_{th},r)$', fontsize=12)

			if snap == 19:		

				ax.set_xlabel(r'$r/R_{vir}$', fontsize=12)

			if snap == 12:				
				
				ax.set_xlabel(r'$z=3.02$', fontsize=12)
				ax.xaxis.set_label_position('top')

			if snap == 19:
				
				ax.set_xlabel(r'$z=1$', fontsize=12)
				ax.xaxis.set_label_position('top')

			if snap == 28:
				
				ax.set_xlabel(r'$z=0$', fontsize=12)
				ax.xaxis.set_label_position('top')

			if snap == 28:

				if color == 'g':

					ax.annotate(ion_label, xy = (0.5, 0.75), xycoords = 'axes fraction', fontsize = 10)
					#ax.annotate(r'$N_{th} = %s$'%cfv_label, xy = (0.5, 0.6), xycoords = 'axes fraction', fontsize = 10)
					#ax.set_ylabel(r'%s, $N_{th} = %.02e$'%(ion_label,covfracvalue), fontsize=6)#, rotation=270, labelpad=2.)
					#ax.yaxis.set_label_position('right')

		#plt.savefig('CoveringFractionPlots_%s_sfr=yes.pdf'%(sim))
		plt.savefig('ColumnDensityPlots_%s_8Slices_ion=%s_sfr=yes.pdf'%(sim, ion), bbox_inches='tight')

'''
with open('Ngalaxies_table.txt', 'w+') as f:
	for item in Ngalaxies_table:
		f.write('%s %i %s %i\n'%(item[0], item[1], item[2], item[3]))
'''
