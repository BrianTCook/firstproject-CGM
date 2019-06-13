import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import time

#random.seed(31)

f = h5py.File('/net/quasar/data2/cook/temp/specwizard_outputs/L0025N0752_lowmass_starforming_highz/spec.000.snap_011_z003p528.0.hdf5', 'r')
gals = np.loadtxt('los/gals_for_los_L0025N0752_lowmass_starforming_highz_central.txt')

N_total = len(gals)

spectrum_strs = ['Spectrum' + str(i) for i in range(N_total)]

'''
gets all spectra
'''

c4_od_all = [ f[which_spect]['c4']['OpticalDepth'] for which_spect in spectrum_strs ]
h1_od_all = [ f[which_spect]['h1']['OpticalDepth'] for which_spect in spectrum_strs ]
ne8_od_all = [ f[which_spect]['ne8']['OpticalDepth'] for which_spect in spectrum_strs ]
o6_od_all = [ f[which_spect]['o6']['OpticalDepth'] for which_spect in spectrum_strs ]
si4_od_all = [ f[which_spect]['si4']['OpticalDepth'] for which_spect in spectrum_strs ]

c4_f_all = [ f[which_spect]['c4']['Flux'] for which_spect in spectrum_strs ]
h1_f_all = [ f[which_spect]['h1']['Flux'] for which_spect in spectrum_strs ]
ne8_f_all = [ f[which_spect]['ne8']['Flux'] for which_spect in spectrum_strs ]
o6_f_all = [ f[which_spect]['o6']['Flux'] for which_spect in spectrum_strs ]
si4_f_all = [ f[which_spect]['si4']['Flux'] for which_spect in spectrum_strs ]

c4_peculiar_all = [ f[which_spect]['c4']['RealSpaceNionWeighted']['LOSPeculiarVelocity_KMpS'] for which_spect in spectrum_strs ]
h1_peculiar_all = [ f[which_spect]['h1']['RealSpaceNionWeighted']['LOSPeculiarVelocity_KMpS'] for which_spect in spectrum_strs ]
ne8_peculiar_all = [ f[which_spect]['ne8']['RealSpaceNionWeighted']['LOSPeculiarVelocity_KMpS'] for which_spect in spectrum_strs ]
o6_peculiar_all = [ f[which_spect]['o6']['RealSpaceNionWeighted']['LOSPeculiarVelocity_KMpS'] for which_spect in spectrum_strs ]
si4_peculiar_all = [ f[which_spect]['si4']['RealSpaceNionWeighted']['LOSPeculiarVelocity_KMpS'] for which_spect in spectrum_strs ]

peculiar_all = [ f[which_spect]['RealSpaceMassWeighted']['LOSPeculiarVelocity_KMpS'] for which_spect in spectrum_strs ]

'''
N random spectra

N = 5
indices = random.sample(range(len(spectrum_strs)), N)

c4_od_random = [ c4_od_all[index] for index in indices ]
h1_od_random = [ h1_od_all[index] for index in indices ]
ne8_od_random = [ ne8_od_all[index] for index in indices ]
o6_od_random = [ o6_od_all[index] for index in indices ]
si4_od_random = [ si4_od_all[index] for index in indices ]

c4_f_random = [ c4_f_all[index] for index in indices ]
h1_f_random = [ h1_f_all[index] for index in indices ]
ne8_f_random = [ ne8_f_all[index] for index in indices ]
o6_f_random = [ o6_f_all[index] for index in indices ]
si4_f_random = [ si4_f_all[index] for index in indices ]

c4_p_random = [ c4_peculiar_all[index] for index in indices ]
h1_p_random = [ h1_peculiar_all[index] for index in indices ]
ne8_p_random = [ ne8_peculiar_all[index] for index in indices ]
o6_p_random = [ o6_peculiar_all[index] for index in indices ]
si4_p_random = [ si4_peculiar_all[index] for index in indices ]

p_random = [ peculiar_all[index] for index in indices ]

gals = np.loadtxt('los/gals_for_los_L0025N0752_lowmass_starforming_highz_central.txt')
gals_random = [ gals[index] for index in indices ]
'''

xvals_original = f['VHubble_KMpS']
mx = max(xvals_original)
xvals_original_median = np.median(xvals_original)
xvals = [ x - xvals_original_median for x in xvals_original ]

box_size = 25.

'''
rolling_indices = np.zeros((N_total))

for i in range(N_total):
	
	if i%100 == 0:
		print 'i is', i

	gal = gals[i]
	gal_z = gal[6]*(mx/box_size)

	#p = p_random[i]
	p = peculiar_all[i]

	xvals_minus_galz_abs = [ abs(x - gal_z) for x in xvals_original ]
	xvals_abs = [ abs(x) for x in xvals ]
	gal_index = xvals_minus_galz_abs.index( min(xvals_minus_galz_abs) )

	rolling_indices[i] = int( xvals_abs.index(min(xvals_abs)) - gal_index )

np.savetxt('rolling_indices.txt', rolling_indices)
'''

rolling_indices = np.loadtxt('rolling_indices.txt')
rolling_indices = [ int(ri) for ri in rolling_indices ]

c4_od_all = [ np.roll(c4_od_all[i], rolling_indices[i]) for i in range(N_total) ]
h1_od_all = [ np.roll(h1_od_all[i], rolling_indices[i]) for i in range(N_total) ]
ne8_od_all = [ np.roll(ne8_od_all[i], rolling_indices[i]) for i in range(N_total) ]
o6_od_all = [ np.roll(o6_od_all[i], rolling_indices[i]) for i in range(N_total) ]
si4_od_all = [ np.roll(si4_od_all[i], rolling_indices[i]) for i in range(N_total) ]

c4_f_all = [ np.roll(c4_f_all[i], rolling_indices[i]) for i in range(N_total) ]
h1_f_all = [ np.roll(h1_f_all[i], rolling_indices[i]) for i in range(N_total) ]
ne8_f_all = [ np.roll(ne8_f_all[i], rolling_indices[i]) for i in range(N_total) ]
o6_f_all = [ np.roll(o6_f_all[i], rolling_indices[i]) for i in range(N_total) ]
si4_f_all = [ np.roll(si4_f_all[i], rolling_indices[i]) for i in range(N_total) ]


optdeps = [c4_od_all, h1_od_all, ne8_od_all, o6_od_all, si4_od_all]
fluxes = [c4_f_all, h1_f_all, ne8_f_all, o6_f_all, si4_f_all]
ion_labels = ['C IV', 'H I', 'Ne VIII', 'O VI', 'Si IV']
ions = ['c4', 'h1', 'ne8', 'o6', 'si4']

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

for j in range(len(ion_labels)):

	vals, ion_label = optdeps[j], ion_labels[j]

	fig, ax = plt.subplots()
	
	yvals1 = np.percentile( vals, 25, axis = 0)
	yvals2 = np.percentile( vals, 50, axis = 0)
	yvals3 = np.percentile( vals, 75, axis = 0)

	ax.semilogy(xvals, yvals1, 'k', linestyle = '--', linewidth=0.5)
	ax.semilogy(xvals, yvals2, 'k', linestyle = '--', linewidth=1)
	ax.semilogy(xvals, yvals3, 'k', linestyle = '--', linewidth=0.5)

	ax.annotate(r'$z = 3.53$', xy = (0.8, 0.7), xycoords = 'axes fraction', fontsize = 10)
	ax.annotate(r'$N_{galaxies} = %i$'%N_total, xy = (0.8, 0.75), xycoords = 'axes fraction', fontsize = 10)
	ax.annotate(ion_label, xy = (0.8, 0.8), xycoords = 'axes fraction', fontsize = 10)

	ax.set_ylabel('Optical Depth', fontsize=12)
	ax.set_ylim(1e-6, 2e0)

	ax.set_xlabel('Hubble velocity (galaxy frame, km/s)', fontsize=12)
	ax.set_xlim(min(xvals), max(xvals))

	xts = [ -1000, -750, -500, -250, 0, 250, 500, 750, 1000 ]
	ax.set_xticks(xts)
	ax.set_xticklabels( [ str(abs(x)) for x in xts ] )

	plt.tight_layout()
	plt.savefig('stacked_%s_od.pdf'%(ions[j]))
	plt.close()

for j in range(len(ion_labels)):

	vals, ion_label = fluxes[j], ion_labels[j]

	fig, ax = plt.subplots()

	yvals1 = np.percentile( vals, 25, axis = 0)
	yvals2 = np.percentile( vals, 50, axis = 0)
	yvals3 = np.percentile( vals, 75, axis = 0)

	ax.plot(xvals, yvals1, 'k', linestyle = '--', linewidth=0.5)
	ax.plot(xvals, yvals2, 'k', linestyle = '--', linewidth=1)
	ax.plot(xvals, yvals3, 'k', linestyle = '--', linewidth=0.5)

	ax.annotate(r'$z = 3.53$', xy = (0.8, 0.8), xycoords = 'axes fraction', fontsize = 10)
	ax.annotate(r'$N_{galaxies} = %i$'%N_total, xy = (0.8, 0.85), xycoords = 'axes fraction', fontsize = 10)
	ax.annotate(ion_label, xy = (0.8, 0.9), xycoords = 'axes fraction', fontsize = 10)

	ax.set_ylabel('Flux', fontsize=12)
	ax.set_ylim(0.0, 1.2)

	ax.set_xlabel('Hubble velocity (galaxy frame, km/s)', fontsize=12)
	ax.set_xlim(min(xvals), max(xvals))

	xts = [ -1000, -750, -500, -250, 0, 250, 500, 750, 1000 ]
	ax.set_xticks(xts)
	ax.set_xticklabels( [ str(abs(x)) for x in xts ] )

	plt.tight_layout()
	plt.savefig('stacked_%s_f.pdf'%(ions[j]))
	plt.close()

'''
for single halo plots

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

for i in range(N):

	gal = gals_random[i]
	gal_z = gal[6]*(mx/box_size)

	p = p_random[i]

	xvals_minus_galz = [ x - gal_z for x in xvals_original ]
	xvals_minus_galz_abs = [ abs(x - gal_z) for x in xvals_original ]
	gal_index = xvals_minus_galz_abs.index( min(xvals_minus_galz_abs) )

	gal_loc = gal_z + p[gal_index]

	ion_samples = [[c4_f_random, 'C IV', 'C3'], [h1_f_random, 'H I', 'C2'], [ne8_f_random, 'Ne VIII', 'C1'], [o6_f_random, 'O VI', 'C9'], [si4_f_random, 'Si IV', 'C7']]

	fig, ax1 = plt.subplots()
	ax2 = ax1.twiny()

	ax1.set_ylabel('Flux', fontsize=12)
	ylabel = 'fluxes'

	for ion_sample, ion_label, ion_color in ion_samples:

		ax1.plot(xvals_original, ion_sample[i], linewidth=1, alpha = 0.8, color = ion_color, label=ion_label)

	ax1.axvline(x=gal_loc, linewidth=0.5, color='black' )
	ax1.set_ylim(0.0, 1.1)

	ax1.set_xlabel('Hubble velocity (observer frame, km/s)', fontsize=12)
	ax1.set_xlim(min(xvals_original), max(xvals_original))
	ax2.set_xlim(0, box_size)

	ax2.set_xlabel('Line-of-sight distance (cMpc)', fontsize=12)

	xts_2 = [0, 5, 10, 15, 20, 25]

	ax2.set_xticks(xts_2)
	ax2.set_xticklabels([str(element) for element in xts_2])
	
	ax1.legend(loc='lower left')
	plt.title('Galaxy ID: %i'%gal[0], fontsize=16, y=1.12)
	plt.tight_layout()
	plt.savefig('%i_flux.pdf'%(gal[0]))
	plt.close()

for i in range(N):

	gal = gals_random[i]
	gal_z = gal[6]*(mx/box_size)

	p = p_random[i]

	xvals_minus_galz = [ x - gal_z for x in xvals_original ]
	xvals_minus_galz_abs = [ abs(x - gal_z) for x in xvals_original ]
	gal_index = xvals_minus_galz_abs.index( min(xvals_minus_galz_abs) )

	gal_loc = gal_z + p[gal_index]

	ion_samples = [[c4_f_random, 'C IV', 'C3'], [h1_f_random, 'H I', 'C2'], [ne8_f_random, 'Ne VIII', 'C1'], [o6_f_random, 'O VI', 'C9'], [si4_f_random, 'Si IV', 'C7']]

	fig, ax1 = plt.subplots()
	ax2 = ax1.twiny()

	ax1.set_ylabel('Optical Depth', fontsize=12)
	ylabel = 'fluxes'

	for ion_sample, ion_label, ion_color in ion_samples:

		ax1.semilogy(xvals_original, ion_sample[i], linewidth=1, alpha = 0.8, color = ion_color, label=ion_label)

	ax1.axvline(x=gal_loc, linewidth=0.5, color='black' )
	ax1.set_ylim(1e-5, 2e0)

	ax1.set_xlabel('Hubble velocity (observer frame, km/s)', fontsize=12)
	ax1.set_xlim(min(xvals_original), max(xvals_original))
	ax2.set_xlim(0, box_size)

	ax2.set_xlabel('Line-of-sight distance (cMpc)', fontsize=12)

	xts_2 = [0, 5, 10, 15, 20, 25]

	ax2.set_xticks(xts_2)
	ax2.set_xticklabels([str(element) for element in xts_2])
	
	ax1.legend(loc='lower left')
	plt.title('Galaxy ID: %i'%gal[0], fontsize=16, y=1.12)
	plt.tight_layout()
	plt.savefig('%i_optdep.pdf'%(gal[0]))
	plt.close()
'''
