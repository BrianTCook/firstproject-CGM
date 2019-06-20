import numpy as np
import random
import glob
import os
import random

random.seed(61)

snap = 11
mlow = '1.00e+09'
option = 'central'

box_size = 25.

#gals_str = glob.glob('/home/cook/Desktop/CGM-Research/package_HsmlAndProject_pywrapper/python_wrapper/L0025N0752_apsize=30_galaxylists/gals_L0025N0752_RECALIBRATED_apsize=30_snap=%i_mlow=%s_sfr=yes_%s.txt'%(snap,mlow,option))[0]

gals_str = glob.glob('gals_L0025N0752_apsize=30_snap=11_mlow=3.16e+09_observation_comparison.txt')[0]

if os.stat(gals_str).st_size > 0:

	gals = np.loadtxt(gals_str)

	xvals = gals[:,4]
	yvals = gals[:,5]

	rvals = gals[:,7]
	rvals = [r/1000./(1+3.53) for r in rvals]

        rmin, rmax = 20/1000./(1+3.53), 320/1000./(1+3.53)

	random_vals_r = [ rmin + (rmax-rmin)*random.random() for i in range(len(rvals)) ]
	random_vals_theta = [ random.random() for i in range(len(rvals)) ]

	x_perts = [ random_vals_r[i] * np.cos(2*np.pi*random_vals_theta[i]) for i in range(len(rvals)) ]
	y_perts = [ random_vals_r[i] * np.sin(2*np.pi*random_vals_theta[i]) for i in range(len(rvals)) ]

	points = [ [(xvals[i]+x_perts[i])/box_size, (yvals[i]+y_perts[i])/box_size, 0] for i in range(len(xvals)) ]

with open('los_L0025N0752_observation_comparison.txt', 'w+') as f:
	f.write('%i \n'%len(points))
	for point in points:
		f.write('%.04f %.04f %i\n'%(tuple(point)))

with open('gals_for_los_L0025N0752_observation_comparison.txt', 'w+') as f:
	#f.write('Galaxy_ID , sub_Mass_Total, sub_Mass_Star, sub_SFR, sub_gcom_x, sub_gcom_y, sub_gcom_z, fof_critR200, sub_sgn, fof_snap\n')
	for gal in gals:
		f.write('%i %.04f %.04f %.04f %.04f %.04f %.04f %.04f %i %i\n'%(tuple(gal)))

