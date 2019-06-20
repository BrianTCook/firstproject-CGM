from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.cosmology import Planck13 as cosmo #same cosmology as EAGLE

mu = 1.75 #primordial
solarmass = 1.9891e30 #in kilograms
G, mp, kB = 6.67e-11, 1.672e-27, 1.38e-23 #constants in SI units

converts_H_to_SI = 2.2e-18/67
cofactor = 1/5. * G * mu * mp / kB #1/5 gravitational constant * mu * proton mass / Boltzmann constant

def rcrit200(M, z):

	H = cosmo.H(z).value*converts_H_to_SI

	return ((G*M)/(200*H**2))**(1/3.)

def Tvir(M, z):

	#in SI units
	return cofactor*mu*(M/rcrit200(M,z))

#H0 = 67 km/s/Mpc = 2.2e-18 s^{-1}

z_samples, M_samples = 50, 50

zvals = np.linspace(0, 4., z_samples)
totalmasses_SI = np.linspace(1e8*solarmass, 1e10*solarmass, M_samples)

virialtemps_array = np.asarray([Tvir(M,z) for M in totalmasses_SI for z in zvals])
virialtemps_array = np.reshape(virialtemps_array, (-1, M_samples))

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('viridis')

xmin, xmax, ymin, ymax = min(zvals), max(zvals), np.log10(min(totalmasses_SI)/solarmass), np.log10(max(totalmasses_SI)/solarmass)
extent = [xmin, xmax, ymin, ymax]

im = ax.imshow(virialtemps_array, origin='lower', extent = extent, \
                aspect='auto', norm = LogNorm(), cmap = cmap)
ion_levels = [7.94e4, 1.26e5, 3.16e6, 6.31e6]
cont = ax.contour(virialtemps_array, levels=ion_levels, extent=extent, colors='red', alpha=1.0)

fmt = {}
strs = ['Si IV', 'C IV', 'O VI', 'Ne VIII']
for l, s in zip(cont.levels, strs):
    fmt[l] = s

ax.clabel(cont, cont.levels, inline=True, inline_spacing=1, \
          fmt=fmt, fontsize=10)
ax.set_xlabel(r'Redshift ($z$)', fontsize = 18)
ax.set_ylabel(r'$\log_{10}($Total Mass [$M_{\odot}$])', fontsize = 18)
ax.set_title(r'$T_{vir}$ [K]', fontsize = 24)

fig.colorbar(im)

#plt.tight_layout()
plt.savefig('VirialTemperature_allgalaxyclasses.pdf')
