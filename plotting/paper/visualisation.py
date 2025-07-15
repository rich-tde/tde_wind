""" Plot density proj / Dendiss proj / bunch of slices."""
#%%
abspath = '/Users/paolamartire/shocks'
import sys
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import healpy as hp
import astropy.coordinates as coord
from astropy import units as u
import colorcet
import src.orbits as orb
import Utilities.prelude as prel
from Utilities.sections import make_slices
from Utilities.time_extractor import days_since_distruption
from Utilities.operators import from_cylindric, to_spherical_components, make_tree

##
# PARAMETERS
##
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = ''
choosen_snaps = np.array([100, 237, 348])
save = True

DeltaE = orb.energy_mb(Rstar, mstar, Mbh, G=1) # specific energy of the mb debris 
DeltaE_cgs = DeltaE * prel.en_converter/prel.Msol_cgs
t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
t_fall_hour = t_fall * 24

Rt = Rstar * (Mbh/mstar)**(1/3)
Rp = beta * Rt
R0 = 0.6 * Rt
Rs = 2*prel.G*Mbh/prel.csol_cgs**2
Rg = Rs/2
apo = orb.apocentre(Rstar, mstar, Mbh, beta)
a_mb = orb.semimajor_axis(Rstar, mstar, Mbh, G=1)
e_mb = orb.e_mb(Rstar, mstar, Mbh, beta)
if check in ['NewAMR', 'HiResNewAMR', 'LowResNewAMR']:
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}' 
else:
    folder = f'opacity_tests/R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
solarR_to_au = 215

#%% make cfr
radii_grid = [Rt/apo, a_mb/apo, 1] #*apo 
styles = ['dashed', 'solid', 'solid']
xcfr_grid, ycfr_grid, cfr_grid = [], [], []
for i, radius_grid in enumerate(radii_grid):
    xcr, ycr, cr = orb.make_cfr(radius_grid)
    xcfr_grid.append(xcr)
    ycfr_grid.append(ycr)
    cfr_grid.append(cr)

theta_arr = np.arange(0, 2*np.pi, 0.01)
r_arr_ell = orb.keplerian_orbit(theta_arr, a_mb, Rp, ecc=e_mb)
x_arr_ell, y_arr_ell = from_cylindric(theta_arr, r_arr_ell)
r_arr_par = orb.keplerian_orbit(theta_arr, a_mb, Rp, ecc=1)
x_arr_par, y_arr_par = from_cylindric(theta_arr, r_arr_par)
precession_angle = orb.precession_angle(Rstar, mstar, Mbh, beta, c = prel.csol_cgs, G=1)
# x_arr_rot, y_arr_rot = rotate_coordinate(x_arr_ell, y_arr_ell, precession_angle)

# HEALPIX
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz).T
x, y, z = observers_xyz[:, 0], observers_xyz[:, 1], observers_xyz[:, 2]
r = np.sqrt(x**2 + y**2 + z**2)   # Radius (should be 1 for unit vectors)
theta = np.arctan2(y, x)          # Azimuthal angle in radians
phi = np.arccos(z / r)            # Elevation angle in radians
longitude_moll = theta              
latitude_moll = np.pi / 2 - phi 
indecesorbital = np.concatenate(np.where(latitude_moll==0))
first_idx, last_idx = np.min(indecesorbital), np.max(indecesorbital)

#%% 3 Den proj
time = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDentime_proj.txt')
snaps = [int(i) for i in time[0]]
tfb = time[1]
fig = plt.figure(figsize=(10, 18))
gs = gridspec.GridSpec(3, 2, width_ratios=[1,0.03], height_ratios=[1,1,1], hspace=0.2, wspace = 0.03)
for i, snap in enumerate(choosen_snaps):
    # load the data
    idx = np.argmin(np.abs(snaps-snap))
    tfb_single = tfb[idx]
    x_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenxarray.npy')
    y_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenyarray.npy')
    flat_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenproj{snap}.npy')
    flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs # [g/cm2]
    
    ax = fig.add_subplot(gs[i, 0])  # First plot
    # NB: can't use imshow beacuse of the data are not on a regular linspaced grid
    img = ax.pcolormesh(x_radii/apo, y_radii/apo, flat_den_cgs.T, cmap = 'plasma',
                        norm = colors.LogNorm(vmin = 5e-2, vmax = 1e7), rasterized = True)
    
    if i == 0:
        # Create an inset axis in the top right corner with more distance from the border
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad = 2.5)

        # Define the zoom-in region in physical units
        x_min, x_max = -1, 0.2    # Y range in physical units
        y_min, y_max = -0.7, 0.7  # X range in physical units

        # Get mask for selected region
        x_mask = (x_radii/apo >= x_min) & (x_radii/apo <= x_max)
        y_mask = (y_radii/apo >= y_min) & (y_radii/apo <= y_max)

        # Apply the mask to get zoomed-in grid points
        x_zoom = x_radii[x_mask] / apo
        y_zoom = y_radii[y_mask] / apo
        flat_den_zoom = flat_den_cgs[np.ix_(x_mask, y_mask)]  # Zoomed-in density data

        # Use pcolormesh for the inset plot
        img_inset = ax_inset.pcolormesh(x_zoom, y_zoom, flat_den_zoom.T, cmap='plasma',
                                        norm=colors.LogNorm(vmin=5e-2, vmax=1e7), rasterized=True)

        # Remove labels but keep ticks
        # Inset plot ticks
        ax_inset.tick_params(axis='x', direction='in', length=7, width=1.5, colors='white', labelsize=10)  # X-axis ticks
        ax_inset.tick_params(axis='y', direction='in', length=7, width=1.5, colors='white', labelsize=10)  # Y-axis ticks
        ax_inset.yaxis.tick_right()  # Move ticks to the right
        # ax_inset.set_xticklabels([])  # Remove numbers
        # ax_inset.set_yticklabels([])  # Remove numbers

        # Set the aspect ratio to match the physical size of the zoom region
        aspect_ratio = (x_max - x_min) / (y_max - y_min)
        ax.set_aspect(aspect_ratio, adjustable='box')  # Main plot aspect ratio
        ax_inset.set_aspect(aspect_ratio, adjustable='box')  # Inset plot aspect ratio

        # Add a rectangle to indicate the zoomed-in region
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        # Inset rectangle should have the same physical size as the main plot rectangle
        inset_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                                linewidth=2, edgecolor='white', facecolor='none')
        ax_inset.add_patch(inset_rect)

    # Photosphere
    dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/_photo{snap}.txt')
    xph, yph, zph, volph= dataph[0], dataph[1], dataph[2], dataph[3]
    midph= np.abs(zph) < volph**(1/3)
    # xph_mid, yph_mid, zph_mid = make_slices([xph, yph, zph], midph)
    ax.plot(xph[indecesorbital]/apo, yph[indecesorbital]/apo, c = 'white', markersize = 5, marker = 'H', label = r'$R_{\rm ph}$')
    # just to connect the first and last 
    ax.plot([xph[first_idx]/apo, xph[last_idx]/apo], [yph[first_idx]/apo, yph[last_idx]/apo], c = 'white', markersize = 1, marker = 'H')
        
    if i == 0:
        ax.plot(x_arr_par/apo, y_arr_par/apo, c= 'white', linestyle = 'dashed', alpha = 0.7)
    else: 
        ax.plot(x_arr_ell/apo, y_arr_ell/apo, c= 'white', linestyle = 'dashed', alpha = 0.7)
    for j in range(len(radii_grid)):
        ax.contour(xcfr_grid[j], ycfr_grid[j], cfr_grid[j], levels=[0], colors='white', alpha = 0.5)
    
    ax.text(-5.5, 1.35, f't = {np.round(tfb[idx],2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 16)
    ax.set_ylabel(r'$Y [R_{\rm a}]$')#, fontsize = 20)
    ax.tick_params(axis='x', which='major', width = .7, length = 7, color = 'white')
    ax.tick_params(axis='y', which='major', width = .7, length = 7, color = 'white')
    ax.set_xlim(-6, 2.5)
    ax.set_ylim(-3, 2)
    
    if i == 2:
        ax.text(Rt/apo + 0.05, 0, r'$R_{\rm t}$', color = 'white', fontsize = 14)
        ax.text(a_mb/apo + 0.05, 0, r'$a_{\rm mb}$', color = 'white', fontsize =14)
        ax.text(1 + 0.05, 0, r'$R_{\rm a}$', color = 'white', fontsize =14)

# Create a colorbar that spans the first two subplots
cbar_ax = fig.add_subplot(gs[0:3, 1])  # Colorbar subplot below the first two
cb = fig.colorbar(img, cax=cbar_ax, orientation='vertical')
cb.ax.tick_params(which='major',length = 5)
cb.ax.tick_params(which='minor',length = 3)
cb.set_label(r'Column density [g/cm$^2$]', fontsize = 20)
ax.set_xlabel(r'$X [R_{\rm a}]$')#, fontsize = 20)
plt.tight_layout()

if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/3denprojph.png', bbox_inches='tight')
plt.show() 

#%% ORBITAL ENERGY AND a
time_slice = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/slices/z/z0_time.txt')
snaps_slice, tfb_slice = time_slice[0], time_slice[1]
idx = np.argmin(np.abs(snaps_slice-snap))
tfb_single = tfb_slice[idx]
zslice = np.load(f'/Users/paolamartire/shocks/data/{folder}/slices/z/z0slice_{snap}.npy')
x_mid, y_mid, z_mid, dim_mid, den_mid, temp_mid, ie_den_mid, orb_en_den_mid, Rad_den_mid =\
    zslice[0], zslice[1], zslice[2], zslice[3], zslice[4], zslice[5], zslice[6], zslice[7], zslice[8]
orb_en_spec = orb_en_den_mid / den_mid
orb_en_spec_cgs = orb_en_spec * prel.en_converter / prel.Msol_cgs
orb_en_mid = orb_en_den_mid * dim_mid**3
orb_en_mid_cgs = orb_en_mid * prel.en_converter
a_mid = prel.G * Mbh / (2*np.abs(orb_en_spec))
vminoe_spec_cgs = 4e15 #4e40
vmaxoe_spec_cgs = 1e18 #9e42
vminoe_spec = vminoe_spec_cgs / (prel.en_converter / prel.Msol_cgs)
vmaxoe_spec = vmaxoe_spec_cgs / (prel.en_converter / prel.Msol_cgs)
vmina = prel.G * Mbh / (2*vmaxoe_spec) # vmaxoe is energy NOT specific
vmaxa = prel.G * Mbh / (2*vminoe_spec)
vminoe = orb_en_mid_cgs[np.argmin((np.abs(orb_en_spec_cgs-vminoe_spec_cgs)))] 
vmaxoe = orb_en_mid_cgs[np.argmin((np.abs(orb_en_spec_cgs-vmaxoe_spec_cgs)))]

#%%
img, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
img = ax1.scatter(x_mid/apo, y_mid/apo, c = np.abs(orb_en_spec_cgs)/DeltaE_cgs, cmap = 'spring', s = 2,
                    norm = colors.LogNorm(vmin = vminoe_spec_cgs/DeltaE_cgs, vmax = vmaxoe_spec_cgs/DeltaE_cgs))
cb = plt.colorbar(img)
cb.set_label(r'$|$ Specific orbital energy$| [\Delta E$]', fontsize = 22)
ax1.set_ylabel(r'$Y/R_{\rm a}$',)# fontsize = 22)

img = ax2.scatter(x_mid/apo, y_mid/apo, c = a_mid/a_mb, cmap = 'winter', s = 2,
                    norm = colors.LogNorm(vmin = vmina/a_mb, vmax = vmaxa/a_mb))
cb = plt.colorbar(img)
cb.set_label(r'Semi-major axis $[a_{\rm mb}$]', fontsize = 22)
ax2.set_ylabel(r'$Y/R_{\rm a}$')#, fontsize = 22)
ax2.set_xlabel(r'$X/R_{\rm a}$')#, fontsize = 22)
for ax in [ax1, ax2]:
    ax.set_xlim(-1.2, 40/apo)
    ax.set_ylim(-0.4, 0.4)
    for j in range(len(radii_grid)):
        ax.contour(xcfr_grid[j], ycfr_grid[j], cfr_grid[j], [0], colors = 'k', linestyle = styles[j], alpha = 0.5)

plt.suptitle( f't = {np.round(tfb_slice[idx],2)}' + r' $t_{\rm fb}$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/OE_a{snap}.png')
plt.show()

# %%
from plotting.paper.IHopeIsTheLast import split_data_red
talk = True
overview = False

if talk:
    from Utilities.operators import sort_list
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps, Lum, tfb = split_data_red('NewAMR')
    # dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_NewAMRcutDen.txt')
    # tfbdiss, LDiss = dataDiss[0], dataDiss[2] * prel.en_converter/prel.tsol_cgs 
    for i, snap in enumerate(snaps):
        if snap not in np.arange(96,97):
            continue
        print(snap)
        x_denproj = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denxarray.npy')
        y_denproj = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denyarray.npy')
        flat_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denproj{snap}.npy')
        flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs # [g/cm2]
        flat_diss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Dissproj{snap}.npy')
        flat_diss_cgs = flat_diss * prel.en_den_converter * prel.Rsol_cgs / prel.tsol_cgs# [erg/s/cm2]
        # make =1 the nan values so they disappera with logcolor
        flat_diss_cgs_plot = flat_diss_cgs
        flat_diss_cgs_plot[np.isnan(flat_diss_cgs_plot)] = 1
        flat_diss_cgs_plot[flat_diss_cgs_plot == 0] = 1

        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (40,15), gridspec_kw={'width_ratios': [1, 1, 0.8]})
        ax3.scatter(tfb[:i+1], Lum[:i+1], c = 'k', s = 18, label = r'L$_{\rm FLD}$')
        # ax3.plot(tfbdiss[:i+1], LDiss[:i+1], ls = '--', c = 'k', label = r'L$_{\rm diss}$')
        ax3.set_xlabel(r't $[t_{\rm fb}]$')
        ax3.set_ylabel(r'L [erg/s]')
        ax3.set_yscale('log')
        ax3.set_xlim(0.06, 1.75)
        ax3.set_ylim(1e37, 8e42)
        ax3.legend(fontsize = 25)
    
        img = ax1.pcolormesh(x_denproj/apo, y_denproj/apo, flat_den_cgs.T, cmap = 'plasma', \
                          norm = colors.LogNorm(vmin = 1e2, vmax = 5e7))
        cbar = plt.colorbar(img, orientation = 'horizontal', pad = 0.15)
        cbar.set_label(r'Column density [g cm$^{-2}$]')
        ax1.set_ylabel(r'Y [$R_{\rm a}$]')
        ax1.text(-1.15, 0.4, f't = {np.round(tfb[i],2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 28)

        img = ax2.pcolormesh(x_denproj/apo, y_denproj/apo, flat_diss_cgs.T, \
                          cmap = 'viridis', norm = colors.LogNorm(vmin = 1e14, vmax = 1e19))
        cbar = plt.colorbar(img, orientation = 'horizontal', pad = 0.15)
        cbar.set_label(r'Dissipation energy column density [erg s$^{-1}$cm$^{-2}]$')
        ax2.text(-1.15, 0.4, f't = {np.round(tfb[i]*t_fall_hour,1)}' + r' hours', color = 'white', fontsize = 28)
        for ax in [ax1, ax2]:
            ax.contour(xcfr_grid[0], ycfr_grid[0], cfr_grid[0], levels=[0], colors='white')
            ax.scatter(0,0,c= 'k', marker = 'x', s=80)
            ax.set_xlim(-1.2,0.1)
            ax.set_ylim(-0.5,0.5)
            ax.set_xlabel(r'X [$R_{\rm a}$]')
        
        plt.tight_layout()
        # plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/projection/denproj_diss{snap}.png')
        # plt.close()

if overview:
    t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
    t_fall_cgs = t_fall * 24 * 3600
    time = np.loadtxt(f'{abspath}/data/{folder}/slices/z/z0_time.txt')
    snaps, tfb_all = time[0], time[1]
    snaps = np.array([int(snap) for snap in snaps])
    snap = 164
    for snap in [122, 164]:
        tfb = tfb_all[np.argmin(np.abs(snaps-snap))]
        data_mid = np.load(f'{abspath}/data/{folder}/slices/z/z0slice_{snap}.npy')
        x_cut, y_cut, z_cut, dim_cut, den_cut, temp_cut, ie_den_cut, orb_en_den_cut, Rad_den_cut, VX_cut, VY_cut, VZ_cut =\
            data_mid[0], data_mid[1], data_mid[2], data_mid[3], data_mid[4], data_mid[5], data_mid[6], data_mid[7], data_mid[8], data_mid[9], data_mid[10], data_mid[11]#, data_mid[12]
        orb_en_spec_cut = orb_en_den_cut / den_cut
        ie_spec_cut = ie_den_cut / den_cut
        fig, ax = plt.subplots(2,3, figsize = (30,10))
        img = ax[0][0].scatter(x_cut/apo, y_cut/apo, c = den_cut, cmap = 'jet', s= 5, \
                    norm = colors.LogNorm(vmin = 2e-9, vmax = 5e-6))
        cb = plt.colorbar(img)
        cb.set_label(r'Density [$M_\odot/R_\odot^3$]', fontsize = 14)

        img = ax[0][1].scatter(x_cut/apo, y_cut/apo, c = temp_cut, cmap = 'jet', s= 5, \
                    norm = colors.LogNorm(vmin = 1e4, vmax = 8e6))
        cb = plt.colorbar(img)
        cb.set_label(r'T [K]', fontsize = 14)

        # img = ax[0][2].scatter(x_cut/apo, y_cut/apo, c = np.abs(Diss_den_cut)*prel.en_den_converter, cmap = 'jet', s= 5, \
        #             norm = colors.LogNorm(vmin = 1e6, vmax = 2e14))
        # cb = plt.colorbar(img)
        # cb.set_label(r'$|$Dissipation energy density$|$ [erg/cm$^3$]', fontsize = 14)

        img = ax[1][0].scatter(x_cut/apo, y_cut/apo, c = np.abs(orb_en_spec_cut)*prel.en_converter/prel.Msol_cgs, cmap = 'jet', s= 5, \
                    norm = colors.LogNorm(vmin = 5e15, vmax = 4e17))
        cb = plt.colorbar(img)
        cb.set_label(r'Absolute specific orbital energy [erg/g]', fontsize = 14)

        img = ax[1][1].scatter(x_cut/apo, y_cut/apo, c = ie_spec_cut*prel.en_converter/prel.Msol_cgs, cmap = 'jet', s= 5, \
                    norm = colors.LogNorm(vmin = 7e12, vmax = 5e16))
        cb = plt.colorbar(img)
        cb.set_label(r'Specific IE [erg]', fontsize = 14)

        img = ax[1][2].scatter(x_cut/apo, y_cut/apo, c = Rad_den_cut*prel.en_den_converter, cmap = 'jet', s= 5, \
                    norm = colors.LogNorm(vmin = 1e3, vmax = 2e11))
        cb = plt.colorbar(img)
        cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 14)

        for i in range(2):
            for j in range(3):
                ax[i][j].set_xlabel(r'$X/R_{\rm a}$', fontsize = 20)
                ax[i][j].set_xlim(-.2, .1)#(-340,25)
                ax[i][j].set_ylim(-.1, .1)#(-70,70)
            ax[i][0].set_ylabel(r'$Y/R_{\rm a}$', fontsize = 20)

        ax[1][0].text(-0.18, 0.08, f't = {np.round(tfb, 2)}' + r'$t_{\rm fb}$', fontsize = 20)
        ax[0][0].text(-0.18, 0.08, f'snap {int(snap)}', fontsize = 16)
        plt.tight_layout()
        plt.savefig(f'{abspath}/Figs/{folder}/slices/BIGvisual{snap}.png')


    # %%
