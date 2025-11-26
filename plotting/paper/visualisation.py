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
import healpy as hp
from astropy import units as u
import src.orbits as orb
import Utilities.prelude as prel
from Utilities.sections import make_slices
from Utilities.operators import from_cylindric, sort_list
from plotting.paper.IHopeIsTheLast import split_data_red
import Utilities.prelude as prel

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
check = 'HiResNewAMR'
save = True

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}' 
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
a_mb = things['a_mb']
e_mb = things['ecc_mb']
t_fall = things['t_fb_days']
t_fall_hour = t_fall * 24
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs

# make cfr
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
# precession_angle = orb.precession_angle(Rstar, mstar, Mbh, beta, c = prel.csol_cgs, G=1)
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
fig = plt.figure(figsize=(10, 21))
gs = gridspec.GridSpec(3, 2, width_ratios=[1,0.03], height_ratios=[1,1,1], hspace=0.2, wspace = 0.03)
for i, snap in enumerate(snaps):
    # load the data
    tfb_single = tfb[i]
    x_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenxarray.npy')
    y_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenyarray.npy')
    flat_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenproj{snap}.npy')
    flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs # [g/cm2]
    dmin, dmax = 5e-2, 1e7
    ax = fig.add_subplot(gs[i, 0])  # First plot
    # NB: can't use imshow beacuse of the data are not on a regular linspaced grid
    img = ax.pcolormesh(x_radii/apo, y_radii/apo, flat_den_cgs.T, cmap = 'plasma',
                        norm = colors.LogNorm(vmin = dmin, vmax = dmax), rasterized = True)
    
    if i == 0:
        # Create an inset axis in the top right corner with more distance from the border
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad = 2.5)

        # Define the zoom-in region in physical units
        x_min, x_max = -1, 0.5    # Y range in physical units
        y_min, y_max = -0.5, 0.5  # X range in physical units

        # Get mask for selected region
        x_mask = (x_radii/apo >= x_min) & (x_radii/apo <= x_max)
        y_mask = (y_radii/apo >= y_min) & (y_radii/apo <= y_max)

        # Apply the mask to get zoomed-in grid points
        x_zoom = x_radii[x_mask] / apo
        y_zoom = y_radii[y_mask] / apo
        flat_den_zoom = flat_den_cgs[np.ix_(x_mask, y_mask)]  # Zoomed-in density data

        # Use pcolormesh for the inset plot
        img_inset = ax_inset.pcolormesh(x_zoom, y_zoom, flat_den_zoom.T, cmap = 'plasma',
                                        norm=colors.LogNorm(vmin=dmin, vmax=dmax), rasterized=True)

        # Remove labels but keep ticks
        # Inset plot ticks
        ax_inset.tick_params(axis='x', direction='in', length=7, width=1.5, colors='white', labelsize=10)  # X-axis ticks
        ax_inset.tick_params(axis='y', direction='in', length=7, width=1.5, colors='white', labelsize=10)  # Y-axis ticks
        ax_inset.yaxis.tick_right()  # Move ticks to the right
        # ax_inset.set_xticklabels([])  # Remove numbers
        # ax_inset.set_yticklabels([])  # Remove numbers

        # Set the aspect ratio to match the physical size of the zoom region
        aspect_ratio = (x_max - x_min) / (y_max - y_min)
        # ax.set_aspect(aspect_ratio, adjustable='box')  # Main plot aspect ratio
        # ax_inset.set_aspect(aspect_ratio, adjustable='box')  # Inset plot aspect ratio

        # Add a rectangle to indicate the zoomed-in region
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        # Inset rectangle should have the same physical size as the main plot rectangle
        inset_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                                linewidth=2, edgecolor='white', facecolor='none')
        ax_inset.add_patch(inset_rect)

    # Photosphere
    dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
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
    
    if i == 1:
        ax.text(-5.5, 2.35, f't = {np.round(tfb_single,1)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 16)
    else:
        ax.text(-5.5, 2.35, f't = {np.round(tfb_single,2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 16)
    ax.set_ylabel(r'$Y [r_{\rm a}]$')#, fontsize = 20)
    ax.tick_params(axis='x', which='major', width = .7, length = 7, color = 'white')
    ax.tick_params(axis='y', which='major', width = .7, length = 7, color = 'white')
    ax.set_xlim(-6, 2.5)
    ax.set_ylim(-3, 3)
    
    if i == 2:
        ax.text(Rt/apo + 0.02, 0.05, r'$r_{\rm t}$', color = 'white', fontsize = 14)
        ax.text(a_mb/apo + 0.04, 0.16, r'$a_{\rm mb}$', color = 'white', fontsize =14)
        ax.text(1 + 0.01, 0.3, r'$r_{\rm a}$', color = 'white', fontsize =14)

# Create a colorbar that spans the first two subplots
cbar_ax = fig.add_subplot(gs[0:3, 1])  # Colorbar subplot below the first two
cb = fig.colorbar(img, cax=cbar_ax, orientation='vertical')
cb.ax.tick_params(which='major',length = 5)
cb.ax.tick_params(which='minor',length = 3)
cb.set_label(r'Column density [g/cm$^2$]', fontsize = 20)
ax.set_xlabel(r'$X [r_{\rm a}]$')#, fontsize = 20)
plt.tight_layout()

if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/3denprojph.png', bbox_inches='tight')
plt.show() 


#%% with diss proj as well
dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3] * prel.en_converter/prel.tsol_cgs
time = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDentime_proj.txt')
snaps = [int(i) for i in time[0]]
tfb = time[1]
fig = plt.figure(figsize=(24, 25))
gs = gridspec.GridSpec(4, 2, width_ratios=[1,1], height_ratios=[1.05,1.05,1.05, 0.04], hspace=0.2, wspace = 0.2)
for i, snap in enumerate(snaps):
    # load the data
    tfb_single = tfb[i]
    x_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenxarray.npy')
    y_radii = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenyarray.npy')
    flat_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/bigDenproj{snap}.npy')
    flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs # [g/cm2]
    dmin, dmax = 4, 1e7
    x_radiiDiss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/nozzleDissxarray.npy')
    y_radiiDiss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/nozzleDissyarray.npy')
    flat_diss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/nozzleDissproj{snap}.npy')
    flat_diss_cgs = flat_diss * prel.en_converter / (prel.tsol_cgs * prel.Rsol_cgs**2) # [erg/s/cm2]
    Emin, Emax = 1e14, 1e19

    ax = fig.add_subplot(gs[i, 0]) # first
    axDiss = fig.add_subplot(gs[i, 1]) # second
    # NB: can't use imshow beacuse of the data are not on a regular linspaced grid
    img = ax.pcolormesh(x_radii/apo, y_radii/apo, flat_den_cgs.T, cmap = 'plasma',
                        norm = colors.LogNorm(vmin = dmin, vmax = dmax), rasterized = True)
    imgDiss = axDiss.pcolormesh(x_radiiDiss/apo, y_radiiDiss/apo, flat_diss_cgs.T, cmap = 'viridis',
                        norm = colors.LogNorm(vmin = Emin, vmax = Emax), rasterized = True)
    
    if i == 0:
        # Create an inset axis in the top right corner with more distance from the border
        ax_inset = inset_axes(ax, width="40%", height="40%", loc='lower left', borderpad = 2.5)

        # Define the zoom-in region in physical units
        x_min, x_max = -1, 0.5    # Y range in physical units
        y_min, y_max = -0.5, 0.5  # X range in physical units

        # Get mask for selected region
        x_mask = (x_radii/apo >= x_min) & (x_radii/apo <= x_max)
        y_mask = (y_radii/apo >= y_min) & (y_radii/apo <= y_max)

        # Apply the mask to get zoomed-in grid points
        x_zoom = x_radii[x_mask] / apo
        y_zoom = y_radii[y_mask] / apo
        flat_den_zoom = flat_den_cgs[np.ix_(x_mask, y_mask)]  # Zoomed-in density data

        # Use pcolormesh for the inset plot
        img_inset = ax_inset.pcolormesh(x_zoom, y_zoom, flat_den_zoom.T, cmap = 'plasma',
                                        norm=colors.LogNorm(vmin=dmin, vmax=dmax), rasterized=True)

        # Remove labels but keep ticks
        # Inset plot ticks
        ax_inset.tick_params(axis='x', direction='in', length=7, width=1.5, colors='white', labelsize=10)  # X-axis ticks
        ax_inset.tick_params(axis='y', direction='in', length=7, width=1.5, colors='white', labelsize=10)  # Y-axis ticks
        ax_inset.yaxis.tick_right()  # Move ticks to the right
        # ax_inset.set_xticklabels([])  # Remove numbers
        # ax_inset.set_yticklabels([])  # Remove numbers

        # Set the aspect ratio to match the physical size of the zoom region
        aspect_ratio = (x_max - x_min) / (y_max - y_min)
        # ax.set_aspect(aspect_ratio, adjustable='box')  # Main plot aspect ratio
        # ax_inset.set_aspect(aspect_ratio, adjustable='box')  # Inset plot aspect ratio

        # Add a rectangle to indicate the zoomed-in region
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                            linewidth=1, edgecolor='white', facecolor='none')
        ax.add_patch(rect)
        # Inset rectangle should have the same physical size as the main plot rectangle
        inset_rect = plt.Rectangle((0, 0), 1, 1, transform=ax_inset.transAxes,
                                linewidth=2, edgecolor='white', facecolor='none')
        ax_inset.add_patch(inset_rect)

    # Photosphere
    dataph = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
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
    
    ax.set_ylabel(r'$Y [r_{\rm a}]$')#, fontsize = 20)
    ax.tick_params(axis='x', which='major', width = .9, length = 7, color = 'white')
    ax.tick_params(axis='y', which='major', width = .9, length = 7, color = 'white')
    ax.set_xlim(-6, 2.5)
    ax.set_ylim(-3, 3)
    
    axDiss.tick_params(axis='x', which='major', width = .9, length = 7, color = 'white')
    axDiss.tick_params(axis='y', which='major', width = .9, length = 7, color = 'white')
    axDiss.set_xlim(x_min, x_max)
    axDiss.set_ylim(y_min, y_max)

    if i == 1:
        ax.text(-0.9 * 6, 0.8 * 3, f't = {np.round(tfb_single,1)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 22)
    else:
        ax.text(-0.9 * 6, 0.8 * 3, f't = {np.round(tfb_single,2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 22)
    axDiss.text(0.9 * x_min, 0.8 * y_max, r'$\dot{u}_{\rm irr}$ = ' + f'{LDiss[np.argmin(np.abs(tfbdiss-tfb_single))]:.1e} erg/s', color = 'white', fontsize = 18)

    if i == 2:
        ax.text(Rt/apo + 0.02, 0.05, r'$r_{\rm t}$', color = 'white', fontsize = 16)
        ax.text(a_mb/apo + 0.04, 0.16, r'$a_{\rm mb}$', color = 'white', fontsize =16)
        ax.text(1 + 0.01, 0.2, r'$r_{\rm a}$', color = 'white', fontsize =16)
    else:
        ax.tick_params(axis='x', which='both', labelbottom=False)
        axDiss.tick_params(axis='x', which='both', labelbottom=False)

cbar_ax = fig.add_subplot(gs[3, 0])  # Colorbar subplot below the first 3 panels
cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(which='major',length = 5)
cb.ax.tick_params(which='minor',length = 3)
cb.set_label(r'Column density [g/cm$^2$]', fontsize = 20)
ax.set_xlabel(r'$X [r_{\rm a}]$')#, fontsize = 20)
cbar_axDiss = fig.add_subplot(gs[3, 1])  # Colorbar subplot below the first 3 panels
cbDiss = fig.colorbar(imgDiss, cax=cbar_axDiss, orientation='horizontal')
cbDiss.ax.tick_params(which='major',length = 5)
cbDiss.ax.tick_params(which='minor',length = 3)
cbDiss.set_label(r'Dissipation rate column density [erg s$^{-1}$cm$^{-2}]$', fontsize = 20)
axDiss.set_xlabel(r'$X [r_{\rm a}]$')#, fontsize = 20)
plt.tight_layout()

if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/paper/3DenDissprojph.png', bbox_inches='tight', dpi = 300)
plt.show() 


#%%
proj_movie = False
overview = True
n_panels = 3

if proj_movie:
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps, Lum, tfb = split_data_red(check)
    dataDiss = np.loadtxt(f'{abspath}/data/{folder}/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows = 1)
    tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3]
    LDiss = LDiss * prel.en_converter/prel.tsol_cgs # [erg/s]

    # flux_data = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/{check}_phidx_fluxes.txt')
    # snaps_ph, tfb_ph, allindices_ph = flux_data[:, 0].astype(int), flux_data[:, 1], flux_data[:, 2:]
    # flux_test = np.zeros(len(snaps_ph))
    # fluxes = []
    # for i, snap in enumerate(snaps_ph):
    #     selected_lines = np.concatenate(np.where(snaps_ph == snap))
    #     # eliminate the even rows (photosphere indices) of allindices_ph
    #     _, selected_fluxes = selected_lines[0], selected_lines[1]
    #     fluxes.append(allindices_ph[selected_fluxes])
    #     flux_test[i] = np.sum(allindices_ph[selected_fluxes])
    # fluxes = np.array(fluxes)
    # tfb_ph, fluxes, snaps_ph = sort_list([tfb_ph, fluxes, snaps_ph], snaps_ph, unique=True)

    median_ph = np.zeros(len(snaps))

    for i, snap in enumerate(snaps):
        print(snap)
        photo = np.loadtxt(f'/Users/paolamartire/shocks/data/{folder}/photo/{check}_photo{snap}.txt')
        xph, yph, zph, volph, denph, alphaph = photo[0], photo[1], photo[2], photo[3], photo[4], photo[-4]
        # k = alphaph/denph
        r_ph = np.sqrt(xph**2 + yph**2 + zph**2)
        median_ph[i] = np.median(r_ph)
        if snap != 151:
            continue
        # print(k)
        # k_mean = 1/np.mean(1/k)
        # print(f'1/mean(1/k) {k_mean}, median k {np.median(k)}')
        x_denproj = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denxarray.npy')
        y_denproj = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denyarray.npy')
        flat_den = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Denproj{snap}.npy')
        flat_den_cgs = flat_den * prel.den_converter * prel.Rsol_cgs # [g/cm2]
        

        if n_panels == 3:
            flat_diss = np.load(f'/Users/paolamartire/shocks/data/{folder}/projection/Dissproj{snap}.npy')
            flat_diss_cgs = flat_diss * prel.en_den_converter * prel.Rsol_cgs / prel.tsol_cgs# [erg/s/cm2]
            # make =1 the nan values so they disappera with logcolor
            flat_diss_cgs_plot = flat_diss_cgs
            flat_diss_cgs_plot[np.isnan(flat_diss_cgs_plot)] = 1
            flat_diss_cgs_plot[flat_diss_cgs_plot == 0] = 1

            fig, (axd, axDiss, axLC) = plt.subplots(1,3, figsize = (50,16), gridspec_kw={'width_ratios': [1, 1, 0.8]})
            img = axDiss.pcolormesh(x_denproj/Rt, y_denproj/Rt, flat_diss_cgs.T, \
                            cmap = 'viridis', norm = colors.LogNorm(vmin = 1e14, vmax = 1e19))
            cbar = plt.colorbar(img, orientation = 'horizontal', pad = 0.15)
            cbar.set_label(r'Dissipation energy column density [erg s$^{-1}$cm$^{-2}]$')
            cbar.ax.tick_params(which='major', labelsize=25, width = 1, length = 12, pad = 10)
            cbar.ax.tick_params(which='minor',  width = .8, length = 8, pad = 10)
            axLC.plot(tfbdiss[:i+1], LDiss[:i+1], ls = '--', c = 'k', label = r'L$_{\rm diss}$')
            axDiss.set_xlim(-40, 10)
            axDiss.set_ylim(-15, 15)
        
        elif n_panels == 2:
            fig, (axd, axLC) = plt.subplots(1,2, figsize = (45,18), constrained_layout=True)        
        else:
            fig, axd = plt.subplots(1,1, figsize = (25,18), constrained_layout=True)

        img = axd.pcolormesh(x_denproj/Rt, y_denproj/Rt, flat_den_cgs.T, cmap = 'plasma', \
                          norm = colors.LogNorm(vmin=5e-2, vmax=1e7))
        cbar = plt.colorbar(img, orientation = 'horizontal', pad = 0.03 if n_panels == 2 else 0.1)
        cbar.set_label(r'Column density [g cm$^{-2}$]', fontsize = 60 if n_panels != 3 else 25)
        if n_panels != '':
            axd.plot(xph[indecesorbital]/Rt, yph[indecesorbital]/Rt, c = 'white', markersize = 12, marker = 'H', label = r'$R_{\rm ph}$')
            # just to connect the first and last 
            axd.plot([xph[first_idx]/Rt, xph[last_idx]/Rt], [yph[first_idx]/Rt, yph[last_idx]/Rt], c = 'white', markersize = 12, marker = 'H')
        if n_panels == 2:
            cbar.ax.tick_params(which='major', labelsize=60, width = 2, length = 16, pad = 10)
            cbar.ax.tick_params(which='minor',  width = 1.5, length = 11, pad = 10)
        if n_panels == 3:
            cbar.ax.tick_params(which='major', labelsize=25, width = 1, length = 12, pad = 10)
            cbar.ax.tick_params(which='minor',  width = .8, length = 8, pad = 10)
        axd.set_ylabel(r'Y [$r_{\rm t}$]', fontsize = 60 if n_panels != 3 else 25)
        axd.tick_params(axis='both', which='major', width = 1.5, length = 12, color = 'white', labelsize = 60 if n_panels != 3 else 25)
        axd.text(-70, 42, f't = {np.round(tfb[i],2)}' + r' $t_{\rm fb}$', color = 'white', fontsize = 60 if n_panels != 3 else 25)
        
        if n_panels != '':  
            if n_panels == 2:  
                imgLC = axLC.scatter(tfb[:i+1], Lum[:i+1], s = 75, c = median_ph[:i+1]/Rt, norm = colors.LogNorm(vmin = 1, vmax = 70), cmap = 'rainbow')
                cbar = plt.colorbar(imgLC, orientation = 'horizontal', pad = 0.03)
                cbar.set_label(r'Median $R_{\rm ph}$ [$R_{\rm t}$]', fontsize = 60)
                cbar.ax.tick_params(which='major', labelsize=60, width = 1.8, length = 18, pad = 10)
                cbar.ax.tick_params(which='minor',  width = 1.2, length = 15, pad = 10)
                for ax in [axd, axLC]:
                    ax.tick_params(axis='both', labelsize=60) 
                axLC.tick_params(axis='both', which='major', width = 2, length = 18)
                axLC.tick_params(axis='y', which='minor', width = 1.5, length = 15)
                fig.set_constrained_layout_pads(wspace=0.05)
            else: 
                imgLC = axLC.scatter(tfb[:i+1], Lum[:i+1], s = 25, label = r'L$_{\rm FLD}$', c = 'k')
                for ax in [axd, axDiss]:
                    ax.set_xlabel(r'X [$r_{\rm t}$]', fontsize = 25)
                axLC.legend(fontsize = 25)
                axLC.tick_params(axis='both', which='major', width = 1, length = 12)
                axLC.tick_params(axis='y', which='minor', width = .8, length = 10)
                # fig.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/projection3/denproj_diss{snap}.png')
            axLC.set_xlabel(r't $[t_{\rm fb}]$', fontsize = 60 if n_panels != 3 else 25)
            axLC.set_ylabel(r'L [erg/s]', fontsize = 60 if n_panels != 3 else 25)
            axLC.set_yscale('log')
            axLC.set_xlim(0.06, 2)
            axLC.text(0.15, 3e42, f't = {np.round(tfb[i]*t_fall_hour,1)}' + r' hours', fontsize = 60 if n_panels != 3 else 25)
            axLC.axhline(y=Ledd_cgs, c = 'k', linestyle = '-.', linewidth = 2)
            # axLC.text(0.1, 0.9*Ledd, r'$L_{\rm Edd}$', fontsize = 35)
        
        axLC.set_ylim(1e38, 8e42)
        axd.contour(xcfr_grid[0], ycfr_grid[0], cfr_grid[0], levels=[0], colors='white')
        axd.scatter(0,0,c= 'white', marker = 'x', s = 100)
        axd.set_xlim(-3*apo/Rt, 2*apo/Rt)
        axd.set_ylim(-2*apo/Rt, 2*apo/Rt)
        axd.set_xlabel(r'X [$r_{\rm t}$]', fontsize = 60 if n_panels != 3 else 25)

        plt.tight_layout()
        # fig.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/projection{n_panels}/denproj_{snap}.png')

        # plt.close()

if overview:
    # t_fall = 40 * np.power(Mbh/1e6, 1/2) * np.power(mstar,-1) * np.power(Rstar, 3/2)
    # t_fall_cgs = t_fall * 24 * 3600
    time = np.loadtxt(f'{abspath}/data/{folder}/slices/z/z0_time.txt')
    snaps, tfb_all = time[0], time[1]
    snaps = np.array([int(snap) for snap in snaps])
    snap_overview = np.arange(21,36)
    for snap in snap_overview:
        tfb = tfb_all[np.argmin(np.abs(snaps-snap))]
        x_cut, y_cut, z_cut, dim_cut, den_cut, temp_cut, ie_den_cut, Rad_den_cut, VX_cut, VY_cut, VZ_cut, Diss_den_cut, IE_den_cut, Press_cut = \
            np.load(f'{abspath}/data/{folder}/slices/z/z0slice_{snap}.npy')
        Diss_den_cut *= prel.en_den_converter/prel.tsol_cgs  # [erg/s/cm3]
        vminDiss, vmaxDiss = 1e3, 5e8
        Trad_cut = (Rad_den_cut*prel.en_den_converter/prel.alpha_cgs)**0.25  # [K]
        tmin, tmax = 1e4, 2e5
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (21,7))
        figT, (ax1T, ax2T) = plt.subplots(1,2, figsize = (21,7))
        img = ax1.scatter(x_cut/apo, y_cut/apo, c = Diss_den_cut, cmap = 'viridis', s= 1, \
                    norm = colors.LogNorm(vmin = vminDiss, vmax = vmaxDiss))
        ax1.set_title(r'All', fontsize = 22)

        img = ax2.scatter(x_cut[temp_cut>1e5]/apo, y_cut[temp_cut>1e5]/apo, c = Diss_den_cut[temp_cut>1e5], cmap = 'viridis', s= 1, \
                    norm = colors.LogNorm(vmin = vminDiss, vmax = vmaxDiss))
        cb = plt.colorbar(img)
        cb.set_label(r'$\dot{u}_{\rm irr}$ [erg s$^{-1}$ cm$^{-3}$]')
        ax2.set_title(r'T $>10^5$ K', fontsize = 22)

        img = ax1T.scatter(x_cut/apo, y_cut/apo, c = temp_cut, cmap = 'jet', s= 1, \
                    norm = colors.LogNorm(vmin = tmin, vmax = tmax))
        cb = plt.colorbar(img)
        cb.set_label(r'T [K]')
        img = ax2T.scatter(x_cut/apo, y_cut/apo, c = Trad_cut, cmap = 'jet', s= 1, \
                    norm = colors.LogNorm(vmin = tmin, vmax = tmax))
        cb = plt.colorbar(img)
        cb.set_label(r'T$_{\rm rad}$ [K]')

        for ax in [ax1, ax2, ax1T, ax2T]:
            ax.set_xlabel(r'$X [r_{\rm a}]$')
            ax.set_xlim(-1, .1)#(-340,25)
            ax.set_ylim(-.4, .4)#(-70,70)
            if ax in [ax1, ax1T]:
                ax.set_ylabel(r'$Y [r_{\rm a}]$')

        fig.suptitle(f't = {np.round(tfb, 2)}' + r'$t_{\rm fb}$', fontsize = 20)
        figT.suptitle(f't = {np.round(tfb, 2)}' + r'$t_{\rm fb}$', fontsize = 20)
        fig.tight_layout()
        figT.tight_layout()
        fig.savefig(f'{abspath}/Figs/{folder}/slices/DissOrbPl{snap}.png')
        figT.savefig(f'{abspath}/Figs/{folder}/slices/TempOrbPl{snap}.png')
        plt.close()
        


    # %%
