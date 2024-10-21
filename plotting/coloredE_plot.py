import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
# import colorcet
import matplotlib.colors as colors
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

Rt = Rstar * (Mbh/mstar)**(1/3)
R0 = 0.6 * Rt
apo = Rt**2 / Rstar #2 * Rt * (Mbh/mstar)**(1/3)

#
## DECISIONS
##
save = True
xaxis = 'radii'
res1 = '' #'', 'HiRes', 'DoubleRad'
res2 = 'HiRes' 
weight = 'mixed' #'mixed', 'weightE' or '' if you have weight for vol/mass
cut = '' # or '' or '_NOcut' or '_all1e-19' (if weight == 'mixed', cut = '' but it's 1e-19)
if xaxis == 'angles':
    apo = 1
#
## DATA
#

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'/Users/paolamartire/shocks/data/{folder}'
# Res1 data
datares1 = np.load(f'{path}{res1}/colormapE_Alice/coloredE_{res1}_{xaxis}{weight}{cut}.npy') #shape (3, len(tfb), len(radii))
tfb_datares1 = np.loadtxt(f'{path}{res1}/colormapE_Alice/coloredE_{res1}_days.txt')
snap_res1 = tfb_datares1[0]
tfb_res1_all = tfb_datares1[1]
radiires1 = np.load(f'{path}{res1}/colormapE_Alice/{xaxis}En_{res1}.npy')
if weight == 'mixed':
    col_ieres1, col_orb_enres1, col_Radres1 = datares1[0], datares1[1], datares1[2]
else:
    col_ieres1, col_orb_enres1, col_Radres1_cutsmall, col_Radres1 = datares1[0], datares1[1], datares1[2], datares1[3]
    col_Radres1_cutsmall *= prel.en_den_converter
# convert to cgs
col_ieres1 *= prel.en_converter/prel.Msol_cgs
col_orb_enres1 *= prel.en_converter/prel.Msol_cgs
col_Radres1 *= prel.en_den_converter
abs_col_orb_enres1 = np.abs(col_orb_enres1)

#%% Plot 
fig, ax = plt.subplots(1,3, figsize = (16,5))
img = ax[0].pcolormesh(radiires1/apo, tfb_res1_all, abs_col_orb_enres1, norm=colors.LogNorm(vmin=4e16, vmax = 1e18), cmap = 'viridis')
cb = fig.colorbar(img)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)

img = ax[1].pcolormesh(radiires1/apo, tfb_res1_all, col_ieres1,  norm=colors.LogNorm(vmin=7e12, vmax = 4e14), cmap = 'viridis')
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[1].set_title('Specific internal energy', fontsize = 20)

img = ax[2].pcolormesh(radiires1/apo, tfb_res1_all, col_Radres1, norm=colors.LogNorm(vmin=1e4, vmax= 7e9), cmap = 'viridis')
cb = fig.colorbar(img)
ax[2].set_title('Radiation energy density', fontsize = 20)
cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)
for i in range(3):
    ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
    ax[i].set_xscale('log')
    ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}{res1}/coloredE_{xaxis}{weight}{cut}.pdf')


#%% Res2 data
datares2 = np.load(f'{path}{res2}/colormapE_Alice/coloredE_{res2}_{xaxis}{weight}{cut}.npy')
tfb_datares2 = np.loadtxt(f'{path}{res2}/colormapE_Alice/coloredE_{res2}_days.txt')
snap_res2 = tfb_datares2[0]
tfb_res2 = tfb_datares2[1]
radiires2 = np.load(f'{path}{res2}/colormapE_Alice/{xaxis}En_{res2}.npy')
if weight == 'mixed':
    col_ieres2, col_orb_enres2, col_Radres2 = datares2[0], datares2[1], datares2[2]
else:
    col_ieres2, col_orb_enres2, col_Radres2, col_Radres2_nofluff = datares2[0], datares2[1], datares2[2], datares2[3]
    col_Radres2_nofluff *= prel.en_den_converter

# convert to cgs
col_ieres2 *= prel.en_converter/prel.Msol_cgs
col_orb_enres2 *= prel.en_converter/prel.Msol_cgs
col_Radres2 *= prel.en_den_converter
abs_col_orb_enres2 = np.abs(col_orb_enres2)

# Consider Low data only up to the time of the res2 data
n_res2 = len(col_ieres2)
snap_res1 = snap_res1[:n_res2]
tfb_res1 = tfb_res1_all[:n_res2]
col_ieres1 = col_ieres1[:n_res2]
col_orb_enres1 = col_orb_enres1[:n_res2]
abs_col_orb_enres1 = abs_col_orb_enres1[:n_res2]
col_Radres1 = col_Radres1[:n_res2]
# col_Radres1_cutsmall = col_Radres1_cutsmall[:n_res2]
#%%
# PLOT
##
# check the difference in time
# plt.scatter(snap_res2,tfb_Low-tfb_res2, color='k', label = 'time')
# # plt.plot(snap_Low-snap_res2, color='r', label = 'snap')
# plt.scatter(snap_res2[37], tfb_Low[37]-tfb_res2[37], c='dodgerblue')
# plt.scatter(snap_res2[119], tfb_Low[119]-tfb_res2[119], c='orange')
# plt.scatter(snap_res2[129], tfb_Low[129]-tfb_res2[129], c='orchid')
# plt.xlabel('Snapshot')
# plt.ylabel(r'$t_L-t_H$')
# plt.legend(fontsize = 20)

#%%
p40_orb_ensix = np.percentile(abs_col_orb_enres1, 40)
p95_orb_ensix = 1e18#np.percentile(abs_col_orb_enres1, 99)
p40_iesix = np.percentile(col_ieres1, 40)
p95_iesix = np.percentile(col_ieres1, 95)
p40_Radsix = np.percentile(col_Radres1, 40)
p95_Radsix = np.percentile(col_Radres1, 95)

cmap = plt.cm.viridis
norm_orb_ensix = colors.LogNorm(vmin=5e16, vmax = 1e18)#vmin=p40_orb_ensix, vmax=p95_orb_ensix)
norm_iesix = colors.LogNorm(vmin=7e12, vmax = 4e14)#p40_iesix, vmax=p95_iesix)
norm_Radsix = colors.LogNorm(vmin=1e4, vmax= 7e9)#p40_Radsix, vmax=p95_Radsix)

fig, ax = plt.subplots(2,3, figsize = (14,8))
# Res1
img = ax[0][0].pcolormesh(radiires1/apo, tfb_res1, abs_col_orb_enres1, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[0][0].text(0.4, 0.77,  f'{res1} res', fontsize = 18, color = 'w')

img = ax[0][1].pcolormesh(radiires1/apo, tfb_res1, col_ieres1,  norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
ax[0][1].set_title('Specific internal energy', fontsize = 20)

img = ax[0][2].pcolormesh(radiires1/apo, tfb_res1, col_Radres1, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
ax[0][2].set_title('Radiation energy density', fontsize = 20)
cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

# res2
img = ax[1][0].pcolormesh(radiires2/apo, tfb_res2, abs_col_orb_enres2, norm=norm_orb_ensix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)
if res2 == 'HiRes':
    ax[1][0].text(0.4, 0.77, f'High res', fontsize = 18, color = 'w')

img = ax[1][1].pcolormesh(radiires2/apo, tfb_res2, col_ieres2, norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)

img = ax[1][2].pcolormesh(radiires2/apo, tfb_res2, col_Radres2, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

for i in range(2):
    for j in range(3):
        # ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        # ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)
        # Grid for radii, to be matched with the cfr in slices.py
        if xaxis == 'radii':
            ax[i][j].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            # ax[i][j].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            # ax[i][j].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            # ax[i][j].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            # ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
            # ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
            # ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
            # ax[i][j].axvline(1, c = 'white', linewidth = 0.4)
            ax[i][j].set_xscale('log')
        
        elif xaxis == 'angles':
            ax[i][j].axvline(-2.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(-1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(2.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)

        if i == 1:
            if xaxis == 'radii':
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
            elif xaxis == 'angles':
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
                ax[i][j].set_xlabel(r'$\theta$ [rad]', fontsize = 20)

# Layout
ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')

plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/multiple/6coloredE_{xaxis}{weight}{cut}.pdf')
plt.show()

# %% Plot (absolute) differences. They start from the same point
diff_ie = col_ieres1 - col_ieres2
diff_orb_en = col_orb_enres1 - col_orb_enres2
diff_Rad = col_Radres1 - col_Radres2
diff_ie_abs = np.abs(diff_ie)
diff_orb_en_abs = np.abs(diff_orb_en)
diff_Rad_abs = np.abs(diff_Rad)
# set to zero where the value of one of them is zero
diff_ie_abs[col_ieres1==0 ] = 0
diff_ie_abs[col_ieres2==0] = 0
diff_orb_en_abs[col_orb_enres1==0] = 0
diff_orb_en_abs[col_orb_enres2==0] = 0
diff_Rad_abs[col_Radres1==0] = 0
diff_Rad_abs[col_Radres2==0] = 0
# pass to log
# difflog_ie = np.log10(diff_ie_abs)
# difflog_orb_en = np.log10(diff_orb_en_abs)
# difflog_Rad = np.log10(diff_Rad_abs)
# difflog_ie[np.isnan(difflog_ie)] = 0
# difflog_orb_en[np.isnan(difflog_orb_en)] = 0
# difflog_Rad[np.isnan(difflog_Rad)] = 0

fig, ax = plt.subplots(1,3, figsize = (18,5))

img = ax[0].pcolormesh(radiires2/apo, tfb_res2, diff_orb_en_abs, norm=colors.LogNorm(vmin=1e15, vmax=1e17),
                     cmap = 'bwr')#, vmin = 15, vmax = 17)
cb = fig.colorbar(img)
cb.set_label(r'$|\Delta|$', fontsize = 20)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 14)

img = ax[1].pcolormesh(radiires2/apo, tfb_res2, diff_ie_abs, norm=colors.LogNorm(vmin=4e11, vmax=4e13),
                     cmap = 'bwr')#, vmin = 11.5, vmax = 14)
cb = fig.colorbar(img)
cb.set_label(r'$|\Delta|$', fontsize = 20)
ax[1].set_title('Specific internal energy', fontsize = 14)

img = ax[2].pcolormesh(radiires2/apo, tfb_res2, diff_Rad_abs, norm=colors.LogNorm(vmin=1e4, vmax=4e7),
                     cmap = 'bwr')#, vmin = 3, vmax = 8.5)
cb = fig.colorbar(img)
ax[2].set_title('Radiation energy density', fontsize = 14)
cb.set_label(r'$|\Delta|$', fontsize = 20)#$\Delta|E_{rad}/$Vol$|$', fontsize = 14, labelpad = 5)

for i in range(3):
    ax[i].axhline(0.205, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.52, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.75, c = 'white', linewidth = 0.4)
    if xaxis == 'radii':
        ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        # ax[i].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        # ax[i].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        # ax[i].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        # ax[i].axvline(0.1, c = 'white', linewidth = 0.4)
        # ax[i].axvline(0.3, c = 'white', linewidth = 0.4)
        # ax[i].axvline(0.5, c = 'white', linewidth = 0.4)
        # ax[i].axvline(1, c = 'white', linewidth = 0.4)
        ax[i].set_xscale('log')
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)

    elif xaxis == 'angles':
        ax[i].axvline(-2.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(-1, c = 'white', linewidth = 0.4)
        ax[i].axvline(2.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(1, c = 'white', linewidth = 0.4)
        ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
        ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
        ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)
ax[0].set_xlabel(r'$R/R_a$', fontsize = 20)
plt.tick_params(axis = 'both', which = 'both', direction='in')
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_diff_{xaxis}{weight}{cut}.png')
plt.show()

# %% Relative differences
denominator_orb_en = (col_orb_enres1 + col_orb_enres2)/2
denominator_ie = (col_ieres1 + col_ieres2)/2
denominator_Rad = (col_Radres1 + col_Radres2)/2
rel_orb_en = np.abs(diff_orb_en / denominator_orb_en)
rel_ie = np.abs(diff_ie / denominator_ie)
rel_Rad = np.abs(diff_Rad / denominator_Rad)

rel_orb_en[np.isnan(rel_orb_en)] = 0 #nothing
rel_ie[np.isnan(rel_ie)] = 0
rel_Rad[np.isnan(rel_Rad)] = 0


#%%
# take just the time after 0.3
minidx = np.argmin(np.abs(tfb_res2-0.3))
median_orb_en = np.median(rel_orb_en)#np.median(rel_orb_en[minidx:])
median_ie = np.median(rel_ie)
median_Rad = np.median(rel_Rad)

print(f'Median relative difference in Orbital energy: {np.round(median_orb_en,2)}')
print(f'Median relative difference in IE: {np.round(median_ie,2)}')
print(f'Median relative difference in Radiation energy density: {np.round(median_Rad,2)}')

rel_orb_en_forlog = np.copy(rel_orb_en) #you need it for the Log
rel_ie_forlog = np.copy(rel_ie)
rel_Rad_forlog = np.copy(rel_Rad)
rel_orb_en_forlog[rel_orb_en_forlog==0] = 1 #you need it for the Log
rel_ie_forlog[rel_ie_forlog==0] =1
rel_Rad_forlog[rel_Rad_forlog==0] = 1

cmap = plt.cm.inferno
norm_orb_en = colors.LogNorm(vmin=2e-3, vmax=6e-1)#np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 5), vmax=np.percentile(rel_orb_en_forlog[rel_orb_en_forlog!=1], 95))
norm_ie = colors.LogNorm(vmin=2e-2, vmax=4e-1)#np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 5), vmax=np.percentile(rel_ie_forlog[rel_ie_forlog!=1], 95))
norm_Rad = colors.LogNorm(vmin=0.04, vmax=1.5)#np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 5), vmax=np.percentile(rel_Rad_forlog[rel_Rad_forlog!=1], 95))


fig, ax = plt.subplots(1,3, figsize = (20,6))
img = ax[0].pcolormesh(radiires2/apo, tfb_res2, rel_orb_en_forlog, cmap=cmap, norm=norm_orb_en)#, vmin = np.min(rel_orel_orb_en_fologrb_en), vmax = 0.2)
cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=20)
ax[0].set_title('Specific (absolute) orbital energy', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#Relative difference $|E_{orb}|$/Mass', fontsize = 14, labelpad = 5)

img = ax[1].pcolormesh(radiires2/apo, tfb_res2, rel_ie_forlog, cmap=cmap, norm=norm_ie)#, vmin = np.min(rel_ie_forlog), vmax = np.max(rel_ie_forlog))
cb = fig.colorbar(img)
cb.ax.tick_params(labelsize=20)
ax[1].set_title('Specific internal energy', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#Relative difference $|$IE$|$/Mass', fontsize = 14, labelpad = 5)

img = ax[2].pcolormesh(radiires2/apo, tfb_res2, rel_Rad_forlog, cmap=cmap, norm=norm_Rad)#, vmin = np.min(rel_Rad_forlog), vmax = np.max(rel_Rad_forlog))
cb = fig.colorbar(img)
# cb.ax.tick_params(labelsize=2)
ax[2].set_title('Radiation energy density', fontsize = 20)
cb.set_label('Relative difference', fontsize = 25)#'Relative difference $|E_{rad}|$/Vol', fontsize = 14, labelpad = 5)

for i in range(3):
    # ax[i].axhline(0.205, c = 'white', linewidth = 0.4)
    # ax[i].axhline(0.52, c = 'white', linewidth = 0.4)
    # ax[i].axhline(0.75, c = 'white', linewidth = 0.4)
    if xaxis == 'radii':
        ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        # ax[i].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        # ax[i].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        # ax[i].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        # ax[i].axvline(0.1, c = 'white', linewidth = 0.4)
        # ax[i].axvline(0.3, c = 'white', linewidth = 0.4)
        # ax[i].axvline(0.5, c = 'white', linewidth = 0.4)
        # ax[i].axvline(1, c = 'white', linewidth = 0.4)
        ax[i].set_xscale('log')
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)
        ax[i].set_xlabel(r'$R/R_a$', fontsize = 20)

    elif xaxis == 'angles':
        if xaxis == 'angles':
            ax[i].axvline(-2.5, c = 'white', linewidth = 0.4)
            ax[i].axvline(-1, c = 'white', linewidth = 0.4)
            ax[i].axvline(2.5, c = 'white', linewidth = 0.4)
            ax[i].axvline(1, c = 'white', linewidth = 0.4)
            ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
            ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)
            ax[i].set_xlabel(r'$\theta$ [rad]', fontsize = 20)

# Layout
ax[0].set_ylabel(r't/t$_{fb}$', fontsize = 25)

plt.tick_params(axis = 'both', which = 'both', direction='in')
# plt.suptitle(r'Relative differences: $|$Low-High$|$/mean $M_{BH}=10^4 M_\odot, m_\star$ = ' + f'{mstar} M$_\odot, R_\star$ = {Rstar} R$_\odot$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_relative_diff_{xaxis}{weight}{cut}.pdf')
plt.show()

#%%
if xaxis == 'radii':
    res = res2
    tfb_res = tfb_res2
    # Load data Low res
    dataenergycut = np.load(f'{path}/coloredEenergy_{res}_{xaxis}.npy') 
    ie, orb_en, rad = dataenergycut[0], dataenergycut[1], dataenergycut[2]
    ie, orb_en, rad = ie[:n_res2], orb_en[:n_res2], rad[:n_res2]
    dataenergyNOcut = np.load(f'{path}/coloredEenergy_{res}_{xaxis}NOcut.npy') 
    ieNOcut, orb_enNOcut, radNOcut = dataenergyNOcut[0], dataenergyNOcut[1], dataenergyNOcut[2]
    ieNOcut, orb_enNOcut, radNOcut = ieNOcut[:n_res2], orb_enNOcut[:n_res2], radNOcut[:n_res2]
    # compute the differences to see how much energy is cut
    orb_diff = orb_enNOcut - orb_en
    orb_diff_abs = np.abs(orb_diff)
    iediff = ieNOcut - ie
    rad_diff = radNOcut - rad
    norm_orb = np.sum(orb_enNOcut, axis = 0)
    # norm_orb = np.transpose([norm_orb] * len(radiires2))
    norm_ie = np.sum(ieNOcut, axis = 0)
    # norm_ie = np.transpose([norm_ie] * len(radiires2))
    norm_rad = np.sum(radNOcut, axis = 0)
    # norm_rad = np.transpose([norm_rad] * len(radiires2))
    # relative differences
    orb_diff_abs_rel = np.abs(orb_diff/orb_enNOcut) 
    iediff_rel = np.abs(iediff/ieNOcut)
    rad_diff_rel = np.abs(rad_diff/radNOcut)
    # orb_diff_abs_rel = np.copy(orb_diff_abs)
    # iediff_rel = np.copy(iediff)
    # rad_diff_rel = np.copy(rad_diff)
    # for i in range(len(tfb_res)):
    #     orb_diff_abs_rel[i] /= np.sum(np.abs(orb_enNOcut[i]))
    #     iediff_rel[i] /= np.sum(ieNOcut[i])
    #     rad_diff_rel[i] /= np.sum(radNOcut[i])

    colors_tick = ['w', 'k']
    fig, ax = plt.subplots(2,3, figsize = (14,8))
    img = ax[0][0].pcolormesh(radiires1/apo, tfb_res, orb_diff_abs*prel.en_converter,  cmap = cmap, norm = colors.LogNorm(vmin=1e42, vmax = 1e46))
    cb = fig.colorbar(img)
    ax[0][0].set_title('Absolute orbital energy', fontsize = 20)

    img = ax[0][1].pcolormesh(radiires1/apo, tfb_res, iediff*prel.en_converter, cmap = cmap, norm = colors.LogNorm(vmin=1e40, vmax = 1e43))
    cb = fig.colorbar(img)
    ax[0][1].set_title('Internal energy', fontsize = 20)

    img = ax[0][2].pcolormesh(radiires1/apo, tfb_res, rad_diff*prel.en_converter, cmap = cmap, norm = colors.LogNorm(vmin=1e40, vmax = 1e45))
    cb = fig.colorbar(img)
    ax[0][2].set_title('Radiation energy', fontsize = 20)
    cb.set_label(r'Energy lost[erg]', fontsize = 20, labelpad = 2)

    img = ax[1][0].pcolormesh(radiires2/apo, tfb_res, orb_diff_abs_rel, cmap = 'plasma', vmin=0, vmax = .5)
    cb = fig.colorbar(img)

    img = ax[1][1].pcolormesh(radiires2/apo, tfb_res, iediff_rel, cmap = 'plasma', vmin=0, vmax = .5)
    cb = fig.colorbar(img)

    img = ax[1][2].pcolormesh(radiires2/apo, tfb_res, rad_diff_rel, cmap = 'plasma', vmin=0, vmax = 1)
    cb = fig.colorbar(img)
    cb.set_label(r'$\Delta_{rel}$', fontsize = 20, labelpad = 2)

    for i in range(2):
        for j in range(3):
            ax[i][j].set_xscale('log')
            ax[i][j].tick_params(axis='both', which='major', color = colors_tick[i], size=5)
            ax[i][j].tick_params(axis='both', which='minor', color = colors_tick[i], size=3)
            if i == 1:
                ax[i][j].set_xlabel(r'$R/R_a$', fontsize = 20)
            if j == 0:
                ax[i][j].set_ylabel(r't/t$_{fb}$', fontsize = 20)

    plt.suptitle(f'Cut energy in {res} res: difference between NOcut and cut 1e-9', fontsize = 20)
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/{res}/whatIsCut.png')
    plt.show()
    #%% Lines
    indices = [np.argmin(np.abs(tfb_res1-0.5)), np.argmin(np.abs(tfb_res1-0.7)), np.argmin(np.abs(tfb_res1-0.86))]
    heigth_text = np.array([3e40, 1e43, 4e44])
    colors_indices = ['navy', 'royalblue', 'deepskyblue']
    col_Lum1 = np.load(f'{path}/coloredE_Low_radiimixedLum.npy') 
    radiiLum1 = np.load(f'{path}/radiiEn_LowLum.npy')
    col_Lum1 = col_Lum1 * prel.en_den_converter
    col_Lum2 = np.load(f'{path}/coloredE_HiRes_radiimixedLum.npy') 
    radiiLum2 = np.load(f'{path}/radiiEn_HiResLum.npy')
    col_Lum2 = col_Lum2 * prel.en_den_converter
    col_Lum1 = col_Lum1[:len(col_Lum2)]
    radiiLum1 = np.repeat([radiiLum1],len(col_Lum1), axis = 0)
    radiiLum2 = np.repeat([radiiLum2],len(col_Lum2), axis = 0)
    
    Lum_cgs = col_Lum1  * prel.c * 4 * np.pi * (radiiLum1*prel.Rsol_cgs)**2 
    Lum2_cgs = col_Lum2 * prel.c * 4 * np.pi * (radiiLum2*prel.Rsol_cgs)**2 
    denom = (Lum_cgs + Lum2_cgs)/2
    Lum_difference = np.abs(Lum_cgs[indices]-Lum2_cgs[indices])/denom[indices]

    img, ax = plt.subplots(1,2, figsize = (20,7))
    for i,idx in enumerate(indices):
        where_zero1 = np.concatenate(np.where(Lum_cgs[idx]<1e-18))
        where_zero2 = np.concatenate(np.where(Lum2_cgs[idx]<1e-18))
        where_zero = np.concatenate([where_zero1, where_zero2])
        Lum_cgs_toplot= np.delete(Lum_cgs[idx], where_zero)
        Lum2_cgs_toplot = np.delete(Lum2_cgs[idx], where_zero)
        radiiLum1_toplot = np.delete(radiiLum1[idx], where_zero)
        radiiLum2_toplot = np.delete(radiiLum2[idx], where_zero)
        Lum_difference_toplot = np.delete(Lum_difference[i], where_zero)
        ax[0].text(np.min(radiiLum1_toplot)/apo, heigth_text[i], f't = {np.round(tfb_res1[indices[i]],2)}' +  r'$t_{fb}$', fontsize = 18)
        if i == 0:
            ax[0].plot(radiiLum1_toplot/apo, Lum_cgs_toplot, c = colors_indices[i], label = f'{res1} res')#t/tfb = {np.round(tfb_Low,2)}')
            ax[0].plot(radiiLum2_toplot/apo, Lum2_cgs_toplot, '--', c = colors_indices[i], label = f'{res2} res')#t/tfb = {np.round(tfb_res2,2)}')
            ax[1].plot(radiiLum1_toplot/apo, Lum_difference_toplot, c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low,2)}')
        else:   
            ax[0].plot(radiiLum1_toplot/apo, Lum_cgs_toplot, c = colors_indices[i])
            ax[0].plot(radiiLum2_toplot/apo, Lum2_cgs_toplot, '--', c = colors_indices[i])
            ax[1].plot(radiiLum1_toplot/apo, Lum_difference_toplot, c = colors_indices[i])

        mean_error =  np.round(np.mean(Lum_difference_toplot[-10:-1]),2)
        print(f'Mean relative error for t/tfb = {tfb_res1[indices[i]]} is {mean_error}')
    
    ax[0].set_ylim(1e40, 5e45)
    ax[1].set_ylim(0.1, 1.8)
    # ax[0].text(15, np.max(Lum_cgs[0]), r'$t/t_{fb}$ = '+ f'{np.round(tfb_res1[indices[0]],2)}', fontsize = 20)
    # ax[0].text(20, 1e44, r'$t/t_{fb}$ = '+ f'{np.round(tfb_res1[indices[1]],2)}', fontsize = 20)
    # ax[0].text(20, 1e45, r'$t/t_{fb}$ = '+ f'{np.round(tfb_res1[indices[2]],2)}', fontsize = 20)
    # ax[1].text(620, 0.7, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[0][-10:-1]),2)}', fontsize = 20)
    # ax[1].text(620, 0.5, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[1][-10:-1]),2)}', fontsize = 19)
    # ax[1].text(620, 0.4, r'$\Delta_{rel}\approx$ '+ f'{np.round(np.mean(Lum_difference[2][-10:-1]),2)}', fontsize = 20)

    ax[0].tick_params(axis='both', which='major', labelsize=25)
    ax[1].tick_params(axis='both', which='major', labelsize=25)
    ax[0].tick_params(axis='both', which='minor', size=4)
    ax[1].tick_params(axis='both', which='minor', size=4)
    ax[0].tick_params(axis='both', which='major', size=6)
    ax[1].tick_params(axis='both', which='major', size=6)
    ax[0].set_xlabel(r'R/R$_a$', fontsize = 28)
    ax[1].set_xlabel(r'$R/R_a$', fontsize = 28)
    ax[0].set_ylabel(r'Luminosity [erg/s]', fontsize = 25)
    ax[1].set_ylabel(r'Relative difference', fontsize = 25, labelpad = 1)
    ax[0].loglog()
    ax[0].legend(fontsize = 25)
    ax[0].grid()
    ax[1].grid()
    ax[1].loglog()
    plt.subplots_adjust(wspace=4)
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/Luminosity{weight}{cut}.pdf')
    plt.show()

    # for i,idx in enumerate(indices):
    #     if i ==2:
    #         continue
    #     after = Lum2_cgs[indices[i+1]]
    #     before = Lum2_cgs[indices[i]]
    #     mean_lastafter = np.mean(after[-10:-1])
    #     mean_lastbefore = np.mean(before[-10:-1])
    #     print(tfb_res1[indices[i+1]], tfb_res1[indices[i]])
        # print(f'The relative difference of the last point of the high res line is {np.round((mean_lastafter-mean_lastbefore),2)}')


# %%
