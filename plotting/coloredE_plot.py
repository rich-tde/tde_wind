import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import matplotlib.pyplot as plt
import colorcet
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
save = False
xaxis = 'radii'
res1 = 'Low'
res2 = 'HiRes' #'HiRes', 'LowDoubleRad'
weight = 'weightE' #'weightE' or '' if you have weight for vol/mass
if xaxis == 'angles':
    apo = 1
#
## DATA
#

folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
path = f'/Users/paolamartire/shocks/data/{folder}/colormapE_Alice'
# Res1 data
datares1 = np.load(f'{path}/coloredE_{res1}_{xaxis}{weight}_all1e-19.npy') #shape (3, len(tfb), len(radii))
tfb_datares1 = np.loadtxt(f'{path}/coloredE_{res1}_days.txt')
snap_res1 = tfb_datares1[0]
tfb_res1 = tfb_datares1[1]
radiires1 = np.load(f'{path}/{xaxis}En_{res1}.npy')
col_ieres1, col_orb_enres1, col_Radres1_lowcut, col_Radres1 = datares1[0], datares1[1], datares1[2], datares1[3]
# convert to cgs
col_ieres1 *= prel.en_converter/prel.Msol_to_g
col_orb_enres1 *= prel.en_converter/prel.Msol_to_g
col_Radres1_lowcut *= prel.en_den_converter
col_Radres1 *= prel.en_den_converter
abs_col_orb_enres1 = np.abs(col_orb_enres1)

# Res2 data
datares2 = np.load(f'{path}/coloredE_{res2}_{xaxis}{weight}_all1e-19.npy')
tfb_datares2 = np.loadtxt(f'{path}/coloredE_{res2}_days.txt')
snap_res2 = tfb_datares2[0]
tfb_res2 = tfb_datares2[1]
radiires2 = np.load(f'{path}/{xaxis}En_{res2}.npy')
col_ieres2, col_orb_enres2, col_Radres2, col_Radres2_nofluff = datares2[0], datares2[1], datares2[2], datares2[3]

# convert to cgs
col_ieres2 *= prel.en_converter/prel.Msol_to_g
col_orb_enres2 *= prel.en_converter/prel.Msol_to_g
col_Radres2 *= prel.en_den_converter
col_Radres2_nofluff *= prel.en_den_converter
abs_col_orb_enres2 = np.abs(col_orb_enres2)

# Consider Low data only up to the time of the res2 data
n_res2 = len(col_ieres2)
snap_res1 = snap_res1[:n_res2]
tfb_res1 = tfb_res1[:n_res2]
col_ieres1 = col_ieres1[:n_res2]
col_orb_enres1 = col_orb_enres1[:n_res2]
abs_col_orb_enres1 = abs_col_orb_enres1[:n_res2]
col_Radres1 = col_Radres1[:n_res2]
col_Radres1_lowcut = col_Radres1_lowcut[:n_res2]
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
ax[0][0].text(np.min(radiires2/apo), 0.1, f'{res1} res', fontsize = 20)

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
ax[1][0].text(np.min(radiires2/apo), 0.1, f'{res2} res', fontsize = 20)

img = ax[1][1].pcolormesh(radiires2/apo, tfb_res2, col_ieres2, norm=norm_iesix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Specific energy [erg/g]', fontsize = 20, labelpad = 2)

img = ax[1][2].pcolormesh(radiires2/apo, tfb_res2, col_Radres2, norm=norm_Radsix, cmap = cmap)
cb = fig.colorbar(img)
cb.set_label(r'Energy density [erg/cm$^3$]', fontsize = 20, labelpad = 2)

for i in range(2):
    for j in range(3):
        ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)
        # Grid for radii, to be matched with the cfr in slices.py
        if xaxis == 'radii':
            ax[i][j].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)
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
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_{xaxis}{weight}.png')
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
        ax[i].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(1, c = 'white', linewidth = 0.4)
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
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_diff_{xaxis}{weight}.png')
plt.show()

######################
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
    ax[i].axhline(0.205, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.52, c = 'white', linewidth = 0.4)
    ax[i].axhline(0.75, c = 'white', linewidth = 0.4)
    if xaxis == 'radii':
        ax[i].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
        ax[i].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i].transAxes, color = 'k')
        ax[i].axvline(0.1, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.3, c = 'white', linewidth = 0.4)
        ax[i].axvline(0.5, c = 'white', linewidth = 0.4)
        ax[i].axvline(1, c = 'white', linewidth = 0.4)
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
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredE_relative_diff_{xaxis}{weight}.png')
plt.show()

# %% Do just for radiation without fluff as  check
diff_Rad_nofluff = col_Radres1_lowcut - col_Radres2_nofluff
denominator_Rad_nofluff = (col_Radres1_lowcut + col_Radres2_nofluff)/2
rel_Rad_nofluff = np.abs(diff_Rad_nofluff / denominator_Rad_nofluff)
rel_Rad_nofluff[np.isnan(rel_Rad_nofluff)] = 0
normRadnofluff = colors.LogNorm(vmin=np.percentile(col_Radres1_lowcut[col_Radres1_lowcut>0], 5), vmax=np.percentile(col_Radres1_lowcut[col_Radres1_lowcut>0], 95))

fig, ax = plt.subplots(2,2, figsize = (12,10))
img = ax[0][0].pcolormesh(radiires1/apo, tfb_res1, col_Radres1_lowcut, norm=normRadnofluff, cmap = 'viridis')
cb = fig.colorbar(img)
cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 15, labelpad = 5)
ax[0][0].text(np.min(radiires1/apo), 0.15, f'{res1} res', fontsize = 25)
img = ax[0][1].pcolormesh(radiires2/apo, tfb_res2, col_Radres2_nofluff, norm=normRadnofluff, cmap = 'viridis')
cb = fig.colorbar(img)
cb.set_label(r'Radiation energy density [erg/cm$^3$]', fontsize = 15, labelpad = 5)
ax[0][1].text(np.min(radiires2/apo), 0.15, f'{res2} res', fontsize = 25)
# plot the relaive difference
img = ax[1][0].pcolormesh(radiires2/apo, tfb_res2, rel_Rad_nofluff, cmap='inferno', norm=norm_Rad)
cb = fig.colorbar(img)
cb.set_label('Relative difference', fontsize = 15, labelpad = 5)
for i in range(2):
    ax[i][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
    for j in range(2):
        ax[i][j].axhline(0.205, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.52, c = 'white', linewidth = 0.4)
        ax[i][j].axhline(0.75, c = 'white', linewidth = 0.4)
        if xaxis == 'radii':
            ax[i][j].axvline(Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(Rt/apo, 0.65, r'R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(1.5*Rt/apo, linestyle ='dashed', c = 'white', linewidth = 0.8)
            ax[i][j].text(1.5*Rt/apo+0.1, 0.65, r'1.5R$_t$', fontsize = 14, rotation = 90, transform = ax[i][j].transAxes, color = 'k')
            ax[i][j].axvline(0.1, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.3, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(0.5, c = 'white', linewidth = 0.4)
            ax[i][j].axvline(1, c = 'white', linewidth = 0.4)
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

plt.tick_params(axis = 'both', which = 'both', direction='in', size = 10, labelsize=15)
plt.tick_params(axis = 'both', which = 'major',  size = 10)
plt.tick_params(axis = 'x', which = 'minor',  size = 7)
plt.suptitle('Radiation energy density without fluff', fontsize = 30)
fig.delaxes(ax[1][1])
plt.tight_layout()

if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/coloredRadE_nofluff_{xaxis}{weight}.png')
plt.show()

#%%
if xaxis == 'radii':
    from matplotlib.colors import ListedColormap, BoundaryNorm
    # Check wh has the higher energy between Low and High
    colorsthree = ['orchid', 'white', 'dodgerblue']  # Example: red for negative, white for zero, green for positive
    cmapthree = ListedColormap(colorsthree)
    boundaries = [-1e9, -0.1, 0.1, 1e9]  # Slight offset to differentiate zero from positive
    normthree = BoundaryNorm(boundaries, cmapthree.N, clip=True)

    fig, ax = plt.subplots(2,2, figsize = (12,12))
    img = ax[0][0].pcolormesh(radiires2/apo, tfb_res2, np.abs(col_orb_enres1) - np.abs(col_orb_enres2), cmap = cmapthree, norm = normthree)
    ax[0][0].set_title('Specific (absolute) orbital energy', fontsize = 20)
    img1 = ax[0][1].pcolormesh(radiires2/apo, tfb_res2, diff_ie, cmap = cmapthree, norm = normthree)
    ax[0][1].set_title('Specific internal energy', fontsize = 20)
    img2 = ax[1][0].pcolormesh(radiires2/apo, tfb_res2, diff_Rad, cmap = cmapthree, norm = normthree)
    ax[1][0].set_title('Radiation energy density', fontsize = 20)
    img3 = ax[1][1].pcolormesh(radiires2/apo, tfb_res2, diff_Rad_nofluff, cmap = cmapthree, norm = normthree)
    ax[1][1].set_title('Radiation energy density NO fluff', fontsize = 20)
    # Create an axis for the colorbar
    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # Add the colorbar to the figure
    cb = fig.colorbar(img, cax=cbar_ax)
    cb.set_label(f'{res1}-{res2}', fontsize = 30)
    ax[1][0].set_xlabel(r'$R/R_a$', fontsize = 20)
    ax[1][1].set_xlabel(r'$R/R_a$', fontsize = 20)
    ax[0][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
    ax[1][0].set_ylabel(r't/t$_{fb}$', fontsize = 20)
    for i in range(2):
        for j in range(2):
            ax[i][j].set_xscale('log')
    plt.tight_layout()
 
    # Lines
    # indices = [70, 100, 136]
    indices = [np.argmin(np.abs(tfb_res1-0.46)), np.argmin(np.abs(tfb_res1-0.66)), np.argmin(np.abs(tfb_res1-0.86))]
    colors_indices = ['navy', 'royalblue', 'deepskyblue']
    lines_difference = (col_Radres1[indices]-col_Radres2[indices])/col_Radres2[indices]
    img, ax = plt.subplots(1,2, figsize = (16,4))
    for i,idx in enumerate(indices):
        ax[0].plot(radiires1, col_Radres1[idx], c = colors_indices[i], label = f'{res1} t/tfb = {np.round(tfb_res1[idx],2)}')
        ax[0].plot(radiires2, col_Radres2[idx], '--', c = colors_indices[i], label = f'{res2} t/tfb = {np.round(tfb_res2[idx],2)}')
        ax[1].plot(radiires1, lines_difference[i], c = colors_indices[i], label = f't/tfb = {np.round(tfb_res1[idx],2)}')
    ax[0].set_xlabel(r'R/R$_a$', fontsize = 20)
    ax[0].set_ylabel(r'(Rad/Vol) [erg/cm$^3$]', fontsize = 20)
    ax[1].set_xlabel(r'$R/R_a$', fontsize = 20)
    ax[1].set_ylabel(r'$|$'+f'{res1}-{res2}'+r'$|$/mean', fontsize = 20)
    ax[0].loglog()
    ax[1].loglog()
    ax[0].legend(fontsize = 16)
    ax[1].legend(fontsize = 18)
    ax[0].grid()
    ax[1].grid()
    plt.suptitle(f'Relative differences: 1-{res2}/{res1}')
    plt.tight_layout()
    plt.show()

    #%%
    Lum_cgs = col_Radres1  * prel.c * 4 * np.pi * (radiires1*prel.Rsol_to_cm)**2 
    Lumres2_cgs = col_Radres2 * prel.c * 4 * np.pi * (radiires2*prel.Rsol_to_cm)**2 
    denom = (Lum_cgs + Lumres2_cgs)/2
    Lum_difference = np.abs(Lum_cgs[indices]-Lumres2_cgs[indices])/denom[indices]

    img, ax = plt.subplots(1,2, figsize = (20,7))
    for i,idx in enumerate(indices):
        if i == 0:
            ax[0].plot(radiires1, Lum_cgs[idx], c = colors_indices[i], label = f'{res1} res')#t/tfb = {np.round(tfb_Low[idx],2)}')
            ax[0].plot(radiires2, Lumres2_cgs[idx], '--', c = colors_indices[i], label = f'{res2} res')#t/tfb = {np.round(tfb_res2[idx],2)}')
            ax[1].plot(radiires1, Lum_difference[i], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
        else:   
            ax[0].plot(radiires1, Lum_cgs[idx], c = colors_indices[i])#, label = f'Low t/tfb = {np.round(tfb_Low[idx],2)}')
            ax[0].plot(radiires2, Lumres2_cgs[idx], '--', c = colors_indices[i])#, label = f'res2 t/tfb = {np.round(tfb_res2[idx],2)}')
            ax[1].plot(radiires1, Lum_difference[i], c = colors_indices[i])#, label = f't/tfb = {np.round(tfb_Low[idx],2)}')
    ax[0].set_ylim(1e40, 5e45)
    # ax[1].set_ylim(0.3, 1.8)
    ax[0].text(15, 1e43, r'$t/t_{fb}$ = '+ f'{np.round(tfb_res1[indices[0]],2)}', fontsize = 20)
    ax[0].text(20, 1e44, r'$t/t_{fb}$ = '+ f'{np.round(tfb_res1[indices[1]],2)}', fontsize = 20)
    ax[0].text(20, 1e45, r'$t/t_{fb}$ = '+ f'{np.round(tfb_res1[indices[2]],2)}', fontsize = 20)
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
    plt.subplots_adjust(wspace=1)
    # plt.suptitle('Relative differences: $|$Low-High$|$/mean')
    plt.tight_layout()
    if save:
        plt.savefig(f'/Users/paolamartire/shocks/Figs/{folder}/multiple/Luminosity{weight}.png')
    plt.show()

    for i in range(3):
        mean_error =  np.round(np.mean(Lum_difference[i][-10:-1]),2)
        print(f'Mean relative error for t/tfb = {tfb_res1[indices[i]]} is {mean_error}')
    # print the difference of the last point of Hires lum line
    for i,idx in enumerate(indices):
        if i ==2:
            continue
        after = Lumres2_cgs[indices[i+1]]
        before = Lumres2_cgs[indices[i]]
        mean_lastafter = np.mean(after[-10:-1])
        mean_lastbefore = np.mean(before[-10:-1])
        print(tfb_res1[indices[i+1]], tfb_res1[indices[i]])
        print(f'The relative difference of the last point of the high res line is {np.round((mean_lastafter-mean_lastbefore),2)}')

# %%
