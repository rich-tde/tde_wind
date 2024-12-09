abspath = '/Users/paolamartire/shocks/'
opac_path = f'{abspath}/src/Opacity'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import colorcet
import cmocean
from src.Opacity.linextrapolator import extrapolator_flipper, rich_extrapolator
import Utilities.prelude as prel

##
#%% FUNCTIONS
##
# Old functions from MATLAB Elad
def linearpad(D0,z0):
    factor = 100
    dz = z0[-1] - z0[-2]
    # print(np.shape(D0))
    dD = D0[:,-1] - D0[:,-2]
    
    z = [zi for zi in z0]
    z.append(z[-1] + factor*dz)
    z = np.array(z)
    #D = [di for di in D0]

    to_stack = np.add(D0[:,-1], factor*dD)
    to_stack = np.reshape(to_stack, (len(to_stack),1) )
    D = np.hstack((D0, to_stack))
    #D.append(to_stack)
    return np.array(D), z

def pad_interp(x,y,V):
    Vn, xn = linearpad(V, x)
    Vn, xn = linearpad(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = linearpad(Vn, y)
    Vn, yn = linearpad(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn
   
##
# MAIN
##
#%% Load data (they are the ln of the values)
T_cool = np.loadtxt(f'{opac_path}/T.txt') 
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt') 
rossland = np.loadtxt(f'{opac_path}/ross.txt') # Each row is a fixed T, column a fixed rho
T_plot = np.exp(T_cool)
Rho_plot = np.exp(Rho_cool)
exp_ross = np.exp(rossland)
#%% Check steps in T and rho from opacity table
# img, ax = plt.subplots(1,2, figsize = (10,5))
# ax[0].plot(np.diff(T_cool), 'o', color = 'forestgreen', markersize = 5)
# ax[1].plot(np.diff(Rho_cool), 'o', color = 'red', markersize = 5)
# ax[0].set_title('Step in T')
# ax[1].set_title('Step in rho')
# ax[0].grid()
# ax[1].grid()
# plt.suptitle('Steps in T and rho from the table')
# plt.tight_layout()
# plt.savefig(f'{abspath}/Figs/Test/step_Opacitytable.png')

# scattering opacity
scatt = 0.2*(1+0.7381) * Rho_plot #cm^2/g
# multiply column i of ross by Rho_plot[i] to get kappa
ross_rho = []
for i in range(len(T_plot)):
    ross_rho.append(exp_ross[i, :]/Rho_plot)
ross_rho = np.array(ross_rho)

# As we were doing all this time from Matlab (factor 100)
T_cool100, Rho_cool100, rossland100_t = pad_interp(T_cool, Rho_cool, rossland.T)
rossland100 = rossland100_t.T  #transpose back
T_plot100 = np.exp(T_cool100)
Rho_plot100 = np.exp(Rho_cool100)
exp_ross100 = np.exp(rossland100)
ross_rho100 = []
for i in range(len(T_plot100)):
    ross_rho100.append(exp_ross100[i, :]/Rho_plot100)
ross_rho100 = np.array(ross_rho100)

# K+P RICH 
T_coolflip, Rho_coolflip, rosslandflip = extrapolator_flipper(T_cool, Rho_cool, rossland, slope_length=5)
T_plotflip = np.exp(T_coolflip)
Rho_plotflip = np.exp(Rho_coolflip)
exp_rossflip = np.exp(rosslandflip)
ross_rhoflip = []
for i in range(len(T_plotflip)):
    ross_rhoflip.append(exp_rossflip[i, :]/Rho_plotflip)
ross_rhoflip = np.array(ross_rhoflip)

#%% RICH from me
T_RICH, Rho_RICH, rosslandRICH = rich_extrapolator(T_cool, Rho_cool, rossland)
T_plotRICH = np.exp(T_RICH)
Rho_plotRICH = np.exp(Rho_RICH)
exp_rossRICH = np.exp(rosslandRICH)
ross_rhoRICH = []
for i in range(len(T_plotRICH)):
    ross_rhoRICH.append(exp_rossRICH[i, :]/Rho_plotRICH)
ross_rhoRICH = np.array(ross_rhoRICH)

#%% Test to understand colormesh
# x = np.arange(100)
# y = np.arange(80)
# Z = np.random.rand(80, 100) # 80 rows, 100 columns
# plt.pcolormesh(x, y, Z) # x correspond to z columns, y to z rows
# you expect: opacity to increase with density, decrease with temperature
#%%
chosenTs = [1e4, 1e5, 1e7]
print('min T:' , np.min(T_plot), 'max T:', np.max(T_plot))
fig, ax = plt.subplots(1,3, figsize = (15,5))
for i,chosenT in enumerate(chosenTs):
    iT = np.argmin(np.abs(T_plot - chosenT))
    # iT_2 = np.argmin(np.abs(T_plot100 - chosenT))
    iT_3 = np.argmin(np.abs(T_plotflip - chosenT))
    iT_4 = np.argmin(np.abs(T_plotRICH - chosenT))
    # ax[i].plot(Rho_plot100, ross_rho100[iT_2, :], label = '100 extrap')
    ax[i].plot(Rho_plotflip, ross_rhoflip[iT_3, :], '-.', label = 'RICH extrap')
    ax[i].plot(Rho_plotRICH, ross_rhoRICH[iT_4, :], ':', label = 'double Extrapolation')
    ax[i].plot(Rho_plot, ross_rho[iT, :], '--', label = 'original')
    ax[i].plot(Rho_plot, scatt/Rho_plot,  color = 'r', linestyle = '--', label = 'scattering')
    ax[i].loglog()
    ax[i].set_ylim(5e-2, 1e4)
    ax[i].set_xlim(1e-18,1e6)
    ax[i].set_xlabel(r'$\rho$')
    ax[i].set_title(f'T = {chosenT} K')
    ax[i].legend()
ax[0].set_ylabel(r'$\kappa [cm^2g^{-1}]$')
plt.tight_layout()

#%%
print(np.min(Rho_plot))
#%%
chosenRhos = [1e-9, 1e-14] # you want 1e-6, 1e-11 kg/m^3 (too far from Elad's table, u want plot it)
colors_plot = ['forestgreen', 'r']
lines = ['solid', 'dashed']
plt.figure(figsize = (10,5))
for i,chosenRho in enumerate(chosenRhos):
    irho_4 = np.argmin(np.abs(Rho_plotRICH - chosenRho))
    plt.plot(T_plotRICH, ross_rhoRICH[:, irho_4], linestyle = lines[i], c = colors_plot[i], label = r'$\rho$ = '+f'{chosenRho} g/cm3')
plt.xlabel(r'T')
plt.ylabel(r'$\kappa [cm^2/g]$')
plt.ylim(7e-3, 2e2) #the axis from 7e-4 to 2e1 m2/g
plt.xlim(1e1,1e7)
plt.loglog()
plt.legend()
plt.grid()
plt.tight_layout()

#%% CHECK mine and K+P RICH extrap
print(np.min(Rho_plot))
chosenRhos = [1e-5, 1e-9] # you want 1e-6, 1e-11 kg/m^3 (too far from Elad's table, u want plot it)
fig, ax = plt.subplots(1,2, figsize = (15,5))
for i,chosenRho in enumerate(chosenRhos):
    irho = np.argmin(np.abs(Rho_plot - chosenRho))
    irho_4 = np.argmin(np.abs(Rho_plotRICH - chosenRho))
    ax[i].plot(T_plotRICH, ross_rhoRICH[:, irho_4], ':', label = 'RICH')
    ax[i].plot(T_plot, ross_rho[:, irho], '--', label = 'original')
    ax[i].set_xlabel(r'T')
    ax[i].set_ylabel(r'$\kappa [cm^2/g]$')
    ax[i].set_ylim(7e-3, 2e2) #the axis from 7e-4 to 2e1 m2/g
    ax[i].set_xlim(1e1,1e7)
    ax[i].loglog()
    ax[i].set_title(r'$\rho$ = ' +  f'{chosenRho}' + r'$g/cm^3$')
    ax[i].legend()
    ax[i].grid()
    plt.tight_layout()

#%% Test if it's in CGS
# fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,5))
# img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), exp_ross.T, norm = LogNorm(vmin=1e-15, vmax=1e12), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
# cbar = plt.colorbar(img)
# cbar.set_label(r'$\kappa_E [1/cm]$')
# # cbar.set_label(r'$\kappa$')
# ax1.set_xlabel(r'$\log_{10} T$')
# ax1.set_ylabel(r'$\log_{10} \rho$')

# img = ax2.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), np.transpose(ross_rho), norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
# cbar = plt.colorbar(img)
# cbar.set_label(r'$\kappa [cm^2/g]$')
# ax1.set_xlabel(r'$\log_{10} T$')
# ax2.set_ylabel(r'$\log_{10} \rho$')

# plt.suptitle('Original, no conversion', fontsize = 20)
# plt.tight_layout()

# #%% CHECK if is CGS
# exp_ross_conv = exp_ross /prel.Rsol_cgs #convert to CGS?
# ross_rho_conv = ross_rho * prel.Rsol_cgs**2/prel.Msol_cgs
# Rho_conv = Rho_plot * prel.Msol_cgs/prel.Rsol_cgs**3

# fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,5))
# img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_conv), exp_ross_conv.T, norm = LogNorm(vmin=1e-19, vmax=1e-4), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
# cbar = plt.colorbar(img)
# cbar.set_label(r'$\kappa_E [1/cm]$')
# # cbar.set_label(r'$\kappa$')
# ax1.set_xlabel(r'$\log_{10} T$')
# ax1.set_ylabel(r'$\log_{10} \rho$')

# img = ax2.pcolormesh(np.log10(T_plot), np.log10(Rho_conv), ross_rho_conv.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
# cbar = plt.colorbar(img)
# cbar.set_label(r'$\kappa [cm^2/g]$')
# ax1.set_xlabel(r'$\log_{10} T$')
# ax2.set_ylabel(r'$\log_{10} \rho$')

# plt.suptitle('Original, hypothesis: table NOT in CGS. Not reasonable', fontsize = 20)
# plt.tight_layout() 
#%%
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (18,5))
img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), ross_rho.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
ax1.set_ylabel(r'$\log_{10} \rho$')
ax1.set_title('Table')

# img = ax2.pcolormesh(np.log10(T_plot100), np.log10(Rho_plot100), ross_rho100.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
# cbar = plt.colorbar(img)
# ax2.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
# ax2.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
# ax2.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
# ax2.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
# ax2.set_title('Old Extrapolation (factor 100)')

img = ax2.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), ross_rhoRICH.T,  norm = LogNorm(vmin = 1e-5, vmax=1e5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax2.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax2.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax2.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax2.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax2.set_title('RICH Extrapolation')

img = ax3.pcolormesh(np.log10(T_plotflip), np.log10(Rho_plotflip), ross_rhoflip.T,  norm = LogNorm(vmin = 1e-5, vmax=1e5), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax3.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax3.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax3.set_title('K+P Extrapolation')


for ax in [ax1, ax2, ax3]:
    # Get the existing ticks on the x-axis
    original_ticksx = ax.get_xticks()
    # Calculate midpoints between each pair of ticks
    if ax==ax1:
        midpointsx = (original_ticksx[:-1] + original_ticksx[1:]) / 2
    else:
        midpointsx = np.arange(original_ticksx[0], original_ticksx[-1])
    # Combine the original ticks and midpointsx
    new_ticksx = np.sort(np.concatenate((original_ticksx, midpointsx)))
    labelsx = [str(np.round(tick,2)) if tick in original_ticksx else "" for tick in new_ticksx]   
    ax.set_xticks(new_ticksx)
    ax.set_xticklabels(labelsx)

    original_ticks = ax.get_yticks()
    # Calculate midpoints between each pair of ticks
    if ax==ax1:
        midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
    else:
        midpoints = np.arange(original_ticks[0], original_ticks[-1], 2)
    # Combine the original ticks and midpoints
    new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
    labels = [str(np.round(tick,2)) if tick in original_ticks else "" for tick in new_ticks]   
    ax.set_yticks(new_ticks)
    ax.set_yticklabels(labels)

    ax.tick_params(axis='x', which='major', width=1.6, length=7, color = 'k')
    ax.tick_params(axis='y', which='major', width=1.6, length=7, color = 'k')
    ax.set_xlabel(r'$\log_{10} T$')
    if ax == ax1:
        ax.set_xlim(np.min(np.log10(T_plot)), np.max(np.log10(T_plot)))
        ax.set_ylim(np.min(np.log10(Rho_plot)), np.max(np.log10(Rho_plot)))
    else:
        ax.set_xlim(0.8,11)
        ax.set_ylim(-19,11)

plt.tight_layout()


# %%
# diff between 3 and 4
diff = np.zeros((len(T_plotRICH), len(Rho_plotRICH)))
for i in range(len(T_plotRICH)):
    for j in range(len(Rho_plotRICH)):
        diff[i,j] = 2*np.abs(ross_rhoflip[i, j] - ross_rhoRICH[i,j])/(ross_rhoRICH[i,j]+ross_rhoflip[i, j])


#%%
# Load data for photosphere from P+K RICH code
from Utilities.operators import make_tree
wanted_snap = 164
alldataph = np.loadtxt(f'{abspath}/data/R0.47M0.5BH10000beta1S60n1.5Compton/_phidx.txt')
snaps, times, indices_ph = alldataph[:, 0], alldataph[:, 1], alldataph[:, 2:]
snap = np.array(snaps)
selected_idx = np.argmin(np.abs(snaps - wanted_snap))
snap = int(snaps[selected_idx])
idx_ph = indices_ph[selected_idx]
idx_ph = idx_ph.astype(int)
path = f'/Users/paolamartire/shocks/TDE/R0.47M0.5BH10000beta1S60n1.5Compton/{snap}'
data = make_tree(path, snap, energy = False)
dencut = data.Den > 1e-19
X, Y, Z, Temp, Den_cgs = data.X[dencut], data.Y[dencut], data.Z[dencut], data.Temp[dencut], data.Den[dencut]
Den = Den_cgs * prel.Msol_cgs/prel.Rsol_cgs**3
xph, yph, zph, Tph, Rhoph = X[idx_ph], Y[idx_ph], Z[idx_ph], Temp[idx_ph], Den[idx_ph]
#%%

fig, (ax3,ax4, ax1) = plt.subplots(1,3, figsize = (15,5))
img = ax3.pcolormesh(np.log10(T_plotflip), np.log10(Rho_plotflip), ross_rhoflip.T,  norm = LogNorm(vmin = 1e-12, vmax=1e10), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax3.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax3.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax3.set_title('K+P RICH Extrapolation')

img = ax4.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), ross_rhoRICH.T,  norm = LogNorm(vmin = 1e-12, vmax=1e10), cmap = 'jet', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax4.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax4.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax4.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax4.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax4.set_title('My new RICH Extrapolation')

img = ax1.pcolormesh(np.log10(T_plotRICH), np.log10(Rho_plotRICH), diff.T, cmap = 'jet')
ax1.scatter(np.log10(Tph), np.log10(Rhoph), color = 'k', s = 1)
cbar=plt.colorbar(img)
cbar.set_label(r'$\log_{10}\Delta_{rel}$')

for ax in [ax1, ax3, ax4]:
    ax.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
    ax.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
    ax.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
    ax.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
    ax.set_xlim(np.min(np.log10(T_plotRICH)), np.max(np.log10(T_plotRICH)))
    ax.set_ylim(np.min(np.log10(Rho_plotRICH)), np.max(np.log10(Rho_plotRICH)))
    ax.set_xlabel(r'$\log_{10} T$')
    ax.set_ylabel(r'$\log_{10} \rho$')

plt.tight_layout()
# %%

