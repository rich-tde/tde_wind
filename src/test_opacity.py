abspath = '/Users/paolamartire/shocks/'
opac_path = f'{abspath}/src/Opacity'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import colorcet
import cmocean
from src.Opacity.linextrapolator import extrapolator_flipper, double_extrapolator
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
T_cool = np.loadtxt(f'{opac_path}/T.txt') #they are ln(T)
Rho_cool = np.loadtxt(f'{opac_path}/rho.txt') #they are ln(rho)
rossland = np.loadtxt(f'{opac_path}/ross.txt') # they are ln(K). each row is a fixed T, column a fixed rho
T_plot = np.exp(T_cool)
Rho_plot = np.exp(Rho_cool)
exp_ross = np.exp(rossland)
# scattering opacity
scatt = 0.2*(1+0.7381) * Rho_plot #cm^2/g
# multiply column i of ross by Rho_plot[i] to get kappa
ross_rho = []
for i in range(len(T_plot)):
    ross_rho.append(exp_ross[i, :]/Rho_plot)
ross_rho = np.array(ross_rho)

# As we were doing all this time from Matlab
T_cool2, Rho_cool2, rossland2_t = pad_interp(T_cool, Rho_cool, rossland.T)
rossland2 = rossland2_t.T  #transpose back
T_plot2 = np.exp(T_cool2)
Rho_plot2 = np.exp(Rho_cool2)
exp_ross2 = np.exp(rossland2)
ross_rho2 = []
for i in range(len(T_plot2)):
    ross_rho2.append(exp_ross2[i, :]/Rho_plot2)
ross_rho2 = np.array(ross_rho2)

# RICH from me and K
T_cool3, Rho_cool3, rossland3 = extrapolator_flipper(T_cool, Rho_cool, rossland, slope_length=5)
T_plot3 = np.exp(T_cool3)
Rho_plot3 = np.exp(Rho_cool3)
exp_ross3 = np.exp(rossland3)
ross_rho3 = []
for i in range(len(T_plot3)):
    ross_rho3.append(exp_ross3[i, :]/Rho_plot3)
ross_rho3 = np.array(ross_rho3)

#%% RICH from me
T_cool4, Rho_cool4, rossland4 = double_extrapolator(T_cool, Rho_cool, rossland, slope_length=5)
T_plot4 = np.exp(T_cool4)
Rho_plot4 = np.exp(Rho_cool4)
exp_ross4 = np.exp(rossland4)
ross_rho4 = []
for i in range(len(T_plot4)):
    ross_rho4.append(exp_ross4[i, :]/Rho_plot4)
ross_rho4 = np.array(ross_rho4)

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
    iT_2 = np.argmin(np.abs(T_plot2 - chosenT))
    iT_3 = np.argmin(np.abs(T_plot3 - chosenT))
    iT_4 = np.argmin(np.abs(T_plot4 - chosenT))
    ax[i].plot(Rho_plot2, ross_rho2[iT_2, :], label = '100 extrap')
    ax[i].plot(Rho_plot3, ross_rho3[iT_3, :], '-.', label = 'RICH extrap')
    ax[i].plot(Rho_plot4, ross_rho4[iT_4, :], ':', label = 'double Extrapolation')
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
# print(np.min(Rho_plot))
chosenRhos = [1e-10, 1e-9] 
fig, ax = plt.subplots(1,2, figsize = (10,5))
for i,chosenRho in enumerate(chosenRhos):
    irho = np.argmin(np.abs(Rho_plot - chosenRho))
    irho_2 = np.argmin(np.abs(Rho_plot2 - chosenRho))
    irho_3 = np.argmin(np.abs(Rho_plot3 - chosenRho))
    irho_4 = np.argmin(np.abs(Rho_plot4 - chosenRho))
    ax[i].plot(T_plot2, ross_rho2[:, irho_2],  label = '100 extrap')
    ax[i].plot(T_plot3, ross_rho3[:, irho_3], '-.', label = 'RICH extrap')
    ax[i].plot(T_plot4, ross_rho4[:, irho_4], ':', label = 'double Extrapolation')
    ax[i].plot(T_plot, ross_rho[:, irho], '--', label = 'original')
    ax[i].set_xlabel(r'T')
    ax[i].set_ylabel(r'$\kappa [cm^2/g]$')
    ax[i].set_ylim(7e-3, 2e2) #the axis from 7e-4 to 2e1 m2/g
    ax[i].set_xlim(1e1,1e9)
    ax[i].loglog()
    ax[i].set_title(r'$\rho$ = ' +  f'{chosenRho}' + r'$g/cm^3$')
    ax[i].legend()
    ax[i].grid()
    plt.tight_layout()

#%%
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,5))
img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), exp_ross.T, norm = LogNorm(vmin=1e-15, vmax=1e12), cmap = 'cet_rainbow4') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa_E [1/cm]$')
# cbar.set_label(r'$\kappa$')
ax1.set_xlabel(r'$\log_{10} T$')
ax1.set_ylabel(r'$\log_{10} \rho$')

img = ax2.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), np.transpose(ross_rho), norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax1.set_xlabel(r'$\log_{10} T$')
ax2.set_ylabel(r'$\log_{10} \rho$')

plt.suptitle('Original, no conversion', fontsize = 20)
plt.tight_layout()

#%% CHECK if is CGS
exp_ross_conv = exp_ross /prel.Rsol_cgs #convert to CGS?
ross_rho_conv = ross_rho * prel.Rsol_cgs**2/prel.Msol_cgs
Rho_conv = Rho_plot * prel.Msol_cgs/prel.Rsol_cgs**3

fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,5))
img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_conv), exp_ross_conv.T, norm = LogNorm(vmin=1e-19, vmax=1e-4), cmap = 'cet_rainbow4') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa_E [1/cm]$')
# cbar.set_label(r'$\kappa$')
ax1.set_xlabel(r'$\log_{10} T$')
ax1.set_ylabel(r'$\log_{10} \rho$')

img = ax2.pcolormesh(np.log10(T_plot), np.log10(Rho_conv), ross_rho_conv.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax1.set_xlabel(r'$\log_{10} T$')
ax2.set_ylabel(r'$\log_{10} \rho$')

plt.suptitle('Original, hypothesis: table NOT in CGS. Not reasonable', fontsize = 20)
plt.tight_layout() 
#%%
fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4, figsize = (20,5))
img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), ross_rho.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'gist_rainbow', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
ax1.set_ylabel(r'$\log_{10} \rho$')
ax1.set_title('Original')

img = ax2.pcolormesh(np.log10(T_plot2), np.log10(Rho_plot2), ross_rho2.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'gist_rainbow', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
ax2.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax2.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax2.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax2.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax2.set_title('Old Extrapolation (factor 100)')

img = ax3.pcolormesh(np.log10(T_plot3), np.log10(Rho_plot3), ross_rho3.T,  norm = LogNorm(vmin = 1e-5, vmax=1e5), cmap = 'gist_rainbow', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax3.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax3.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax3.set_title('K+P RICH Extrapolation')

img = ax4.pcolormesh(np.log10(T_plot4), np.log10(Rho_plot4), ross_rho4.T,  norm = LogNorm(vmin = 1e-5, vmax=1e5), cmap = 'gist_rainbow', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax4.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax4.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax4.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax4.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax4.set_title('My new RICH Extrapolation')

for ax in [ax1, ax2, ax3, ax4]:
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
diff = np.zeros((len(T_plot4), len(Rho_plot4)))
for i in range(len(T_plot4)):
    for j in range(len(Rho_plot4)):
        diff[i,j] = 2*np.abs(ross_rho3[i, j] - ross_rho4[i,j])/(ross_rho4[i,j]+ross_rho3[i, j])


#%%
fig, (ax3,ax4, ax1) = plt.subplots(1,3, figsize = (15,5))
img = ax3.pcolormesh(np.log10(T_plot3), np.log10(Rho_plot3), ross_rho3.T,  norm = LogNorm(vmin = 1e-12, vmax=1e10), cmap = 'gist_rainbow', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax3.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax3.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax3.set_title('K+P RICH Extrapolation')

img = ax4.pcolormesh(np.log10(T_plot4), np.log10(Rho_plot4), ross_rho4.T,  norm = LogNorm(vmin = 1e-12, vmax=1e10), cmap = 'gist_rainbow', alpha = 0.7) #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax4.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax4.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax4.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax4.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax4.set_title('My new RICH Extrapolation')

img = ax1.pcolormesh(np.log10(T_plot4), np.log10(Rho_plot4), diff.T, cmap = 'Oranges')
cbar=plt.colorbar(img)
cbar.set_label(r'$\log_{10}\Delta_{rel}$')

for ax in [ax1, ax3, ax4]:
    ax.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
    ax.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
    ax.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
    ax.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
    ax.set_xlim(np.min(np.log10(T_plot4)), np.max(np.log10(T_plot4)))
    ax.set_ylim(np.min(np.log10(Rho_plot4)), np.max(np.log10(Rho_plot4)))
    ax.set_xlabel(r'$\log_{10} T$')
    ax.set_ylabel(r'$\log_{10} \rho$')

plt.tight_layout()
# %%
