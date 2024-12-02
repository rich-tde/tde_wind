abspath = '/Users/paolamartire/shocks/'
opac_path = f'{abspath}/src/Opacity'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import colorcet
import cmocean
from src.Opacity.linextrapolator import extrapolator_flipper
import Utilities.prelude as prel

##
# FUNCTIONS
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
    ross_rho.append(exp_ross[i, :] / Rho_plot)
ross_rho = np.array(ross_rho)

# As we were doing all this time from Matlab
T_cool2, Rho_cool2, rossland2_t = pad_interp(T_cool, Rho_cool, rossland.T)
rossland2 = rossland2_t.T  #transpose back
T_plot2 = np.exp(T_cool2)
Rho_plot2 = np.exp(Rho_cool2)
exp_ross2 = np.exp(rossland2)
ross_rho2 = []
for i in range(len(T_plot2)):
    ross_rho2.append(exp_ross2[i, :] / Rho_plot2)
ross_rho2 = np.array(ross_rho2)

# RICH from me and K
T_cool3, Rho_cool3, rossland3_t = extrapolator_flipper(T_cool, Rho_cool, rossland.T)
rossland3 = rossland3_t.T  #transpose back
T_plot3 = np.exp(T_cool3)
Rho_plot3 = np.exp(Rho_cool3)
exp_ross3 = np.exp(rossland3)
ross_rho3 = []
for i in range(len(T_plot3)):
    ross_rho3.append(exp_ross3[i, :] / Rho_plot3)
ross_rho3 = np.array(ross_rho3)
#%% Test to understand colormesh
# x = np.arange(100)
# y = np.arange(80)
# Z = np.random.rand(80, 100) # 80 rows, 100 columns
# plt.pcolormesh(x, y, Z) # x correspond to z columns, y to z rows
# you expect: opacity to increase with density, decrease with temperature
#%%
chosenTs = [1e4, 1e5, 1e7]
chosenRho = 1e-10

fig, ax = plt.subplots(1,3, figsize = (15,5))
for i,chosenT in enumerate(chosenTs):
    iT = np.argmin(np.abs(T_plot - chosenT))
    iT_2 = np.argmin(np.abs(T_plot2 - chosenT))
    iT_3 = np.argmin(np.abs(T_plot3 - chosenT))
    ax[i].plot(Rho_plot2, ross_rho2[iT_2, :], label = '100 extrap')
    ax[i].plot(Rho_plot3, ross_rho3[iT_3, :], '-.', label = 'RICH extrap')
    ax[i].plot(Rho_plot, ross_rho[iT, :], '--', label = 'original')
    # ax1.plot(Rho_plot, exp_abs[iT, :],  label = 'absorption')
    ax[i].plot(Rho_plot, scatt/Rho_plot,  color = 'r', linestyle = '--', label = 'scattering')
    ax[i].loglog()
    ax[i].set_ylim(5e-2, 1e4)
    ax[i].set_xlim(1e-18,1e6)
    ax[i].set_xlabel(r'$\rho$')
    ax[i].set_title(f'T = {chosenT} K')
    ax[i].legend()
ax[0].set_ylabel(r'$\kappa [cm^2g^{-1}]$')

# irho = np.argmin(np.abs(Rho_plot - chosenRho))
# irho_2 = np.argmin(np.abs(Rho_plot2 - chosenRho))
# irho_3 = np.argmin(np.abs(Rho_plot3 - chosenRho))
# ax2.plot(T_plot2, ross_rho2[:, irho_2],  label = '100 extrap')
# ax2.plot(T_plot3, ross_rho3[:, irho_3], '-.', label = 'RICH extrap')
# ax2.plot(T_plot, ross_rho[:, irho], '--', label = 'original')
# ax2.set_xlabel(r'T')
# ax2.set_ylabel(r'$\kappa [cm^2/g]$')
# ax2.set_ylim(1e-4, 1e4)
# ax2.set_xlim(1e3,1e9)
# ax2.loglog()
# ax2.set_title(f'den = {chosenRho}')
# ax2.legend()
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
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,5))
img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), ross_rho.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
ax1.set_xlabel(r'$\log_{10} T$')
ax1.set_ylabel(r'$\log_{10} \rho$')
ax1.set_title('Original')

img = ax2.pcolormesh(np.log10(T_plot2), np.log10(Rho_plot2), ross_rho2.T, norm = LogNorm(vmin=1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
ax2.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax2.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax2.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax2.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax2.set_xlabel(r'$\log_{10} T$')
# ax2.set_ylabel(r'$\log_{10} \rho$')
ax2.set_title('Old Extrapolation (factor 100)')

img = ax3.pcolormesh(np.log10(T_plot3), np.log10(Rho_plot3), ross_rho3.T,  norm = LogNorm(vmin = 1e-5, vmax=1e5), cmap = 'jet') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa [cm^2/g]$')
ax3.axvline(np.log10(np.min(T_plot)), color = 'k', linestyle = '--')
ax3.axvline(np.log10(np.max(T_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.min(Rho_plot)), color = 'k', linestyle = '--')
ax3.axhline(np.log10(np.max(Rho_plot)), color = 'k', linestyle = '--')
ax3.set_xlabel(r'$\log_{10} T$')
# ax3.set_ylabel(r'$\log_{10} \rho$')
ax3.set_title('RICH Extrapolation')

plt.tight_layout()


# %%
