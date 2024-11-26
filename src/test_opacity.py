abspath = '/Users/paolamartire/shocks/'
opac_path = f'{abspath}/src/Opacity'
import sys
sys.path.append(abspath)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import colorcet

##
# FUNCTIONS
##
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

def pad_new_interp(x,y,V):
    Vn, xn = new_interp(V, x)
    Vn, xn = new_interp(np.fliplr(Vn), np.flip(xn))
    Vn = Vn.T
    Vn, yn = new_interp(Vn, y)
    Vn, yn = new_interp(np.fliplr(Vn), np.flip(yn))
    Vn = Vn.T
    return xn, yn, Vn

def new_interp(V, y, extrarows = 60):
    # Low extrapolation
    yslope_low = y[1] - y[0]
    y_extra_low = [y[0] - yslope_low * (i + 1) for i in range(extrarows)]
    
    # High extrapolation
    yslope_h = y[-1] - y[-2]
    y_extra_high = [y[-1] + yslope_h * (i + 1) for i in range(extrarows)]
    
    # Stack, reverse low to stack properly
    yn = np.concatenate([y_extra_low[::-1], y, y_extra_high])
    
    # 2D low
    Vslope_low = V[1, :] - V[0, :]
    Vextra_low = [V[0, :] - 10*Vslope_low * (i + 1) for i in range(extrarows)]
    
    # 2D high
    Vslope_high = V[-1, :] - V[-2, :]  # Linear difference
    Vextra_high = [V[-1, :] + Vslope_high * (i + 1) for i in range(extrarows)]

    Vn = np.vstack([Vextra_low[::-1], V, Vextra_high]) 

    return Vn, yn

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
scatt = 0.2*(1+0.7381) * Rho_plot

T_cool2, Rho_cool2, rossland2_t = pad_interp(T_cool, Rho_cool, rossland.T)
rossland2 = rossland2_t.T  #transpose back
T_plot2 = np.exp(T_cool2)
Rho_plot2 = np.exp(Rho_cool2)
exp_ross2 = np.exp(rossland2)

T_cool3, Rho_cool3, rossland3_t = pad_new_interp(T_cool, Rho_cool, rossland.T)
rossland3 = rossland3_t.T  #transpose back
T_plot3 = np.exp(T_cool3)
Rho_plot3 = np.exp(Rho_cool3)
exp_ross3 = np.exp(rossland3)

#%% Test to understand colormesh
# x = np.arange(100)
# y = np.arange(80)
# Z = np.random.rand(80, 100) # 80 rows, 100 columns
# plt.pcolormesh(x, y, Z) # x correspond to z columns, y to z rows
# you expect: opacity to increase with density, decrease with temperature
#%%
chosenT = 1e4
chosenRho = 1e-10

iT = np.argmin(np.abs(T_plot - chosenT))
iT_2 = np.argmin(np.abs(T_plot2 - chosenT))
iT_3 = np.argmin(np.abs(T_plot3 - chosenT))
fig, (ax1,ax2) = plt.subplots(1,2, figsize = (10,5))
ax1.plot(Rho_plot2, exp_ross2[iT_2, :], label = 'old extrap')
ax1.plot(Rho_plot3, exp_ross3[iT_3, :], '-.', label = 'new extrap')
ax1.plot(Rho_plot, exp_ross[iT, :], '--', label = 'original')
ax1.set_xlabel(r'$\rho$')
ax1.set_ylabel(r'$\kappa [cm^{-1}]$')
ax1.loglog()
ax1.set_ylim(1e-12, 1e8)
ax1.set_xlim(1e-19,1e6)
ax1.set_title(f'T = {chosenT} K')
ax1.legend()

irho = np.argmin(np.abs(Rho_plot - chosenRho))
irho_2 = np.argmin(np.abs(Rho_plot2 - chosenRho))
irho_3 = np.argmin(np.abs(Rho_plot3 - chosenRho))
ax2.plot(T_plot2, exp_ross2[:, irho_2],  label = 'old extrap')
ax2.plot(T_plot3, exp_ross3[:, irho_3], '-.', label = 'new extrap')
ax2.plot(T_plot, exp_ross[:, irho], '--', label = 'original')
ax2.axhline(scatt[irho],  color = 'r', linestyle = '--', label = 'scattering')
ax2.set_xlabel(r'T')
# ax2.set_ylabel(r'$\kappa [cm^{-1}]$')
ax2.set_ylim(1e-12, 1e-8)
ax2.set_xlim(1e3,1e9)
ax2.loglog()
ax2.set_title(f'den = {chosenRho}')
ax2.legend()

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (15,5))
img = ax1.pcolormesh(np.log10(T_plot), np.log10(Rho_plot), exp_ross.T, norm = LogNorm(vmin=1e-7, vmax=1.1e10), cmap = 'Grays') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa$')
ax1.set_xlabel(r'$\log_{10} T$')
ax1.set_ylabel(r'$\log_{10} \rho$')
ax1.set_title('Original')

img = ax2.pcolormesh(np.log10(T_plot2), np.log10(Rho_plot2), exp_ross2.T, norm = LogNorm(vmin=1e-7, vmax=1.1e10), cmap = 'Grays') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa$')
ax2.set_xlabel(r'$\log_{10} T$')
# ax2.set_ylabel(r'$\log_{10} \rho$')
ax2.set_title('Old Extrapolation')

img = ax3.pcolormesh(np.log10(T_plot3), np.log10(Rho_plot3), exp_ross3.T, norm = LogNorm(vmin=1e-7, vmax=1.1e10), cmap = 'Grays') #exp_ross.T have rows = fixed rho, columns = fixed T
cbar = plt.colorbar(img)
cbar.set_label(r'$\kappa$')
ax3.set_xlabel(r'$\log_{10} T$')
# ax3.set_ylabel(r'$\log_{10} \rho$')
ax3.set_title('New Extrapolation')

plt.tight_layout()