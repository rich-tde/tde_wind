#%%
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import Utilities.prelude
from Utilities.basic_units import radians
import matplotlib.pyplot as plt

import src.analytical_Rsi as aRsi 

##
# CONSTANTS
## 
G = 1
G_SI = 6.6743e-11
Msol = 2e30 #1.98847e30 # kg
Rsol = 7e8 #6.957e8 # m
t = np.sqrt(Rsol**3 / (Msol*G_SI ))
c = 3e8 / (7e8/t)

##
# FUNCTIONS
##

def dRsi_dRp(Mbh, mstar, Rstar, beta):
    Rp = aRsi.pericentre(Mbh, mstar, Rstar, beta)
    a = aRsi.semimajor_axis(Mbh, mstar, Rstar)
    phi, Rsi = aRsi.precession_analyt(Mbh, mstar, Rstar, beta)
    e = aRsi.eccentricity(Mbh, mstar, Rstar, beta)
    first_term = 2*e / (1-e * np.cos(phi/2))
    second_term = -(Rp * (1+e) *  np.cos(phi/2)) / (a * (1 - e * np.cos(phi/2))**2)
    third_term = (phi * e**2 * np.sin(phi/2)) / (1 - e * np.cos(phi/2))**2
    deriv = first_term + second_term + third_term
    return deriv, Rsi, [first_term, second_term, third_term]

def DRsi(Mbh, mstar, Rstar, beta, delta):
    dRsi_Rp, Rsi, terms = dRsi_dRp(Mbh, mstar, Rstar, beta) 
    DeltaRsi = np.multiply(dRsi_Rp, delta)
    return DeltaRsi, Rsi

def dRsi_Nick(Mbh, mstar, Rstar, beta, delta):
    ecc = aRsi.eccentricity(Mbh, mstar, Rstar, beta)
    a = aRsi.semimajor_axis(Mbh, mstar, Rstar)
    cL = c
    # phi_mine, _ = aRsi.precession_analyt(Mbh, mstar, Rstar, beta)
    # phi_half_mine = phi_mine/2
    phi_half = 3*G*Mbh*np.pi/(a*cL**2*(1 - ecc**2))
    first = (2 * ecc)/(1 - ecc* np.cos(phi_half))
    second_outside = a*(1-ecc**2)/(1-ecc*np.cos(phi_half))**2
    second_inside = np.cos(phi_half)/a - \
        (6*G*Mbh*np.pi*ecc**2 * np.sin(phi_half))/(a*cL*((1 - ecc**2)))**2
    second = second_outside * second_inside
    dRsi_Rp = first - second
    DeltaRsi = np.multiply(dRsi_Rp, delta)
    return DeltaRsi

##
# MAIN
##
save = False

mstar = 1
Rstar = 1
# Plot agains mass
m = np.arange(4,8)
Mbh = np.power(10, m)
colors = ['coral', 'purple', 'deepskyblue', 'green', 'red']
m_more = np.arange(3,7.1,.5)
Mbh_more = np.power(10, m_more)
colors_more = ['r','coral', 'orange', 'sandybrown', 'purple', 'orchid', 'deepskyblue', 'b', 'green']
m_many = np.arange(5,7.1,.1)
Mbh_many = np.power(10, m_many)
beta_oneBH = np.array([1, 2, 4, 8, 16])
colors_beta = ['k', 'b', 'yellowgreen', 'orange', 'magenta']
beta_many = np.arange(1,10.1,.1)

#%%
for i in range(len(m)):
    d_Rself, Rsi = DRsi(Mbh[i], mstar, Rstar, beta_many, delta = Rstar)
    Rp = aRsi.pericentre(Mbh[i], mstar, Rstar, beta_many)
    Rg = aRsi.Rg_analyt(Mbh[i])
    plt.plot(beta_many, d_Rself/Rsi, c = colors[i], label = f'$10^{m[i]} M_\odot$')
plt.xlabel(r'$\beta$', fontsize = 18)
plt.grid()
plt.xlim(0.5,4)
plt.ylim(-0.05,0.05)
plt.ylabel(r'$\Delta R_{SI}/R_{SI}$', fontsize = 18)
plt.legend(fontsize = 18)
plt.title(f'$M_\star = {mstar}, R_\star = {Rstar}$', fontsize = 18)
if save:
    plt.savefig('/Users/paolamartire/shocks/Figs/DeltaRsi_beta.png')
plt.show()

#%%
# understand what is the behaviour 
for i in range(len(m_more)):
    d_Rself, Rsi = DRsi(Mbh_more[i], mstar, Rstar, beta_many, delta = Rstar)
    Rp = aRsi.pericentre(Mbh_more[i], mstar, Rstar, beta_many)
    Rg = aRsi.Rg_analyt(Mbh_more[i])
    plt.plot(beta_many, d_Rself/Rsi, c = colors_more[i], label = f'm = {m_more[i]}')
plt.xlabel(r'$\beta$', fontsize = 18)
plt.grid()
plt.ylabel(r'$\Delta R_{SI}/R_{SI}$', fontsize = 18)
plt.legend(fontsize = 10)
plt.title(r'BH masses: $10^m$, ' + f'$M_\star = {mstar}, R_\star = {Rstar}$', fontsize = 18)
if save:
    plt.savefig('/Users/paolamartire/shocks/Figs/DeltaRsi_moreM.png')
plt.show()

#%% Plot each contribution
fig, ax = plt.subplots(5,2, figsize = (16,6))
colors_term = ['orchid', 'lightskyblue', 'salmon']
for i in range(len(m_more)):
    d_Rself, Rsi, terms = dRsi_dRp(Mbh_more[i], mstar, Rstar, beta_many)
    first_term, second_term, third_term = terms[0], terms[1], terms[2]
    if i%2==0:
        col = 0
    else:
        col = 1
    r = int(i/2)
    ax[r][col].plot(beta_many, first_term, c = colors_term[0], label = 'first term')
    ax[r][col].plot(beta_many, second_term, c = colors_term[1], label = 'second term')
    ax[r][col].plot(beta_many, third_term, c = colors_term[2], label = 'third term')
    ax[r][col].set_title(f'm = {m_more[i]}')
    ax[r][col].set_ylim(-120,210)
ax[4][1].axis('off')
ax[4][0].set_xlabel(r'$\beta$', fontsize = 18)
ax[3][1].set_xlabel(r'$\beta$', fontsize = 18)
ax[4][0].legend()
plt.suptitle(r'Contribution to $\Delta R_{SI}$ for BH mass = $10^m M_\odot$' + f'$M_\star = {mstar}, R_\star = {Rstar}$', fontsize = 18)
plt.tight_layout()
if save:
    plt.savefig(f'/Users/paolamartire/shocks/Figs/DeltaRsi_terms_R{Rstar}M{mstar}.png')
plt.show()

#%%
# Nick's plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20,8))
# With respect to Rp/Rg
beta_many_Nick = np.arange(.2,20.1,.1)
for i in range(len(m)):
    if m[i]<5:
        continue
    d_Rself, Rsi = DRsi(Mbh[i], mstar, Rstar, beta_many_Nick, delta = Rstar)
    Rp = aRsi.pericentre(Mbh[i], mstar, Rstar, beta_many_Nick)
    Rg = aRsi.Rg_analyt(Mbh[i])
    ax1.plot(Rp/Rg, d_Rself/Rsi, c = colors[i], label = f'$10^{m[i]} M_\odot$')
ax1.grid()
ax1.set_xlim(9,81)
ax1.set_ylim(2e-4, 2)
ax1.set_xlabel(r'$R_p/R_g$', fontsize = 18)
ax1.set_ylabel(r'$\Delta R_{SI}/R_{SI}$', fontsize = 18)
ax1.set_yscale('log')
ax1.text(40, 1e-3, r'$\beta$ '+'from 0.2 (right)\nto 20 (left)', fontsize = 16)
ax1.legend(fontsize = 16)
# With respect to Mbh
beta_Nick = np.array([1,2,4])
color_Nick = ['purple', 'deepskyblue', 'green']
for i in range(len(beta_Nick)):
    d_Rself, Rsi = DRsi(Mbh_many, mstar, Rstar, beta_Nick[i], delta = Rstar)
    ax2.plot(Mbh_many, d_Rself/Rsi, c = colors_beta[i], label = r'$\beta$' + f' = {beta_Nick[i]}')
for i in range(len(beta_Nick)):
    dRsiNick_onRsi = np.zeros(len(Mbh_many))
    for k in range(len(Mbh_many)):
        der, Rsi = DRsi(Mbh_many[k], mstar, Rstar, beta_Nick[i], delta = Rstar)
        dr = dRsi_Nick(Mbh_many[k], mstar, Rstar, beta_Nick[i], delta = Rstar)
        dRsiNick_onRsi[k] = dr/Rsi
    ax2.plot(Mbh_many, dRsiNick_onRsi, c = color_Nick[i], linestyle = '--', label = r'$\beta$' + f' = {beta_Nick[i]} Nick')
ax2.text(2e5,1e-4, 'The differenece is due to \nthe definition of $\Phi$', fontsize = 16)
ax2.set_xlabel(r'$M_{BH}$', fontsize = 18)
ax2.set_ylabel(r'$\Delta R_{SI}/R_{SI}$', fontsize = 18)
ax2.loglog()
ax2.grid()
ax2.legend(fontsize = 12)
plt.suptitle(f'$M_\star = {mstar}, R_\star = {Rstar}$', fontsize = 18)
if save:
    plt.savefig('/Users/paolamartire/shocks/Figs/DeltaRsi_Nick.png')
plt.show()

# %%
interesteDelta, Rsi = DRsi(1e4, 0.5, 0.47, 1, 0.47)
print(f'expected Delta Rsi: {interesteDelta}, Rsi: {Rsi}')
# %%
