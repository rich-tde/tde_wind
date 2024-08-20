#%%
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import Utilities.prelude
from Utilities.basic_units import radians
import matplotlib.pyplot as plt

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

def tidal_radius(Mbh, mstar, Rstar):
    Rt = Rstar * (Mbh/mstar)**(1/3)
    return Rt

def pericentre(Mbh, mstar, Rstar, beta):
    Rt = tidal_radius(Mbh, mstar, Rstar)
    Rp = Rt/beta
    return Rp

def semimajor_axis(Mbh, mstar, Rstar):
    # a = GM/2E with E=GMR_star / Rt^2 (eq. 21 Rossi+20)
    Rt = tidal_radius(Mbh, mstar, Rstar)
    a = Rt**2 / (2*Rstar)
    return a

def apocentre(Mbh, mstar, Rstar, beta):
    Rp = pericentre(Mbh, mstar, Rstar, beta)
    a = semimajor_axis(Mbh, mstar, Rstar)
    Ra = 2*a - Rp
    return Ra

def eccentricity(Mbh, mstar, Rstar, beta):
    a = semimajor_axis(Mbh, mstar, Rstar)
    Rp = pericentre(Mbh, mstar, Rstar, beta)
    e = 1 - Rp/a
    return e

def Rg_analyt(Mbh):
    """ Gravitational radius of the black hole."""
    Rg = 2e-6 * Mbh
    return Rg

def precession_analyt(Mbh, mstar, Rstar, beta):
    """ Precession angle and self intersection redius as derived in Dai15 Eqs. 1 and 7.
    Insputs are in solar units."""
    Rp = pericentre(Mbh, mstar, Rstar, beta)
    Rt = tidal_radius(Mbh, mstar, Rstar)
    Rg = Rg_analyt(Mbh)
    a = semimajor_axis(Mbh, mstar, Rstar)
    e = eccentricity(Mbh, mstar, Rstar, beta)
    phi = 6 * np.pi/(1+e) * beta * Rg / Rt 
    Rsi = a * (1-e**2) / (1 - e * np.cos(phi/2)) 
    return phi, Rsi

def phi_Rsi(Mbh, mstar, Rstar, beta, Rsi):
    Rp = pericentre(Mbh, mstar, Rstar, beta)
    a = semimajor_axis(Mbh, mstar, Rstar)
    phi = 2 * np.arccos( (a/(a-Rp) * (1 - Rp/Rsi * (2*a-Rp)/a)))    
    return phi

if __name__ == '__main__':

    save = False

    mstar = 1
    Rstar = 1
    # Plot agains mass
    m = np.arange(4,8)
    Mbh = np.power(10, m)
    colors = ['coral', 'purple', 'deepskyblue', 'green']
    m_many = np.arange(5,7.1,.1)
    Mbh_many = np.power(10, m_many)
    beta_oneBH = np.array([1, 2, 4, 8, 16])
    colors_beta = ['k', 'b', 'yellowgreen', 'orange', 'magenta']
    beta_many = np.arange(1,10.1,.1)

    #%% Rsi(phi) test for one BH
    isel = 1
    mass_onebh = Mbh[isel]
    phi_oneBH, Rsi_oneBH = precession_analyt(mass_onebh, mstar, Rstar, beta_many)
    Ra_oneBH = apocentre(mass_onebh, mstar, Rstar, beta_many)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    ax1.scatter(phi_oneBH[0], Rsi_oneBH[0], c='k', marker = 's')
    ax1.scatter(phi_oneBH[-1], Rsi_oneBH[-1], c='k', marker = 'o')
    ax1.plot(phi_oneBH, Rsi_oneBH, c='k')
    ax1.set_xlabel(r'$\phi$ [rad]', fontsize = 18)
    ax1.set_ylabel(r'$R_{SI} [R_\odot]$', fontsize = 18)
    ax1.grid()
    ax2.scatter(phi_oneBH[0], Rsi_oneBH[0]/Ra_oneBH[0], c='k', marker = 's')
    ax2.scatter(phi_oneBH[-1], Rsi_oneBH[-1]/Ra_oneBH[-1], c='k', marker = 'o')
    ax2.plot(phi_oneBH, Rsi_oneBH/Ra_oneBH, c='k')
    ax2.set_xlabel(r'$\phi$ [rad]', fontsize = 18)
    ax2.set_ylabel(r'$R_{SI}/R_a$', fontsize = 18)
    ax2.text(0.05, 0.4, 'square = starting point \ndot = ending point', fontsize = 15)
    ax2.grid()
    plt.suptitle(f'$M_h = 10^{m[isel]} M_\odot, R_\star = {Rstar} R_\odot, M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$, ' + r'$\beta\in$ [' + f'{np.round(beta_many[0], 1)}, {np.round(beta_many[-1])}]')
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/Rsi_on_phi.png')
    plt.show()

    #%% Reproduce Dai15 Fig 2
    for i in range(len(beta_oneBH)):
        Rg = Rg_analyt(Mbh_many)
        _, Rsi = precession_analyt(Mbh_many, mstar, Rstar, beta_oneBH[i])
        RsiRg = Rsi/Rg
        plt.plot(Mbh_many, RsiRg, c = colors_beta[i], label = r'$\beta$ '+f'={beta_oneBH[i]}' )
    plt.xlabel(r'$M_{BH}$', fontsize = 18)
    plt.ylabel(r'$R_{SI}/R_g$', fontsize = 18)
    plt.title(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$', fontsize = 18)
    plt.loglog()
    plt.grid()
    plt.legend(fontsize = 18)
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/Rsi_on_Mbh.png')
    plt.show()

    #%% Nick's plot
    beta_Nick = np.arange(0.1,10.01,.01)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    for i in range(len(m)):
        # Nick's
        Rg = Rg_analyt(Mbh[i])
        if int(m[i])!=4:
            Rp = pericentre(Mbh[i], mstar, Rstar, beta_Nick)
            RpRg = Rp/Rg
            _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_Nick)
            RsiRg = Rsi/Rg
            ax1.scatter(RpRg[-1], RsiRg[-1], c = colors[i], marker = 'o')
            ax1.scatter(RpRg[0], RsiRg[0], c = colors[i], marker = 's')
            ax1.plot(RpRg, RsiRg, c = colors[i], label = f'$M = 10^{m[i]} M_\odot$')
        # more reasonable betas
        Rp = pericentre(Mbh[i], mstar, Rstar, beta_many)
        RpRg = Rp/Rg
        _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_many)
        RsiRg = Rsi/Rg
        ax2.scatter(RpRg[-1], RsiRg[-1], c = colors[i], marker = 'o')
        ax2.scatter(RpRg[0], RsiRg[0], c = colors[i], marker = 's')
        ax2.plot(RpRg, RsiRg, c = colors[i], label = f'$M = 10^{m[i]} M_\odot$')
    ax1.set_xlabel(r'$R_p/R_g$', fontsize = 18)
    ax2.set_xlabel(r'$R_p/R_g$', fontsize = 18)
    ax1.set_ylabel(r'$R_{SI}/R_g$', fontsize = 18)
    ax1.text(1.1, 200, r'$\beta\in$ [' + f'{np.round(beta_Nick[0], 1)}, {np.round(beta_Nick[-1])}]', fontsize = 18)
    ax2.text(1.1, 200, r'$\beta\in$ [' + f'{np.round(beta_many[0], 1)}, {np.round(beta_many[-1])}]', fontsize = 18)
    ax1.text(100, 20, 'square = starting point \ndot = ending point', fontsize = 15)
    ax1.grid()
    ax2.grid()
    ax1.loglog()
    ax2.loglog()
    ax2.legend(fontsize = 15)
    plt.suptitle(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$', fontsize = 18)
    plt.tight_layout()
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/Rsi_on_Rp.png')
    plt.show()

    #%% Nick's plot but with x=\beta
    beta_Nick = np.arange(0.1,10.01,.01)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    for i in range(len(m)):
        # Nick's
        Rg = Rg_analyt(Mbh[i])
        if int(m[i])!=4:
            _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_Nick)
            RsiRg = Rsi/Rg
            ax1.scatter(beta_Nick[-1], RsiRg[-1], c = colors[i], marker = 'o')
            ax1.scatter(beta_Nick[0], RsiRg[0], c = colors[i], marker = 's')
            ax1.plot(beta_Nick, RsiRg, c = colors[i], label = f'$M = 10^{m[i]} M_\odot$')
        # more reasonable betas
        _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_many)
        RsiRg = Rsi/Rg
        ax2.scatter(beta_many[-1], RsiRg[-1], c = colors[i], marker = 'o')
        ax2.scatter(beta_many[0], RsiRg[0], c = colors[i], marker = 's')
        ax2.plot(beta_many, RsiRg, c = colors[i], label = f'$M = 10^{m[i]} M_\odot$')
    ax1.set_xlabel(r'$\beta$', fontsize = 18)
    ax2.set_xlabel(r'$\beta$', fontsize = 18)
    ax1.set_ylabel(r'$R_{SI}/R_g$', fontsize = 18)
    ax1.text(.1, 40, r'$\beta\in$ [' + f'{np.round(beta_Nick[0], 1)}, {np.round(beta_Nick[-1])}]', fontsize = 18)
    ax2.text(1.1, 200, r'$\beta\in$ [' + f'{np.round(beta_many[0], 1)}, {np.round(beta_many[-1])}]', fontsize = 18)
    ax1.text(.1, 5, 'square = starting point \ndot = ending point', fontsize = 15)
    ax1.grid()
    ax2.grid()
    ax1.loglog()
    ax2.loglog()
    ax2.legend(fontsize = 15)
    plt.suptitle(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$', fontsize = 18)
    plt.tight_layout()
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/Rsi_on_beta.png')
    plt.show()
    #%% phi(Rsi) for fixed masses
    for i in range(len(m)):
        Ra = apocentre(Mbh[i], mstar, Rstar, beta_many)
        _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_many)
        phi = phi_Rsi(Mbh[i], mstar, Rstar, beta_many, Rsi)
        RsiRa = Rsi/Ra
        plt.scatter(RsiRa[0], phi[0], c = colors[i], marker = 's')
        plt.scatter(RsiRa[-1], phi[-1], c = colors[i], marker = 'o')
        plt.plot(RsiRa, phi, c = colors[i], label = f'$M = 10^{m[i]} M_\odot$' )
    plt.plot(Rsi_oneBH/Ra_oneBH, phi_oneBH, '--', c = 'orchid', label = f'10$^{m[isel]}$ from numerical inversion')
    plt.xlabel(r'$R_{SI}/R_a$', fontsize = 15)
    plt.ylabel(r'$\phi$ [rad]', fontsize = 15)
    plt.text(0.6, 1, 'square = starting point \ndot = ending point', fontsize = 15)
    plt.title(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$, ' + r'$\beta\in$ [' + f'{np.round(beta_many[0], 1)}, {np.round(beta_many[-1])}]', fontsize = 18)
    plt.grid()
    plt.legend(fontsize = 15)
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/phi_on_Rsi.png')
    plt.show()

    #%% zoom in 
    fig, ax = plt.subplots(int(len(m)/2), 2, figsize = (15, 5))
    for i in range(len(m)):
        if i%2==0:
            j=0
            ax[int(i/2)][j].set_ylabel(r'$\phi$ [rad]', fontsize = 15)
        else:
            j=1
        Ra = apocentre(Mbh[i], mstar, Rstar, beta_many)
        _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_many)
        phi = phi_Rsi(Mbh[i], mstar, Rstar, beta_many, Rsi)
        RsiRa = Rsi/Ra
        ax[int(i/2)][j].scatter(RsiRa[0], phi[0], c = colors[i], marker = 's')
        ax[int(i/2)][j].scatter(RsiRa[-1], phi[-1], c = colors[i], marker = 'o')
        ax[int(i/2)][j].plot(RsiRa, phi, c = colors[i], label = f'$M = 10^{m[i]} M_\odot$' )
        ax[int(i/2)][j].grid()
    ax[1][0].set_xlabel(r'$R_{SI}/R_a$', fontsize = 15)
    ax[1][1].set_xlabel(r'$R_{SI}/R_a$', fontsize = 15)
    plt.suptitle(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$, ' + r'$\beta\in$ [' + f'{np.round(beta_many[0], 1)}, {np.round(beta_many[-1])}]', fontsize = 18)
    plt.tight_layout()
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/phi_on_Rsi_zoomed.png')
    plt.show()

    #%% phi(Rsi) for fixed beta
    for i in range(len(beta_oneBH)):
        Ra = apocentre(Mbh_many, mstar, Rstar, beta_oneBH[i])
        _, Rsi = precession_analyt(Mbh_many, mstar, Rstar, beta_oneBH[i])
        phi = phi_Rsi(Mbh_many, mstar, Rstar, beta_oneBH[i], Rsi)
        RsiRa = Rsi/Ra
        plt.scatter(RsiRa[0], phi[0], c = colors_beta[i], marker = 's')
        plt.scatter(RsiRa[-1], phi[-1], c = colors_beta[i], marker = 'o')
        plt.plot(RsiRa, phi, c = colors_beta[i], label = r'$\beta$' + f'$= {int(beta_oneBH[i])}$' )
    plt.xlabel(r'$R_{SI}/R_a$', fontsize = 15)
    plt.ylabel(r'$\phi$ [rad]', fontsize = 15)
    plt.text(0.6, 1, 'square = starting point \ndot = ending point', fontsize = 15)
    plt.title(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$, ' + r'$M_{BH}=10^mM_\odot, m\in$ [' + f'{np.round(m_many[0], 1)}, {np.round(m_many[-1])}]', fontsize = 18)
    plt.grid()
    plt.legend(fontsize = 15)
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/phi_on_Rsi_beta.png')
    plt.show()

    #%% zoom in 
    fig, ax = plt.subplots(int(len(beta_oneBH)/2)+1, 2, figsize = (15, 5))
    for i in range(len(beta_oneBH)):
        if i%2==0:
            j=0
            ax[int(i/2)][j].set_ylabel(r'$\phi$ [rad]', fontsize = 15)
        else:
            j=1
        Ra = apocentre(Mbh_many, mstar, Rstar, beta_oneBH[i])
        _, Rsi = precession_analyt(Mbh_many, mstar, Rstar, beta_oneBH[i])
        phi = phi_Rsi(Mbh_many, mstar, Rstar, beta_oneBH[i], Rsi)
        RsiRa = Rsi/Ra
        ax[int(i/2)][j].scatter(RsiRa[0], phi[0], c = colors_beta[i], marker = 's')
        ax[int(i/2)][j].scatter(RsiRa[-1], phi[-1], c = colors_beta[i], marker = 'o')
        ax[int(i/2)][j].plot(RsiRa, phi, c = colors_beta[i], label = r'$\beta$' + f'= {int(beta_oneBH[i])}' )
        ax[int(i/2)][j].set_title(r'$\beta$ = ' + f'{int(beta_oneBH[i])}')
        ax[int(i/2)][j].grid()
    ax[1][0].set_xlabel(r'$R_{SI}/R_a$', fontsize = 15)
    ax[1][1].set_xlabel(r'$R_{SI}/R_a$', fontsize = 15)
    plt.suptitle(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$, ' + r'$M_{BH}=10^mM_\odot, m\in$ [' + f'{np.round(m_many[0], 1)}, {np.round(m_many[-1])}]', fontsize = 18)
    plt.tight_layout()
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/phi_on_Rsi_beta_zoomed.png')
    plt.show()


# %%
