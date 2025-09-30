#%%
import sys
sys.path.append('/Users/paolamartire/shocks')
import numpy as np
import Utilities.prelude as prel
from Utilities.basic_units import radians
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import src.orbits as orb

##
# FUNCTIONS
##

def precession_analyt(Mbh, mstar, Rstar, beta):
    """ Precession angle and self intersection redius as derived in Dai15 Eqs. 1 and 7.
    Insputs are in solar units."""
    Rp = orb.pericentre(Rstar, mstar, Mbh, beta)
    Rt = orb.tidal_radius(Rstar, mstar, Mbh)
    Rg = orb.R_grav(Mbh, prel.csol_cgs, prel.G)
    a = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
    e = orb.eccentricity(Rstar, mstar, Mbh, beta)
    phi = 6 * np.pi * beta * Rg / (Rt * (1+e))
    Rsi = a * (1-e**2) / (1 - e * np.cos(phi/2)) 
    return phi, Rsi

def phi_Rsi(Mbh, mstar, Rstar, beta, Rsi):
    Rp = orb.pericentre(Rstar, mstar, Mbh, beta)
    a = orb.semimajor_axis(Rstar, mstar, Mbh, prel.G)
    phi = 2 * np.arccos( (a/(a-Rp) * (1 - Rp/Rsi * (2*a-Rp)/a)))    
    return phi

if __name__ == '__main__':
    save = True

    mstar = 0.5
    Rstar = 0.47
    # Plot agains mass
    m = np.arange(4,8)
    Mbh = np.power(10, m)
    colors_Mbh = ['coral', 'purple', 'deepskyblue', 'green']
    m_many = np.arange(4,7.1,.1)
    Mbh_many = np.power(10, m_many)
    beta_oneBH = np.array([1, 2, 4, 8])
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
    # print(precession_analyt(1e6, 1, 0.47,1))
    print(orb.apocentre(0.47, 0.5, 1e4, 1)/2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    for i in range(len(beta_oneBH)):
        Rg = orb.R_grav(Mbh_many, prel.csol_cgs, prel.G)
        apo = orb.apocentre(Rstar, mstar, Mbh_many, beta_oneBH[i])
        phi, Rsi = precession_analyt(Mbh_many, mstar, Rstar, beta_oneBH[i])
        phi_deg = phi * 180/np.pi
        ax1.plot(Mbh_many, Rsi/Rg, c = colors_beta[i], label = r'$\beta$ '+f'={beta_oneBH[i]}' )
        ax2.plot(Mbh_many, phi_deg, c = colors_beta[i], label = r'$\beta$ '+f'={beta_oneBH[i]}' )
    ax1.set_ylabel(r'$R_{SI}/R_g$', fontsize = 18)
    ax2.set_ylabel(r'$\phi$ [deg]', fontsize = 18)
    ax1.loglog()
    ax2.set_xscale('log')  
    ax2.set_ylim(0,40)
    plt.suptitle(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$', fontsize = 18)
    ax1.legend(fontsize = 18)
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'$M_{BH} [M_\odot]$', fontsize = 18)
        ax.grid()
    plt.tight_layout()
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/precession_toM.png')
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
    phi_max = np.zeros_like(m)
    Rsi_phi_max = np.zeros_like(m)
    for i in range(len(m)):
        Ra = apocentre(Mbh[i], mstar, Rstar, beta_many)
        _, Rsi = precession_analyt(Mbh[i], mstar, Rstar, beta_many)
        phi = phi_Rsi(Mbh[i], mstar, Rstar, beta_many, Rsi)
        phi_max[i] = phi[-1]
        Rsi_phi_max[i] = Rsi[-1]
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

    #%% max phi(Rsi) for fixed masses
    phi_max = np.array(phi_max)
    Rsi_phi_max = np.array(Rsi_phi_max)
    print(phi_max, Rsi_phi_max)
    plt.figure()
    img = plt.scatter(Mbh, phi_max*radians, c = Rsi_phi_max, cmap = 'viridis')
    cbar = plt.colorbar(img)
    cbar.set_label(r'$R_{SI} [R_\odot]$', fontsize = 15)
    plt.xscale('log')
    plt.xlabel(r'$M_{BH}$', fontsize = 15)
    plt.ylabel(r'$\phi_{max}$ [rad]', fontsize = 15)
    plt.grid()
    plt.title(f'$M_\star = {mstar} M_\odot,  R_\star = {Rstar} R_\odot$, ' + r'$\beta\in$ [' + f'{np.round(beta_many[0], 1)}, {np.round(beta_many[-1])}]', fontsize = 18)
    if save:
        plt.savefig('/Users/paolamartire/shocks/Figs/phiRsimax.png')
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
