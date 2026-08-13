""" Volume integrated energies (orbital, kinetic, internal and radiation) at each snapshot.
Cut in density (at 1e-19 code units), but not for radiation."""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/data1/martirep/shocks/shock_capturing'
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    compute = False

import numpy as np
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.operators import make_tree, choose_sections, choose_observers
import Utilities.sections as sec
import src.orbits as orb
import Utilities.prelude as prel
import csv
import os
import gc
import healpy as hp
#
## PARAMETERS STAR AND BH
#%%
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
what_paper = 'paper2' # 'paper1' or 'paper2'
choice = 'split_stream' # only for paper2

#%%
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
a_mb = things['a_mb']
t_fall = things['t_fb_days']
t_fall_cgs = t_fall * 24 * 3600


#%%
if compute:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    if what_paper == 'paper2':
        energies = {} 

    for i,snap in enumerate(snaps):
        # if snap not in [76, 109]:
        #     continue
        print(snap, flush = True)

        path = select_prefix(m, check, mstar, Rstar, beta, n, compton)
        if alice:
            path = f'{path}/snap_{snap}'
        else:            
            path = f'{path}/{snap}'
        data = make_tree(path, snap)
        X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den, Press, Diss_den = \
            data.X, data.Y, data.Z, data.VX, data.VY, data.VZ, data.Mass, data.Vol, data.Den, data.IE, data.Rad, data.Press, data.Diss
        # cut all in density BUT radiation
        cut = den > 1e-19 
        X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den, Press, Diss_den = \
            sec.make_slices([X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den, Press, Diss_den], cut)
        Rsph = np.sqrt(np.power(X, 2) + np.power(Y, 2) + np.power(Z, 2))
        vel = np.sqrt(np.power(VX, 2) + np.power(VY, 2) + np.power(VZ, 2))
        Diss_den[Diss_den<0] = 0

        if what_paper == 'paper1': # Reproduce Fig.3 of Eddington paper
            kin_en = 0.5 * mass *vel**2
            orb_en = orb.orbital_energy(Rsph, vel, mass, params, prel.G)
            ie = ie_den * vol
            Rad = Rad_den * vol

            # total energies with only the cut in density (not in radiation)
            tot_ie = np.sum(ie)
            tot_orb_en_pos = np.sum(orb_en[orb_en > 0])
            tot_orb_en_neg = np.sum(orb_en[orb_en < 0])
            tot_Rad = np.sum(Rad)
            tot_kin_en_pos = np.sum(kin_en[orb_en >= 0])
            tot_kin_en_neg = np.sum(kin_en[orb_en < 0])

            data_E = [snap, tfb[i], tot_ie, tot_orb_en_pos, tot_orb_en_neg, tot_Rad, tot_kin_en_pos, tot_kin_en_neg]
            csv_path = f'{abspath}/data/{folder}/convE_{check}.csv'
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
                    header = ['snap', ' tfb', ' tot_ie', ' tot_orb_en_pos', ' tot_orb_en_neg', ' tot_Rad', ' tot_kin_en_pos', ' tot_kin_en_neg']
                    writer.writerow(header)
                writer.writerow(data_E)
            file.close()

        if what_paper == 'paper2': 
            cut_wind, bern_spec, _ = orb.pick_wind(X, Y, Z, VX, VY, VZ, den, mass, Press, ie_den, Rad_den, params, cond = 'bern')
            dyn_unb = np.logical_and(np.abs(Z)<vol**(1/3), X < -apo)
            cut = np.logical_and(cut_wind, ~dyn_unb)
            X, Y, Z, Rsph, VX, VY, VZ, vel, mass, vol, den, ie_den, Rad_den, Press, Diss_den = \
                sec.make_slices([X, Y, Z, Rsph, VX, VY, VZ, vel, mass, vol, den, ie_den, Rad_den, Press, Diss_den], cut)
            Ekin = 0.5 * mass * vel**2
            orb_en = orb.orbital_energy(Rsph, vel, mass, params, prel.G)
            ie = ie_den * vol
            Rad = Rad_den * vol
            Diss = Diss_den * vol

            sections = choose_sections(X, Y, Z, choice)
            label_obs = []
            cond_sec = []
            for key in sections.keys():
                label_obs.append(sections[key]['label'])
                cond_sec.append(sections[key]['cond'])

            Ekin_sec = np.zeros(len(sections))
            OE_sec = np.zeros(len(sections))
            IE_sec = np.zeros(len(sections))
            Rad_sec = np.zeros(len(sections))
            Diss_sec = np.zeros(len(sections))
            for k, cond in enumerate(cond_sec):
                Ekin_sec[k] = np.sum(Ekin[cond]) if cond.size > 0 else 0
                OE_sec[k] = np.sum(orb_en[cond]) if cond.size > 0 else 0
                IE_sec[k] = np.sum(ie[cond]) if cond.size > 0 else 0
                Rad_sec[k] = np.sum(Rad[cond]) if cond.size > 0 else 0
                Diss_sec[k] = np.sum(Diss[cond]) if cond.size > 0 else 0

            E_snap = {'tfb': tfb[i], 
                      'Ekin_sec': Ekin_sec,
                      'OE_sec': OE_sec,
                      'IE_sec': IE_sec,
                      'Rad_sec': Rad_sec,
                      'Diss_sec': Diss_sec, 
                      'label_obs': label_obs}
            
            key = f"{int(snap)}"
            energies[key] = E_snap

        del X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den, Ekin
        gc.collect()   
    out_path = f'{abspath}/data/{folder}/wind/Diss_bern_{choice}.npy'
    np.save(out_path, energies, allow_pickle=True) 

if plot:
    import matplotlib.pyplot as plt
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    if what_paper == 'paper1':
        data = np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/convE_{check}.csv', delimiter=',', dtype=float, skiprows=1)
        snaps, tfb, IE, OEpos, OEEneg, Rad, Kinpos, Kinneg = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7]
        dataDiss = np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
        tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3] *  prel.en_converter/prel.tsol_cgs
        totalK = Kinneg + Kinpos

        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,7))
        figL, axL = plt.subplots(1,1, figsize = (10,6))
        ax1.plot(tfb, prel.en_converter * OEpos, c = '#fbb4b9', label = 'Orbital energy unbound gas')
        ax1.plot(tfb, np.abs(prel.en_converter * OEEneg), c = '#fbb4b9', ls = ':', label = 'Orbital energy bound gas (abs value) ')
        ax1.set_title(r'OE [erg]', fontsize = 24) 
        ax2.set_ylim(1e43, 6e49)
        # ax1.set_yscale('log')

        ax2.plot(tfb, prel.en_converter * IE, c = '#f768a1', label = 'Thermal energy')
        ax2.plot(tfb, prel.en_converter * Rad, c = '#7a0177', label = 'Radiation energy')
        ax2.plot(tfb, prel.en_converter * Kinpos, c = '#fbb4b9', label = 'Kinetic energy unbound gas')
        ax2.plot(tfb, np.abs(prel.en_converter * Kinneg), c = '#fbb4b9', ls = ':', label = 'Kinetic energy bound gas (abs value)')
        ax2.set_title(r'Thermal and radiation (erg)', fontsize = 24) 

        # compute rates 
        dtH = np.diff(tfb * t_fall_cgs)
        dOEpos = np.diff(OEpos * prel.en_converter)
        dOEEneg = np.diff(OEEneg * prel.en_converter)
        dIE = np.diff(IE * prel.en_converter)
        dRad = np.diff(Rad * prel.en_converter)
        dKinpos = np.diff(Kinpos * prel.en_converter)
        dKinneg = np.diff(Kinneg * prel.en_converter)
        dTotalK = np.diff(totalK * prel.en_converter)
        axL.plot(tfb[:-1], np.abs(dOEpos)/dtH, c = '#fbb4b9', linewidth = 2, label = 'Orb. en. unbound gas')
        axL.plot(tfb[:-1], np.abs(dOEEneg)/dtH, c = '#fbb4b9', ls = ':', linewidth = 2, label = 'Orb. en. bound gas')
        axL.plot(tfb[:-1], np.abs(dIE)/dtH, c = '#f768a1', linewidth = 2, label = 'Thermal energy')
        axL.plot(tfb[:-1], np.abs(dRad)/dtH, c = '#7a0177', linewidth = 2, label = 'Radiation energy')
        axL.plot(tfbdiss, LDiss, c = 'gray', linewidth = 2, label = r'$\dot{E}_{\rm irr}$', ls = '--')
        # axL.plot(tfb[:-1], np.abs(dKinpos)/dtH, c = '#fbb4b9', label = 'Kinetic energy unbound gas')
        # axL.plot(tfb[:-1], np.abs(dKinneg)/dtH, c = '#fbb4b9', ls = ':', label = 'Kinetic energy bound gas (abs value)')
        # axL.plot(tfb[:-1], np.abs(dTotalK)/dtH, c = 'brown', label = 'Total Kinetic energy')
        axL.set_ylabel(r'$|\dot{E}|$ (erg/s)') 
        axL.set_ylim(1e39, 1e44)

        orginal_ticks = axL.get_xticks()
        middle_ticks = (orginal_ticks[:-1] + orginal_ticks[1:]) /2
        new_ticks = np.sort(np.concatenate((orginal_ticks, middle_ticks)))
        labels = [str(np.round(tick,2)) if tick in orginal_ticks else "" for tick in new_ticks]       
        for ax in (ax1, ax2, axL):
            ax.tick_params(axis='both', which='major', width=1.2, length=7)
            ax.tick_params(axis='both', which='minor', width=0.9, length=5)
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel(r't / t$_{\rm fb}$')
            if ax != ax1:
                ax.set_yscale('log')
            ax.legend(fontsize = 15, loc = 'lower right')
            ax.grid()
            ax.set_xlim(0, np.max(tfb))
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/1.paperEddEbudget_{check}.png', dpi = 300)
        figL.tight_layout()
        figL.savefig(f'{abspath}/Figs/1.paperEddEbudget_absrates_{check}.pdf', dpi = 300)


    if what_paper == 'paper2':
        # define regions
        observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
        observers_xyz = np.array(observers_xyz)
        indices_sorted, label_obs, colors_obs, _, _ = choose_observers(observers_xyz, choice = choice)
        
        # data = np.load(f'{abspath}/data/{folder}/wind/Diss_bern_{choice}.npy', allow_pickle=True).item()
        # tfb = data[:, 1]
        # tfb_cgs = tfb * t_fall_cgs
        # Diss_sec = data[:, 2:2+len(label_obs)]
        # bern_sec = data[:, 2+len(label_obs):2+2*len(label_obs)] 
        data = np.load(f'{abspath}/data/{folder}/wind/Diss_bern_{choice}.npy', allow_pickle=True).item()
        
        tfb = np.array([data[key]['tfb'] for key in data.keys()])
        OE_sec = np.array([data[key]['OE_sec'] for key in data.keys()])
        IE_sec = np.array([data[key]['IE_sec'] for key in data.keys()])
        Rad_sec = np.array([data[key]['Rad_sec'] for key in data.keys()])
        Diss_sec = np.array([data[key]['Diss_sec'] for key in data.keys()])
        # bern_neg_sec = np.array([data[key]['bern_neg_sec'] for key in data.keys()])
        # bern_pos_sec = np.array([data[key]['bern_pos_sec'] for key in data.keys()])
        # bern_sec = bern_neg_sec + bern_pos_sec
        # bern_sec[bern_sec < 0] = 1e-20

        Diss_sec_cgs = Diss_sec * prel.en_converter/prel.tsol_cgs 
        en_diss_cgs = Diss_sec_cgs * t_fall_cgs
        # bern_sec_cgs = bern_sec * prel.en_converter
        # delta_en = np.diff(bern_sec_cgs, axis = 0)
        delta_diss = np.diff(en_diss_cgs, axis = 0)

        figE, (axE, axD) = plt.subplots(1, 2, figsize=(18,6))
        for i, lab in enumerate(label_obs): 
            if i not in [0,1]:
                continue
            # axE.plot(tfb, bern_sec_cgs[:, i], c = colors_obs[i], ls = '--') 
            axE.plot(tfb, en_diss_cgs[:, i], c = colors_obs[i], label = lab)
            # axD.plot(tfb[1:], delta_en[:, i], c = colors_obs[i], ls = '--', label = f'total energy' if i == 0 else None)
            axD.plot(tfb[1:], delta_diss[:, i], c = colors_obs[i],  label = f'dissipation' if i == 0 else None)

        for ax in (axE, axD):
            ax.tick_params(axis='both', which='major', width=1.2, length=7)
            ax.tick_params(axis='both', which='minor', width=0.9, length=5)
            ax.grid()
            ax.set_xlabel(r'$t/t_{\rm fb}$')
            ax.set_yscale('log')
            ax.legend(fontsize = 16)
        axE.set_ylabel(r'E (erg)')
        axD.set_ylabel(r'$\Delta$E (erg)')

# %%
