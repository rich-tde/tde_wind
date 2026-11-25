""" Volume integrated energies (orbital, kinetic, internal and radiation) at each snapshot.
Cut in density (at 1e-19 code units), but not for radiation."""
import sys
sys.path.append('/Users/paolamartire/shocks/')
from Utilities.isalice import isalice
alice, plot = isalice()
if alice:
    abspath = '/home/martirep/data1/postprocess/tde'
    compute = True
else:
    abspath = '/Users/paolamartire/shocks'
    compute = False

import numpy as np
import csv
import os
import gc
import healpy as hp
from scipy import integrate as sci
from Utilities.selectors_for_snap import select_snap, select_prefix
from Utilities.operators import make_tree, choose_sections, choose_observers
import Utilities.sections as sec
import src.orbits as orb
import Utilities.prelude as prel
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
what_paper = 'paper1' # 'paper1' or 'paper2'
choice = 'split_stream' # only for paper2
what_to_keep = '_keepDynUnb' # only for paper2, '_keepDynUnb' or ''

#%%
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
Rs = things['Rs']
Rg = things['Rg']
Rt = things['Rt']
Rp = things['Rp']
R0 = things['R0']
apo = things['apo']
a_mb = things['a_mb']
t_fall = things['t_fb_days']
E_mb = things['E_mb']
t_fall_cgs = t_fall * 24 * 3600

#%%
if compute:
    snaps, tfb = select_snap(m, check, mstar, Rstar, beta, n, compton, time = True) #[100,115,164,199,216]
    if what_paper == 'paper2':
        out_path = f'{abspath}/data/{folder}/wind/energies{what_to_keep}_{choice}.npy'

        if os.path.exists(out_path):
            energies = np.load(out_path, allow_pickle=True).item()
        else:
            energies = {}

    for i,snap in enumerate(snaps):
        if what_paper == 'paper2':
            key = f"{int(snap)}"
            # Skip snapshots that were already completed
            if key in energies:
                print(f'Snapshot {snap} already saved, skipping', flush=True)
                continue
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
            cut_bound = ~cut_wind
            dyn_unb = np.logical_and(np.abs(Z)<vol**(1/3), X < -apo)
            if what_to_keep == '_keepDynUnb':
                if i == 0:
                    print('Keeping dynamically unbound material in the energy budget', flush = True)
                cut = cut_wind
            else:
                if i == 0:
                    print('Removing dynamically unbound material from the energy budget', flush = True)
                cut = np.logical_and(cut_wind, ~dyn_unb)

            ie = ie_den * vol
            Rad = Rad_den * vol
            Diss = Diss_den * vol
            bern = bern_spec * mass
            orb_en = orb.orbital_energy(Rsph, vel, mass, params, prel.G)
            Ekin = 0.5 * mass * vel**2
            Lkin = 0.5 * 4 * np.pi * Rsph**2* den * vel**3 

            sections = choose_sections(X, Y, Z, choice)
            label_obs = []
            cond_sec = []
            for key_sec in sections.keys():
                label_obs.append(sections[key_sec]['label'])
                cond_sec.append(sections[key_sec]['cond'])
            IE_sec_tot = np.zeros(len(sections))
            Rad_sec_tot = np.zeros(len(sections))
            Diss_sec_tot = np.zeros(len(sections))
            bern_sec_bound = np.zeros(len(sections))
            Ekin_sec = np.zeros(len(sections))
            Lkin_sec = np.zeros(len(sections))
            OE_sec = np.zeros(len(sections))
            IE_sec = np.zeros(len(sections))
            Rad_sec = np.zeros(len(sections))
            Diss_sec = np.zeros(len(sections))
            bern_sec = np.zeros(len(sections))
            R_sec = np.zeros(len(sections))
            for k, cond in enumerate(cond_sec):
                IE_sec_tot[k] = np.sum(ie[cond]) if cond.size > 0 else 0
                Rad_sec_tot[k] = np.sum(Rad[cond]) if cond.size > 0 else 0
                Diss_sec_tot[k] = np.sum(Diss[cond]) if cond.size > 0 else 0

                bern_sec_bound[k] = np.sum(bern[np.logical_and(cond, cut_bound)]) if cond.size > 0 else 0
            
                Ekin_sec[k] = np.sum(Ekin[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                Lkin_sec[k] = np.sum(Lkin[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                OE_sec[k] = np.sum(orb_en[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                IE_sec[k] = np.sum(ie[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                Rad_sec[k] = np.sum(Rad[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                Diss_sec[k] = np.sum(Diss[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                bern_sec[k] = np.sum(bern[np.logical_and(cond, cut)]) if cond.size > 0 else 0
                R_sec[k] = np.sum(Rsph[np.logical_and(cond, cut)]*mass[np.logical_and(cond, cut)])/np.sum(mass[np.logical_and(cond, cut)]) if cond.size > 0 else 0

            E_snap = {'tfb': tfb[i], 
                      'IE_sec_tot': IE_sec_tot,
                      'Rad_sec_tot': Rad_sec_tot,
                      'Diss_sec_tot': Diss_sec_tot, 
                      'Ekin_sec': Ekin_sec,
                      'Lkin_sec': Lkin_sec,
                      'OE_sec': OE_sec,
                      'IE_sec': IE_sec,
                      'Rad_sec': Rad_sec,
                      'Diss_sec': Diss_sec, 
                      'bern_sec': bern_sec,
                      'Ravg_sec': R_sec,
                      'bern_sec_bound': bern_sec_bound,
                      'label_obs': label_obs}
            
            energies[key] = E_snap

            np.save(out_path, energies, allow_pickle=True) 

            del X, Y, Z, VX, VY, VZ, mass, vol, den, ie_den, Rad_den, Ekin
            gc.collect()   

if plot:
    import matplotlib.pyplot as plt
    folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

    if what_paper == 'paper1':
        data = np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/convE_{check}.csv', delimiter=',', dtype=float, skiprows=1)
        snaps, tfb, IE, OEpos, OEEneg, Rad, Kinpos, Kinneg = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6], data[:, 7]
        dataDiss = np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/Rdiss_{check}.csv', delimiter=',', dtype=float, skiprows=1)
        tfbdiss, LDiss = dataDiss[:,1], dataDiss[:,3] *  prel.en_converter/prel.tsol_cgs
        totalK = Kinneg + Kinpos
        Ediss = sci.cumulative_trapezoid(LDiss, tfb*t_fall_cgs, initial = 0)

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
        ax2.plot(tfb, Ediss, c = 'orange', label = r'$\int \dot{E}_{\rm irr} dt$')
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
            # ax.axhline(E_mb * prel.en_converter, c = 'k', ls = '--', label = r'$E_{\rm mb}$')
        fig.tight_layout()
        fig.savefig(f'{abspath}/Figs/1.paperEddEbudget_{check}.png', dpi = 300)
        figL.tight_layout()
        figL.savefig(f'{abspath}/Figs/1.paperEddEbudget_absrates_{check}.pdf', dpi = 300)


    if what_paper == 'paper2':
        what = ''
        if what == 'ratio':
            from plotting.paperEdd.IHopeIsTheLast import ratio_FirstOverSecond, ratio_BigOverSmall
        
        # define regions
        figE, (axB, axRad, axDiss, axtot) = plt.subplots(1, 4, figsize=(35,7))
        observers_xyz = hp.pix2vec(prel.NSIDE, np.arange(prel.NPIX)) #shape: (3, 192)
        observers_xyz = np.array(observers_xyz)
        indices_sorted, label_obs, colors_obs, _, _, _ = choose_observers(observers_xyz, choice = choice)

        folderH = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}HiResNewAMR'
        dataH = np.load(f'{abspath}/data/{folder}/wind/energies{what_to_keep}_{choice}.npy', allow_pickle=True).item()
        tfbH = np.transpose(np.array([dataH[key]['tfb'] for key in dataH.keys()]))
        Ekin_secH = np.transpose(np.array([dataH[key]['Ekin_sec'] for key in dataH.keys()]))
        OE_secH = np.transpose(np.array([dataH[key]['OE_sec'] for key in dataH.keys()]))
        bern_secH = np.transpose(np.array([dataH[key]['bern_sec'] for key in dataH.keys()]))
        bern_sec_boundH = np.transpose(np.array([dataH[key]['bern_sec_bound'] for key in dataH.keys()]))
        IE_secH = np.transpose(np.array([dataH[key]['IE_sec'] for key in dataH.keys()]))
        Rad_secH = np.transpose(np.array([dataH[key]['Rad_sec'] for key in dataH.keys()]))
        Diss_secH = np.transpose(np.array([dataH[key]['Diss_sec'] for key in dataH.keys()]))
        IE_sec_totH = np.transpose(np.array([dataH[key]['IE_sec_tot'] for key in dataH.keys()]))
        Rad_sec_totH = np.transpose(np.array([dataH[key]['Rad_sec_tot'] for key in dataH.keys()]))
        Diss_sec_totH = np.transpose(np.array([dataH[key]['Diss_sec_tot'] for key in dataH.keys()]))

        # Ekin_sec_cgs = Ekin_sec * prel.en_converter
        # OE_sec_cgs = OE_sec * prel.en_converter
        # bern_sec_cgs = bern_sec * prel.en_converter
        # bern_sec_bound_cgs = bern_sec_bound * prel.en_converter
        # IE_sec_cgs = IE_sec * prel.en_converter
        # Rad_sec_cgs = Rad_sec * prel.en_converter
        # Rad_sec_tot_cgs = Rad_sec_tot * prel.en_converter
        Diss_sec_cgsH = Diss_secH * prel.en_converter/prel.tsol_cgs 
        Diss_sec_tot_cgsH = Diss_sec_totH * prel.en_converter/prel.tsol_cgs
        tfb_cgsH = tfbH * t_fall_cgs
        en_diss_cgsH = sci.cumulative_trapezoid(Diss_sec_cgsH, tfb_cgsH, axis = 1, initial = 0)
        en_diss_cgs_totH = sci.cumulative_trapezoid(Diss_sec_tot_cgsH, tfb_cgsH, axis = 1, initial = 0)
        all_EkinH = np.sum(Ekin_secH, axis = 0)
        all_IEH = np.sum(IE_secH, axis = 0)
        all_Diss_cgsH = np.sum(en_diss_cgsH, axis = 0)
        all_RadH = np.sum(Rad_secH, axis = 0)
        all_IEH_tot = np.sum(IE_sec_totH, axis = 0)
        all_Diss_cgsH_tot = np.sum(en_diss_cgs_totH, axis = 0)
        all_RadH_tot = np.sum(Rad_sec_totH, axis = 0)

        folderM = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}NewAMR'
        dataM = np.load(f'{abspath}/data/{folderM}/wind/energies{what_to_keep}_{choice}.npy', allow_pickle=True).item()
        tfbM = np.transpose(np.array([dataM[key]['tfb'] for key in dataM.keys()]))
        Ekin_secM = np.transpose(np.array([dataM[key]['Ekin_sec'] for key in dataM.keys()]))
        OE_secM = np.transpose(np.array([dataM[key]['OE_sec'] for key in dataM.keys()]))
        bern_secM = np.transpose(np.array([dataM[key]['bern_sec'] for key in dataM.keys()]))
        bern_sec_boundM = np.transpose(np.array([dataM[key]['bern_sec_bound'] for key in dataM.keys()]))
        IE_secM = np.transpose(np.array([dataM[key]['IE_sec'] for key in dataM.keys()]))
        Rad_secM = np.transpose(np.array([dataM[key]['Rad_sec'] for key in dataM.keys()]))
        Diss_secM = np.transpose(np.array([dataM[key]['Diss_sec'] for key in dataM.keys()]))
        IE_sec_totM = np.transpose(np.array([dataM[key]['IE_sec_tot'] for key in dataM.keys()]))
        Rad_sec_totM = np.transpose(np.array([dataM[key]['Rad_sec_tot'] for key in dataM.keys()]))
        Diss_sec_totM = np.transpose(np.array([dataM[key]['Diss_sec_tot'] for key in dataM.keys()]))

        Diss_sec_cgsM = Diss_secM * prel.en_converter/prel.tsol_cgs 
        Diss_sec_tot_cgsM = Diss_sec_totM * prel.en_converter/prel.tsol_cgs
        tfb_cgsM = tfbM * t_fall_cgs
        en_diss_cgsM = sci.cumulative_trapezoid(Diss_sec_cgsM, tfb_cgsM, axis = 1, initial = 0)
        en_diss_cgs_totM = sci.cumulative_trapezoid(Diss_sec_tot_cgsM, tfb_cgsM, axis = 1, initial = 0)
        all_EkinM = np.sum(Ekin_secM, axis = 0)
        all_IEM = np.sum(IE_secM, axis = 0)
        all_Diss_cgsM = np.sum(en_diss_cgsM, axis = 0)
        all_RadM = np.sum(Rad_secM, axis = 0)
        all_IEM_tot = np.sum(IE_sec_totM, axis = 0)
        all_Diss_cgsM_tot = np.sum(en_diss_cgs_totM, axis = 0)
        all_RadM_tot = np.sum(Rad_sec_totM, axis = 0)

        if what == 'ratio':
            axB.set_ylabel(r'E$_{\rm M}$/E$_{\rm H}$', fontsize = 20)
            time_ratio, ratioD_tot, _ = ratio_FirstOverSecond(tfbM, all_Diss_cgsM_tot, tfbH, all_Diss_cgsH_tot)
            axtot.plot(time_ratio, ratioD_tot, c = 'gray', ls = '--', label = r'$E_{\rm diss, tot}$')
            time_ratio, ratioR_tot, _ = ratio_FirstOverSecond(tfbM, all_RadM_tot, tfbH, all_RadH_tot)
            axtot.plot(time_ratio, ratioR_tot, c = 'gray', label = r'$E_{\rm rad, tot}$')
            time_ratio, ratioD, _ = ratio_FirstOverSecond(tfbM, all_Diss_cgsM, tfbH, all_Diss_cgsH)
            axtot.plot(time_ratio, ratioD, c = 'k', ls = '--', label = r'$E_{\rm diss, w}$')
            time_ratio, ratioR, _ = ratio_FirstOverSecond(tfbM, all_RadM, tfbH, all_RadH)
            axtot.plot(time_ratio, ratioR, c = 'k', label = r'$E_{\rm rad, w}$')
            time_ratio, ratioEkin, _ = ratio_FirstOverSecond(tfbM, all_EkinM, tfbH, all_EkinH)
            axtot.plot(time_ratio, ratioEkin, c = 'k', ls = ':', label = r'$E_{\rm kin, w}$')

            # axtot.plot(tfbM, np.sum(Diss_sec_tot_cgsM, axis = 0))
            # axtot.plot(tfbH, np.sum(Diss_sec_tot_cgsH, axis = 0))
            # axtot.set_ylim(1e38, 2e43)
            # axtot.plot(tfbM, all_Diss_cgsM_tot)
            # axtot.plot(tfbH, all_Diss_cgsH_tot)
            # axtot.set_ylim(7e-1, 10)
            axtot.legend(fontsize = 16)

        for i, lab in enumerate(label_obs): 
            if lab in ['Eccentric flow side','South pole']:
                continue
            if what == 'ratio':
                # time_ratio, ratioB, _ = ratio_FirstOverSecond(tfbM, bern_secM[i], tfbH, bern_secH[i])
                # axB.plot(time_ratio, ratioB, c = colors_obs[i], label = label_obs[i])
                time_ratio, ratioEk, _ = ratio_FirstOverSecond(tfbM, Ekin_secM[i], tfbH, Ekin_secH[i])
                axB.plot(time_ratio, ratioEk, c = colors_obs[i], label = label_obs[i])
                time_ratio, ratioRad, _ = ratio_FirstOverSecond(tfbM, Rad_secM[i], tfbH, Rad_secH[i])
                axRad.plot(time_ratio, ratioRad, c = colors_obs[i], label = label_obs[i])
                time_ratio, ratioDiss, _ = ratio_FirstOverSecond(tfbM, en_diss_cgsM[i], tfbH, en_diss_cgsH[i])
                axDiss.plot(time_ratio, ratioDiss, c = colors_obs[i], label = label_obs[i])
            else:
                axB.plot(tfbH, all_IEH_tot, c = 'blue', label = 'IE')
                axRad.plot(tfbH, all_RadH_tot, c = 'orchid', label = 'Rad')
                axDiss.plot(tfbH, all_Diss_cgsH_tot, c = 'orange', label = 'Diss')
                axB.plot(tfbM, all_IEM_tot, c = 'blue', ls = '--')
                axRad.plot(tfbM, all_RadM_tot, c = 'orchid', ls = '--')
                axDiss.plot(tfbM, all_Diss_cgsM_tot, c = 'orange', ls = '--')
                

        for ax in (axB, axRad, axDiss, axtot):
            ax.tick_params(axis='both', which='major', width=1.2, length=7)
            ax.tick_params(axis='both', which='minor', width=0.9, length=5) 
            ax.grid()
            ax.set_xlabel(r'$t/t_{\rm fb}$')
            ax.set_yscale('log')
            ax.legend(fontsize = 16)
            # ax.set_ylim(7e-1, 2e1)
        axB.legend(fontsize = 16)
        # axB.set_title(r'Bernouilli parameter', fontsize = 20)
        axB.set_title(r'Kinetic energy', fontsize = 20)
        axRad.set_title(r'Radiation energy', fontsize = 20)
        axDiss.set_title(r'Dissipation energy', fontsize = 20)

    # %%
    # print('Last value of Ekin')
    # print('middle stream: ', Ekin_sec_cgs[-1, 1], flush = True)
    # print('pole: ', Ekin_sec_cgs[-1, 4], flush = True)
    # print('estimate for Ekin from Mw and v', flush = True)
    # print('stream: ', 0.5*1e-2*mstar/2*prel.Msol_cgs*(0.014*prel.c_cgs)**2, flush = True)
    # print('pole: ', 0.5*1e-4*mstar/2*prel.Msol_cgs*(0.018*prel.c_cgs)**2, flush = True)

    # %%
