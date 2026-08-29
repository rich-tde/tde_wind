""" Compute the time evolution of Mdot fallback and Mdot wind across a spherical surface"""
abspath = '/Users/paolamartire/shocks'
import sys

from scipy import io
sys.path.append(abspath)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
import os
from matplotlib import lines as MdotMines
import healpy as hp
from scipy import integrate as sci
# from scipy.ndimage import uniform_filter1d
import Utilities.prelude as prel
from Utilities.operators import choose_observers, sort_list, area_spherical_zone, area_spherical_cal
from src import orbits as orb
from plotting.paperEdd.IHopeIsTheLast import ratio_BigOverSmall
from src.Wind.Rtrapp_tdiff import load_and_smooth_rtrap

m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR' 
choice = 'split_stream'
how = 'isot'
which_plot = 'MdotM' # 'MdotM' or 'MdotL_conv'
commonfolder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'
folder = f'{commonfolder}{check}'
observers_xyz = hp.pix2vec(prel.NSIDE, range(prel.NPIX))
observers_xyz = np.array(observers_xyz)
indices_obs, label_obs, color_obs, lines_obs, _ = choose_observers(observers_xyz, choice)
Ledd_sol, Medd_sol = orb.Edd(Mbh, 1.44/(prel.Rsol_cgs**2/prel.Msol_cgs), 1, prel.csol_cgs, prel.G)
Ledd_cgs = Ledd_sol * prel.en_converter/prel.tsol_cgs
Medd_cgs = Medd_sol * prel.Msol_cgs/prel.tsol_cgs 
params = [Mbh, Rstar, mstar, beta]
things = orb.get_things_about(params)
t_fb_days = things['t_fb_days']
amin = things['a_mb']
Rt = things['Rt']

r_chosen = 0.5*amin
which_r_title = '05amin'

if which_plot == 'MdotM':
    figM, (axM, axzeta, axMass) = plt.subplots(1, 3, figsize = (24, 7))  
    figr, axr  = plt.subplots(1,1, figsize = (9,7))
    axzeta.set_ylim(1e-3, 2)
    axMass.set_ylim(1e-5, 1.5)
    axzeta.set_ylabel(r'$\zeta = |\dot{M}_{\rm w}/\dot{M}_{\rm fb}|$')
    axes = [axM, axzeta, axr, axMass]
    axr.set_ylabel(r'$r (r_{\rm t})$')
    axMass.set_ylabel(r'M$_{\rm w}/(m_\star$/2)')
    axM.set_ylabel(r'$\dot{M} (r=0.5a_{\rm mb}) /\dot{M}_{\rm Edd}$')  
    axr.set_ylim(1, 1.2e2)
else: 
    figM, (axM, axL) = plt.subplots(1, 2, figsize = (16, 7))  
    figr, (axrM, axrL)  = plt.subplots(1,2, figsize = (16,7))
    axes = [axM, axrM, axL, axrL]
    axL.set_ylim(1e38, 5e42)
    axrM.set_ylim(1, 1e2)
    axrL.set_ylim(1, 1e2)
    axL.set_ylabel(r'$L_{\rm FLD}$ (erg/s)')  
    axM.set_ylabel(r'$\dot{M}_{\rm w} (\dot{M}_{\rm Edd})$')  
axM.set_ylim(1e2, 8e7)

if which_plot == 'MdotM':
    dataM = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps, tfb_Lum, _ = dataM[:, 0], dataM[:, 1], dataM[:, 2]
    tfb_Lum, snaps = sort_list([tfb_Lum, snaps], snaps, unique=True)

    fallback = \
            np.loadtxt(f'{abspath}/data/{folder}/1.paperEdd/wind/Mdot_{check}{which_r_title}mean.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True)
    tfbfb, mfb = fallback[1], fallback[2]
    tfb_to_int = tfbfb * 24 * 3600 / prel.tsol_cgs
    where_nan = np.isnan(mfb)
    mfb[where_nan] = 0
    mass_fb = sci.cumulative_trapezoid(np.abs(mfb), tfb_to_int, initial = 0)

    rph_sec = []
    rtr_sec = []
    snaps = np.array(snaps, dtype=int)
    for s, snap in enumerate(snaps): 
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        x_ph, y_ph, z_ph, Lum_ph = photo['x'], photo['y'], photo['z'], photo['Lum']
        r_ph = np.sqrt(x_ph**2 + y_ph**2 + z_ph**2)
        trap = load_and_smooth_rtrap(f'{abspath}/data/{folder}/trap', check, snap)
        x_tr, y_tr, z_tr = trap['x_tr'], trap['y_tr'], trap['z_tr']
        r_tr = np.sqrt(x_tr**2 + y_tr**2 + z_tr**2)
        if choice in ['tenths','azimuthal', 'funnel', '3d_arch', 'split_stream']:
            mask = []
            rph_slab = np.zeros(len(indices_obs))
            for i, indices in enumerate(indices_obs):
                mask.append(r_tr[indices] > 0)
        else:
            mask = r_tr[indices_obs] > 0
        indices_sec = [row[m] for row, m in zip(indices_obs, mask)]
        rtr_sec.append([np.median(r_tr[row]) for row in indices_sec])
        rph_sec.append([np.median(r_ph[row]) for row in indices_sec])
        # rtr_sec.append(np.mean(r_tr[indices_obs], axis = 1))
        # rph_sec.append(np.mean(r_ph[indices_obs], axis = 1))
    rph_sec = np.transpose(np.array(rph_sec))
    rtr_sec = np.transpose(np.array(rtr_sec))

    if choice == 'split_stream':
        corr_ecc = area_spherical_zone(r_chosen, 80 * np.pi/180)/2 # since the other half is peric side
        corr_mid = area_spherical_zone(r_chosen, 50 * np.pi/180)/2 - corr_ecc
        corr_high = area_spherical_zone(r_chosen, 20 * np.pi/180)/2 - corr_mid - corr_ecc
        corr_pole = area_spherical_cal(r_chosen, 20 * np.pi/180)
        corr_peric = area_spherical_zone(r_chosen, 20 * np.pi/180)/2 # since the other half is stream side
        corr_geom = np.array([corr_ecc, corr_mid, corr_high, corr_peric, corr_pole, corr_pole])
    # print(np.sum(corr_geom)/(4 * np.pi * r_chosen**2))


    wind = \
            np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_wind_NO_P.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True) 
    tfb = wind[1]
    time_to_int = tfb * 24 * 3600 / prel.tsol_cgs
    rest = wind[2:2+len(label_obs)]
    Mdotw = np.copy(rest)
    Mdotw_isot = np.copy(rest)

    boundOut = \
            np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}{which_r_title}{choice}_boundOut.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True) 
    tfbO = boundOut[1]
    timeBoundOut_to_int = tfbO * 24 * 3600 / prel.tsol_cgs
    restboundOut = boundOut[2:2+len(label_obs)]
    MdotboundOut = np.copy(restboundOut)
    MdotboundOut_isot = np.copy(restboundOut)
 
    if how == 'isot':
        area = wind[-len(label_obs):]
        areaboundOut = boundOut[-len(label_obs):]
        corr = []
        corrboundOut = []
        for i in range(len(rest)):
            Mdotw_isot[i][area[i]>0] *=  4 * np.pi * r_chosen**2 / area[i][area[i]>0] 
            factor_corr = corr_geom[i] / area[i]
            factor_corr[area[i]==0] = 0
            corr.append(factor_corr)
            MdotboundOut_isot[i][areaboundOut[i]>0] *=  4 * np.pi * r_chosen**2 / areaboundOut[i][areaboundOut[i]>0]
            factor_corrboundOut = corr_geom[i] / areaboundOut[i]
            factor_corrboundOut[areaboundOut[i]==0] = 0
            corrboundOut.append(factor_corrboundOut)
            # rest[i][np.isnan(rest[i])] = 0
            # restboundOut[i][np.isnan(restboundOut[i])] = 0
        corr = np.array(corr)
        corrboundOut = np.array(corrboundOut)

    MdotO_isot = MdotboundOut_isot + Mdotw_isot
    MdotO= MdotboundOut + Mdotw

    dataMass = np.loadtxt(f'{abspath}/data/{folder}/wind/Mass_unbound{choice}.csv', 
                        delimiter=',', skiprows=1, unpack=True)
    tfbMass = dataMass[1]
    M_tot = dataMass[2:2+len(label_obs)] 
    M_out = dataMass[2+(len(label_obs)):2+2*(len(label_obs))] 
    M_wind = dataMass[2+2*(len(label_obs)):2+3*(len(label_obs))] 

    M_out_corr = np.copy(M_out)
    M_wind_corr = np.copy(M_wind)
    Mass_boundOut_int = []
    Mass_wind_int = []
    for i in range(len(label_obs)):
        MdotO[i][np.isnan(MdotO[i])] = 0
        Mdotw[i][np.isnan(Mdotw[i])] = 0
        Mass_boundOut_int_sigle = sci.cumulative_trapezoid(MdotboundOut[i]*corrboundOut[i], timeBoundOut_to_int, initial = 0)
        Mass_wind_int_single = sci.cumulative_trapezoid(Mdotw[i], time_to_int, initial = 0)
        Mass_boundOut_int.append(Mass_boundOut_int_sigle)
        Mass_wind_int.append(Mass_wind_int_single)
        # don't need to substract to the integral form since you compute Mdot at 0.5amin
        # Mass_out_int_sigle -= M_wind[i, 0]
        # Mass_wind_int_single -= M_wind[i, 0]
        M_out_corr[i, :] -= M_wind[i, 0]
        M_wind_corr[i, :] -= M_wind[i, 0]
        
    Mass_out_int = Mass_boundOut_int + Mass_wind_int 
    handles_color = []
    labels_color = []
    line_styles_parts = ['-', '--']
    labels_parts = [r'$\dot{M}_{\rm w}$', r'$\dot{M}_{\rm out}$']
    axMass.plot(tfbMass, mass_fb/(0.5*mstar), c = 'gray', ls = ':', linewidth = 2)
    axMass.text(1.91, 0.17, r'$\int\,\dot{M}_{\rm fb} {\rm d}t$', fontsize = 16, color = 'gray', rotation = 5)
    # axMass.legend(fontsize = 20, loc = 'upper left')
    for i in range(len(label_obs)):
        if label_obs[i] == 'South pole':
            continue
        axM.plot(tfbO[4:], MdotO_isot[i][4:]/Medd_sol, c = color_obs[i], ls = line_styles_parts[1])
        # axMass.plot(tfbO,  Mass_out_int[i]/(0.5*mstar), c = color_obs[i], label = label_obs[i], ls = ':')
        axMass.plot(tfbMass,  M_out_corr[i]/(0.5*mstar), c = color_obs[i], label = label_obs[i], ls = '--')
        if label_obs[i] != 'Eccentric flow side':
            axM.plot(tfb, Mdotw_isot[i]/Medd_sol,  label = label_obs[i], c = color_obs[i], ls = line_styles_parts[0])
            # axMass.plot(tfbMass,  Mass_wind_int[i]/(0.5*mstar), c = color_obs[i], label = label_obs[i], ls = ':') 
            axMass.plot(tfbMass,  M_wind_corr[i]/(0.5*mstar), c = color_obs[i])
            axzeta.plot(tfb[7:], corr[i][7:] * Mdotw[i][7:]/np.abs(mfb[7:]),  label = label_obs[i], c = color_obs[i])
            print(label_obs[i], 'outflow/half star mass: ', np.median(M_out_corr[i, -3:])/(0.5*mstar), ', wind/out: ', np.median(M_wind_corr[i, -3:])/np.median(M_out_corr[i, -3:]))
        print('sum stream: ', (np.median(M_out_corr[0,-3:])+np.median(M_out_corr[1, -3:])+np.median(M_out_corr[2, -3:]))/(0.5*mstar), ', sum wind/out: ', np.sum(np.median(M_wind_corr[0, -3:])+np.median(M_wind_corr[1, -3:])+np.median(M_wind_corr[2, -3:]))/(np.median(M_out_corr[0, -3:])+np.median(M_out_corr[1, -3:])+np.median(M_out_corr[2, -3:])))
        # print('sum mid+high stream: ', (np.median(M_out_corr[1, -3:])+np.median(M_out_corr[2, -3:]))/(0.5*mstar), ', sum wind/out: ', np.sum(np.median(M_wind_corr[1, -3:])+np.median(M_wind_corr[2, -3:]))/np.sum(np.median(M_out_corr[1, -3:])+np.median(M_out_corr[2, -3:])))

        axr.plot(tfb_Lum, rtr_sec[i]/Rt, c = color_obs[i], ls = ':', label = r'r$_{\rm trap}$' if i == 2 else "")
        axr.plot(tfb_Lum, rph_sec[i]/Rt, c = color_obs[i], label = r'r$_{\rm ph}$' if i == 2 else "")

        handles_color.append(color_obs[i])
        labels_color.append(label_obs[i])

if which_plot == 'MdotL_conv':
    data = np.loadtxt(f'{abspath}/data/{folder}/{check}_red.csv', delimiter=',', dtype=float)
    snaps, _, _ = data[:, 0], data[:, 1], data[:, 2]
    _, snaps, _ = sort_list([_, snaps, _], snaps, unique=True)
    Lum_sec = []
    snaps = np.array(snaps, dtype=int)
    for s, snap in enumerate(snaps): 
        photo = np.load(f'{abspath}/data/{folder}/photo/{check}_photo{snap}.npz')
        Lum_ph = photo['Lum']
        if choice in ['tenths','azimuthal', 'funnel', '3d_arch', 'split_stream']:
            Lum_slab = np.zeros(len(indices_obs))
            rph_slab = np.zeros(len(indices_obs))
            for i, indices in enumerate(indices_obs):
                Lum_slab[i] = np.mean(Lum_ph[indices])
            Lum_sec.append(Lum_slab)
        else:
            Lum_sec.append(np.mean(Lum_ph[indices_obs], axis = 1))

    Lum_sec = np.transpose(np.array(Lum_sec))
    
    windH = \
            np.loadtxt(f'{abspath}/data/{folder}/wind/MdotSec{how}_{check}05amin{choice}_wind.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True) 
    tfb = windH[1]
    rest = windH[2:2+len(label_obs)]

    if how == 'isot':
        area = windH[-len(label_obs):]
        rest *=  4 * np.pi * r_chosen**2 / area 

    # for i in range(len(rest)):
        # rest[i][area[i]>0] = uniform_filter1d(rest[i][area[i]>0], 3)
        # Lum_sec[i] = uniform_filter1d(Lum_sec[i], 3)

    dataM = np.loadtxt(f'{abspath}/data/{commonfolder}NewAMR/NewAMR_red.csv', delimiter=',', dtype=float)
    snapsM  = np.sort(dataM[:, 0])
    Lum_secM = []
    snapsM = np.array(snapsM, dtype=int)
    for s, snap in enumerate(snapsM): 
        photo = np.loadtxt(f'{abspath}/data/{commonfolder}NewAMR/photo/NewAMR_photo{snap}.txt')
        Lum_ph = photo[-2]
        if choice in ['tenths','azimuthal', 'funnel', '3d_arch', 'split_stream']:
            Lum_slab = np.zeros(len(indices_obs))
            rph_slab = np.zeros(len(indices_obs))
            for i, indices in enumerate(indices_obs):
                Lum_slab[i] = np.mean(Lum_ph[indices])
            Lum_secM.append(Lum_slab)
        else:
            Lum_secM.append(np.mean(Lum_ph[indices_obs], axis = 1))

    Lum_secM = np.transpose(np.array(Lum_secM))

    windM = \
            np.loadtxt(f'{abspath}/data/{commonfolder}NewAMR/wind/MdotSec{how}_NewAMR05amin{choice}_wind.csv', 
                    delimiter = ',', 
                    skiprows=1, 
                    unpack=True) 
    tfbM = windM[1]
    restM = windM[2:2+len(label_obs)]

    if how == 'isot':
        area = windM[-len(label_obs):]
        restM *=  4 * np.pi * r_chosen**2 / area
        
    # for i in range(len(restM)):
        # restM[i][area[i]>0] = uniform_filter1d(restM[i][area[i]>0], 3)
        # Lum_secM[i] = uniform_filter1d(Lum_secM[i], 3)

    handles_color = []
    labels_color = []
    line_styles_parts = ['-', '-.']
    labels_parts = ['High res', 'Middle res']

    for i in range(len(rest)):
        if label_obs[i] in ['South pole', 'Eccentric flow side']:
            continue
        line = axM.plot(tfb, rest[i]/Medd_sol,  label = label_obs[i], linewidth = 2, c = color_obs[i], ls = line_styles_parts[0])[0]
        axL.plot(tfb, Lum_sec[i],  label = label_obs[i], linewidth = 2, c = color_obs[i], ls = line_styles_parts[0])
        axM.plot(tfbM, restM[i]/Medd_sol, linewidth = 2, c = color_obs[i], ls = line_styles_parts[1])
        axL.plot(tfbM, Lum_secM[i],  label = label_obs[i], linewidth = 2, c = color_obs[i], ls = line_styles_parts[1])
        handles_color.append(line)
        labels_color.append(label_obs[i])
        time_ratio, ratio, _ = ratio_BigOverSmall(tfb, rest[i], tfbM, restM[i])
        print(label_obs[i], 'Mdot ratio after 1.5: ', np.median(ratio[time_ratio> 1.5]))
        cut = np.logical_and(~np.isinf(ratio), ~np.isnan(ratio))
        axrM.plot(time_ratio[cut], ratio[cut], linewidth = 2, c = color_obs[i], label = label_obs[i])
        time_ratio, ratio, _ = ratio_BigOverSmall(tfb, Lum_sec[i], tfbM, Lum_secM[i])
        cut = np.logical_and(~np.isinf(ratio), ~np.isnan(ratio))
        axrL.plot(time_ratio[cut], ratio[cut], linewidth = 2, c = color_obs[i], label = label_obs[i])
        print(label_obs[i], 'Lum ratio after 1.5: ', np.median(ratio[time_ratio > 1.5]))

    axL.set_xlim(0, np.max(tfb))    # you need it for get.ticks
    axrM.set_ylabel(r'$\dot{M}_{\rm HiRes}/\dot{M}_{\rm MidRes}$')
    axrL.set_ylabel(r'$L_{\rm HiRes}/L_{\rm MidRes}$')
    # axr.set_ylim(1, 11)

axM.set_xlim(0, np.max(tfb))
original_ticks = axM.get_xticks()
midpoints = (original_ticks[:-1] + original_ticks[1:]) / 2
new_ticks = np.sort(np.concatenate((original_ticks, midpoints)))
labels = [str(np.round(tick,2)) if tick in original_ticks else '' for tick in new_ticks]   
days_ticks = new_ticks*t_fb_days
days_labels = [str(np.round(days_ticks[k],2)) if new_ticks[k] in original_ticks else "" for k in range(len(days_ticks))] 
for ax in axes:
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(labels)  
    ax.set_yscale('log')
    ax.set_xlim(0, np.max(tfb))
    ax.tick_params(axis='both', which='major', width=1.2, length=9)
    ax.tick_params(axis='both', which='minor', width=1, length=5)
    ax.grid()
    
    ax2 = ax.twiny()
    ax2.set_xticks(days_ticks)
    ax2.set_xlim(0, np.max(tfb)*t_fb_days)
    # if ax == axM:
    ax2.set_xlabel(r't (days)', y = 1.1)
    ax2.set_xticklabels(days_labels)
    # else: 
    # ax2.tick_params(axis='x', labeltop=False)

    # if ax == axMass:
    ax.set_xlabel(r'$t / t_{\rm fb}$')
    # else: 
    # ax.tick_params(axis='x', labelbottom=False)


# Legend 1: colored observer lines (three colors)
# legend1 = axM.legend(handles=handles_color,
#                     labels=labels_color,
#                     fontsize=15)
proxy_lines = []
proxy_lines = []
for l, line in enumerate(lines_obs):
    if label_obs[l] == 'South pole' or label_obs[l] == 'Eccentric flow':
        continue
    proxy_lines.append(
        MdotMines.Line2D([0], [0], color=color_obs[l], ls=line, linewidth=2,
                    label=label_obs[l]))
legend1 = axM.legend(handles=proxy_lines, fontsize=18, 
                                loc='upper left')
axM.add_artist(legend1)

# Legend 2: line-style explanation (solid vs dashed)
proxy_lines = []
proxy_lines = []
for l, line in enumerate(line_styles_parts):
    proxy_lines.append(
        MdotMines.Line2D([0], [0], color='k', ls=line, linewidth=2,
                    label=labels_parts[l])
    )
axM.legend(handles=proxy_lines, fontsize=19, 
                                loc='upper right' if which_plot == 'MdotL_conv' else 'lower right')

figM.tight_layout()
if which_plot == 'MdotM':
    axr.legend(fontsize = 20)
    figr.tight_layout()
if choice == 'tenths':
    figM.suptitle('Angle varying from +x (0 deg), to +z (90 deg), to -x (180 deg)', fontsize = 30, y = 1.02)
if choice == 'azimuthal':
    figM.suptitle('Azimuthal angle', fontsize = 30, y = 1.02)
if choice == 'split_stream':
    figM.savefig(f'{abspath}/Figs/2.paperWind/{which_plot}_intime_{choice}.pdf', dpi = 300, bbox_inches = 'tight')
else:
    figM.savefig(f'{abspath}/Figs/{folder}/wind/{which_plot}_intime_{choice}.png', dpi = 300, bbox_inches = 'tight')