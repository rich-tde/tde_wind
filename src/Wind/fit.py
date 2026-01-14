""""
@author: paola , konstantinos
"""
import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from src.Luminosity.select_path import select_snap
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [10 , 8]
plt.rcParams['axes.facecolor']= 	'whitesmoke'
AEK = '#F1C410'

##
# VARIABLES
##

m = 6
c = 2.99792458e10 # [cm/s]
c_si = 2.99792458e8 # [m/s]
h = 6.62607015e-27 # [gcm^2/s]
Kb = 1.380649e-16 #[gcm^2/s^2K]
sigma = 5.67037e-5
# ztf: 3.23e14 - 6.71e14 // swift: 6.34e14 - 1.88e15
freq_min = 3.23e14
freq_max = 1.88e15


##
# FUNCTION
##
def tofit(n, R, T):
    const = 2*h/c**2
    planck = const * n**3 / (np.exp(h*n/(Kb*T))-1)
    Lum = 4 * np.pi**2 * R**2 * planck 
    return Lum

##
# MAIN
##

if __name__ == '__main__':
    plot = True
    save = True
    do = True
    check = 'fid'

    # Load & Unpack
    snapshots, days = select_snap(m, check)
    x = np.loadtxt('data/blue/frequencies_m' + str(m) + '.txt') # x = logν
    data = np.loadtxt('data/blue/L_tilda_spectrum_m' + str(m) + '.txt')

    freqs = np.power(10, x)
    init_R = 5e14
    init_T = 2e4
    
    freq_min_idx = np.argmin( np.abs(freqs - freq_min))
    freq_max_idx = np.argmin( np.abs(freqs - freq_max))

    fit_freqs = freqs[freq_min_idx:freq_max_idx]

    if do:
        temp = np.zeros(len(data))
        radius = np.zeros(len(data))
        Blue = np.zeros(len(data))
        
        # NOTE: last 4 because we save on top all the time and should fix that
        for i in range(len(data)):#(len(data) - 4, len(data)):
            Lums = data[i] 
            # print(Lums)
            Lums_fit = Lums[freq_min_idx:freq_max_idx]
            fit = curve_fit(tofit, fit_freqs, Lums_fit, p0 = (init_R, init_T))

            # # Integrate in log10: better to divide in the 3 bands to integrate
            # shouldn't we consider all the frequencies?
            # fitted = tofit(freqs, fit[0][0], fit[0][1] )
            # x_fitted =  freqs * fitted
            # b = np.trapz(x_fitted, x) 
            # b *= np.log(10)
            b = 4 * np.pi * (fit[0][0])**2 * sigma * (fit[0][1])**4
            temp[i] = fit[0][1]
            radius[i] = fit[0][0]
            Blue[i]= b
            
        if save:
           with open('data/blue/bluedata_m' + str(m) + '.txt', 'w') as f:
                f.write('# Fitted quantities for snapshots '+ str(snapshots) + '\n#Temperature \n')
                f.write(' '.join(map(str, temp)) + '\n')
                f.write('# Radius \n')
                f.write(' '.join(map(str, radius)) + '\n')
                f.write('# Bolometric L \n')
                f.write(' '.join(map(str, Blue)) + '\n')
                f.close()
                
    if plot:
        fig, axs = plt.subplots(2,3, tight_layout = True)
        axs2 = []
        for i in range(2):
            for j in range(3):
                axs2.append(axs[i,j])
              
        for i, ax in enumerate(axs2):
            if i == 5:
                break
            Lums = data[i] #data[i + len(data) - 4] # NOTE: last 4 because we save on top all the time and should fix that
            Lums_fit = Lums[freq_min_idx:freq_max_idx]
            fit = curve_fit(tofit, fit_freqs, Lums_fit, p0 = (init_R, init_T))
            
            # Plot
            ax.scatter(fit_freqs, Lums_fit, c = 'coral', label = 'Fititng points', s = 4)
            fitted = tofit(freqs, fit[0][0], fit[0][1] )
            ax.plot(freqs, fitted,
                      color = 'limegreen', label = 'Fitted')
            ax.plot(freqs, Lums,
                     color = 'royalblue', linestyle = 'dashed', label = 'Spectrum')
            
            trial = [ tofit(n, init_R, init_T ) for n in freqs ]
            ax.plot(freqs, trial,
                     color = 'grey', label = 'Initial Guess', linestyle = '--')
            ax.grid()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\nu$ [Hz]')
            ax.set_ylabel(r'$L_\nu$')
            ax.legend(fontsize = 4)
            ax.set_ylim(1e17,1e30)
        if save: 
            plt.savefig('Final_plot/Fit_m' + str(m) + '.png')
        plt.show()