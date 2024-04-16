import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import matplotlib.pyplot as plt
import h5py

def extractor(filename):
    '''
    Loads the file, extracts quantites
    '''
    # Read File
    f = h5py.File(filename, "r")
    key = 'rank0' #the only other keys are Cycle, Time and Box
    
    X = []
    Y = []
    Z = []
    Vol = []
    Entropy = []
    x_data = f[key]['CMx']
    y_data = f[key]['CMy']
    z_data = f[key]['CMz']
    vol_data = f[key]['Volume']
    entropy_data = f[key]['tracers']['Entropy']

    for i in range(len(x_data)):
        X.append(x_data[i])
        Y.append(y_data[i])
        Z.append(z_data[i])
        Vol.append(vol_data[i])
        Entropy.append(entropy_data[i])
    
    f.close()
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    Vol = np.array(Vol)
    Entropy = np.array(Entropy)

    return X, Y, Z, Vol, Entropy

if __name__ == '__main__':
    name = '196'
    path = 'TDE'
    path = f'{path}/{name}/'
    X, Y, Z, Vol, Entropy = extractor(f'TDE/196/snap_grad_196.h5')
    
    # select cells at midplane
    dim_cell = Vol**(1/3)
    X_cross = X[np.abs(Z) < dim_cell]
    Y_cross = Y[np.abs(Z) < dim_cell]
    Entropy_cross = Entropy[np.abs(Z) < dim_cell] 

    fig, ax = plt.subplots(1,1, figsize = (12,7))
    img = ax.scatter(X_cross, Y_cross, c = Entropy_cross, marker= 's', s = 4, cmap = 'jet', vmin = 7e-7, vmax = 2e-6)

    cbar = plt.colorbar(img)#, format='%.0e')
    cbar.set_label(r'Entropy', fontsize = 16)
    # ax.set_xlim(-30,30)
    # ax.set_ylim(-20,20)
    ax.set_xlabel(r'X [$R_\odot$]', fontsize = 18)
    ax.set_ylabel(r'Y [$R_\odot$]', fontsize = 18)
    plt.grid()
    plt.show()

