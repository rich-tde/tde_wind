"""
Extracts data from h5 files
"""
import sys
sys.path.append('/Users/paolamartire/shocks')

import numpy as np
import h5py
from datetime import datetime
snapshot881 = "data/sedov_100.h5"

def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a `.h5` file            
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        # print(filename)
        # h5printR(h, '  ')
        a = h['rank1']['Temperature']
        for i in range(len(a)):
            print(a[i])
        
h5print(snapshot881)

    
    
    
    
    
    
    
    
    
    
    
    
    