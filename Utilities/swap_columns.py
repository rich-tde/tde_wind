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

import pandas as pd

m = 4
Mbh = 10**4
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'HiResNewAMR'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

# Load CSV
df = pd.read_csv(f'{abspath}/data/{folder}/wind/Mdot{with_who}{n_obs}Sec_{check}{which_r_title}dark_bright_z.csv')

# Get column names as a list
cols = list(df.columns)

# Swap 3rd and 4th columns (index 2 and 3)
cols[2], cols[3] = cols[3], cols[2]

# Reorder dataframe
df = df[cols]

# Overwrite original file
df.to_csv(f'{abspath}/data/{folder}/wind/Mdot{with_who}{n_obs}Sec_{check}{which_r_title}dark_bright_z.csv', index=False)