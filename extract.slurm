#!/bin/bash
#SBATCH --job-name=extr_vol
#SBATCH --output=%x.out
#SBATCH --mail-user="martire@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=6-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G

# Load python
module load Python/3.11.3-GCCcore-12.3.0

# Start the env
source /data1/martirep/env_shock/bin/activate

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd shocks/shock_capturing
python3 -m Utilities.old_extractor.py
echo "[$SHELL] ## Script finished"
