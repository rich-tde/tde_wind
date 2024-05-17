#!/bin/bash
#SBATCH --job-name=eladython6
#SBATCH --output=%x.out
#SBATCH --mail-user="kilmetis@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=1-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G

# Load python
module load ALICE/AMD
module load Python/3.10.4-GCCcore-11.3.0
module load MATLAB

# Start the env
source python_venvs/amd_base/bin/activate
export LD_LIBRARY_PATH=/easybuild/software/MATLAB/2023b-r4/bin/ginxa64
module load Python/3.10.4-GCCcore-13.3.0
pip install matlabengine==23.2.3

echo "[$SHELL] #### Starting Python Job"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# Run the file
echo "[$SHELL] ## Run script"
cd TDE/tde_comparison
python3 -m src.Luminosity.Eladython
echo "[$SHELL] ## Script finished"
