#!/bin/bash
#SBATCH --job-name=down
#SBATCH --output=%x.out
#SBATCH --mail-user="martire@strw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu_strw"
#SBATCH --account="gpu_strw"
#SBATCH --time=5-0:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

module load Python/3.10.4-GCCcore-11.3.0
module load rclone

# Place and run this in the directory where you want the files to be downloaded
# Specify the simulation to be downloaded
radius=0.47
starmass=0.5
bhmass=10000 # don't use scientific notation here.
suffix="beta1S60n1.5ComptonNewAMR"

simulation="R"$radius"M"$starmass"BH"$bhmass""$suffix"" 

# Specify the desired snapsots, check which are available with rclone lsl
first_snap=248
last_snap=260

# Make the dir
# mkdir $simulation

# For downloading all snapshots
# rclone copy gdrive:/Snellius/$simulation/ . -P --include "snap_full_*.h5" 

# Download & place into seperete dirs.
for  (( d=$first_snap; d<=$last_snap; d++ )) 
do
    # For downloading specific snapshots. Slow.
    # rclone copy gdrive:/Snellius/$simulation/snap_full_"$d".h5 . -P 
    mkdir $simulation/snap_"$d"
    mv /snap_full_"$d".h5 "$simulation"/snap_"$d"/snap_"$d".h5   
done                                                                 


echo "Job's done!"