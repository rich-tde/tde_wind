#!/usr/bin/env bash
#SBATCH --job-name=tde_multi
#SBATCH --output=output_multi_%j.txt
#SBATCH --error=error_multi_%j.txt
#SBATCH --nodes=8
#SBATCH --ntasks=1536
#SBATCH --exclusive
#SBATCH --partition=genoa
#SBATCH --time=2-10:00:00

#SBATCH --mail-user="martire@strw.leidenuniv.nl"
#SBATCH --mail-type=TIME_LIMIT_50,TIME_LIMIT_90,ALL


# Grey + multigroup polarization post-process of multiple Snapshot3D files.

# Physics settings (Fleck factor 1, emission from every cell outside the deep
# surface) live in test.cpp and are compiled in; the knobs below override
# test.cpp at run time, and extra arguments override the knobs.

set -euo pipefail

# ----------------------------------------------------------------------------
# STATISTICS KNOBS
#
# GENERATIONS is the number of final (statistics) generations. The result is
# the average over them, so the statistical error scales as 1/sqrt(GENERATIONS).
# Each pass adds 21 fixed burn-in/probe generations in front.
# ----------------------------------------------------------------------------

GENERATIONS=2500

# Snapshots to run sequentially inside this Slurm job.
SNAPS=(250 320 429)

# Packets per generation for the cells that were learned to produce escaping
# light: average packets per learned cell (budget) and the cap per cell.
MG_LEARNED_BUDGET=1600
GREY_LEARNED_BUDGET=500
LEARNED_MAX_PER_CELL=50000

# Exploration-packet weight cap.
MG_EXPLORATION_FRACTION=0.005
GREY_EXPLORATION_FRACTION=0.005

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------

here=/home/pmartire/RICH/build/gnuReleaseMPI
root=/home/pmartire/RICH
executable="$here/rich"

snapshot_root="/home/pmartire/tde_wind/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonMG"

# ----------------------------------------------------------------------------
# ENVIRONMENT
# ----------------------------------------------------------------------------

module restore rich_gnu_2025

export RICH_MEASURED_LB_DEBUG_MEMORY=1
export UCX_TLS=ib

# ----------------------------------------------------------------------------
# BASIC CHECKS
# ----------------------------------------------------------------------------

[[ -x $executable ]] || {
    echo "not executable: $executable" >&2
    exit 2
}

for f in \
    "$root/data/STA/MG/frequency_edges.txt" \
    "$root/data/STA/planck.txt" \
    "$root/data/EOS/Tfile.txt"
do
    [[ -e $f ]] || {
        echo "missing: $f" >&2
        exit 2
    }
done

cd "$here"

echo "======================================================"
echo "Starting multi-snapshot polarization job"
echo "Executable:   $executable"
echo "Snapshots:    ${SNAPS[*]}"
echo "Tasks:        ${SLURM_NTASKS:-1}"
echo "Generations:  $GENERATIONS"
echo "MG budget:    $MG_LEARNED_BUDGET"
echo "Grey budget:  $GREY_LEARNED_BUDGET"
echo "Max/cell:     $LEARNED_MAX_PER_CELL"
echo "MG explore:   $MG_EXPLORATION_FRACTION"
echo "Grey explore: $GREY_EXPLORATION_FRACTION"
echo "======================================================"

# ----------------------------------------------------------------------------
# RUN SNAPSHOTS
# ----------------------------------------------------------------------------

for SNAP in "${SNAPS[@]}"
do

    SNAPSHOT="$snapshot_root/snap_$SNAP/snap_$SNAP.h5"

    echo
    echo "======================================================"
    echo "Starting snapshot $SNAP"
    echo "Input: $SNAPSHOT"
    echo "======================================================"

    [[ -e $SNAPSHOT ]] || {
        echo "missing snapshot: $SNAPSHOT" >&2
        exit 2
    }

    mkdir -p "output$SNAP"

    srun \
        --output="output_${SNAP}.txt" \
        --error="error_${SNAP}.txt" \
        "$executable" \
        --input.snapshot "$SNAPSHOT" \
        --output.stem "output$SNAP/tde_${GENERATIONS}polarization$SNAP" \
        --input.multigroup-opacity-directory "$root/data/STA/MG/" \
        --input.grey-opacity-directory "$root/data/STA/" \
        --input.eos-directory "$root/data/EOS/" \
        --flux-source.face-flux hydro \
        --flux-source.spectrum-temperature radiation \
        --transport.split-observer-fraction 0.1 \
        --transport.split-escape-reference 0.1 \
        --transport.split-per-group true \
        --transport.split-group-scale 1 \
        --transport.split-group-floor 1e-3 \
        --transport.split-copy-budget 1000000 \
        --load-balance.final-period 5 \
        --transport.generations "$GENERATIONS" \
        --volume-emission.learned-photons-per-cell-budget "$MG_LEARNED_BUDGET" \
        --volume-emission.learned-photons-per-cell-budget-grey "$GREY_LEARNED_BUDGET" \
        --volume-emission.learned-max-photons "$LEARNED_MAX_PER_CELL" \
        --volume-emission.exploration-weight-fraction "$MG_EXPLORATION_FRACTION" \
        --volume-emission.exploration-weight-fraction-grey "$GREY_EXPLORATION_FRACTION" \
        "$@"

    echo "Finished snapshot $SNAP"

done

echo
echo "======================================================"
echo "All snapshots completed successfully"
echo "======================================================"