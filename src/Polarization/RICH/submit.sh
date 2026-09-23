#!/usr/bin/env bash
#SBATCH --job-name=tde_800
#SBATCH --output=output_800.txt
#SBATCH --error=error_800.txt
#SBATCH --nodes=8
#SBATCH --ntasks=1536
#SBATCH --exclusive
#SBATCH --partition=genoa
#SBATCH --time=1-10:00:00

#SBATCH --mail-user="martire@strw.leidenuniv.nl"
#SBATCH --mail-type=TIME_LIMIT_50,TIME_LIMIT_90,ALL

#
# Grey + multigroup polarization post-process of one Snapshot3D.
#
# Usage (from this directory, so output_<jobid>.txt lands here):
#     cp ../../build/gnuReleaseMPI/rich ./rich      # after building
#     sbatch submit.sh
# Extra arguments are passed straight to the executable, e.g.
#     sbatch submit.sh --output.stem output/my_run
#
# Physics settings (Fleck factor 1, emission from every cell outside the deep
# surface, exploration-packet weight cap) live in test.cpp and are compiled in.

set -euo pipefail

# ----------------------------------------------------------------------------
# STATISTICS KNOBS
#
# GENERATIONS is the number of final (statistics) generations. The result is
# the average over them, so the statistical error scales as 1/sqrt(GENERATIONS).
# The run adds 21 fixed burn-in/probe generations in front; with the budgets
# below, one MG generation takes ~10 s and one grey generation ~12 s, so the
# total run time is roughly 6 min + GENERATIONS * 22 s (75 -> ~38 min).
# Raise --time above when raising GENERATIONS.
GENERATIONS=800

# Packets per generation for the cells that were learned to produce escaping
# light: average packets per learned cell (budget) and the cap per cell.
# MG: 1600/cell -> ~26M learned + ~10M exploration packets = ~36M per generation.
MG_LEARNED_BUDGET=1600
GREY_LEARNED_BUDGET=500
LEARNED_MAX_PER_CELL=50000
# ----------------------------------------------------------------------------

# Inputs
SNAPSHOT=/home/pmartire/tde_wind/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/snap_151/snap_151.h5

here=/home/pmartire/RICH/build/gnuReleaseMPI #$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=/home/pmartire/RICH #$(cd -- "$here/../.." && pwd -P)           repository root (data/ lives there)
executable="$here/rich"                         # copied here by the user

for f in "$executable" "$SNAPSHOT" "$root/data/STA/MG/frequency_edges.txt" \
         "$root/data/STA/planck.txt" "$root/data/EOS/Tfile.txt"; do
    [[ -e $f ]] || { echo "missing: $f" >&2; exit 2; }
done
[[ -x $executable ]] || { echo "not executable: $executable" >&2; exit 2; }

# The executable links VTK, HDF5 and OpenMPI from the module stack.
#command -v ml >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module restore rich_gnu_2025 # I added this line

export RICH_MEASURED_LB_DEBUG_MEMORY=1   # per-rank memory lines in error_<jobid>.txt

echo "exe:         $executable"
echo "snapshot:    $SNAPSHOT"
echo "tasks:       ${SLURM_NTASKS:-1}"
echo "generations: $GENERATIONS  (MG budget $MG_LEARNED_BUDGET, grey $GREY_LEARNED_BUDGET, max $LEARNED_MAX_PER_CELL)"

cd "$here"
mkdir -p output

exec srun "$executable" \
    --input.snapshot "$SNAPSHOT" \
    --input.multigroup-opacity-directory "$root/data/STA/MG/" \
    --input.grey-opacity-directory "$root/data/STA/" \
    --input.eos-directory "$root/data/EOS/" \
    --transport.generations "$GENERATIONS" \
    --volume-emission.learned-photons-per-cell-budget "$MG_LEARNED_BUDGET" \
    --volume-emission.learned-photons-per-cell-budget-grey "$GREY_LEARNED_BUDGET" \
    --volume-emission.learned-max-photons "$LEARNED_MAX_PER_CELL" \
    "$@"
