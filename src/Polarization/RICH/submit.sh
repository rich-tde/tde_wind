#!/usr/bin/env bash
#SBATCH --job-name=tde_2000
#SBATCH --output=output_2000.txt
#SBATCH --error=error_2000.txt
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
# surface) live in test.cpp and are compiled in; the knobs below override
# test.cpp at run time, and extra arguments override the knobs.

set -euo pipefail

# ----------------------------------------------------------------------------
# STATISTICS KNOBS
#
# GENERATIONS is the number of final (statistics) generations. The result is
# the average over them, so the statistical error scales as 1/sqrt(GENERATIONS).
# Each pass adds 21 fixed burn-in/probe generations in front. With the budgets
# and exploration fractions below (both 0.005) the total run time is roughly
# 12 min + GENERATIONS * 26 s (20 -> ~20 min, 75 -> ~45 min, 200 -> ~1 h 40).
# Raise --time above when raising GENERATIONS.
GENERATIONS=75

# Packets per generation for the cells that were learned to produce escaping
# light: average packets per learned cell (budget) and the cap per cell.
# MG: 1600/cell -> ~26M learned + ~10M exploration packets = ~36M per generation.
MG_LEARNED_BUDGET=1600
GREY_LEARNED_BUDGET=500
LEARNED_MAX_PER_CELL=50000

# Exploration-packet weight cap: a thick cell outside the learned set is split
# into packets of at most FRACTION x (last generation's escaping energy). A
# smaller fraction means lighter, more numerous packets: the variance from
# those cells scales with the fraction, the packet count with 1/fraction.
# 0.005 vs 0.05 (test.cpp default): MG sigma_P median 3.3x lower at 200
# generations for ~10% more MG time; grey weakest observers ~10x more effective
# packets for ~2x grey time. Below 0.005 the packet count (~100M MG, ~440M grey
# per generation at 0.005) approaches the node memory limit.
MG_EXPLORATION_FRACTION=0.005
GREY_EXPLORATION_FRACTION=0.005
# ----------------------------------------------------------------------------


# Inputs
SNAPSHOT=/home/pmartire/tde_wind/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/snap_151/snap_151.h5
# SNAPSHOT=/home/pmartire/tde_wind/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonMG/snap_429/snap_429.h5

# Under sbatch the script runs from a copy in the SLURM spool directory, so
# BASH_SOURCE does not point here; use the submission directory instead.
## In Elad script:
# if [[ -n ${SLURM_SUBMIT_DIR:-} ]]; then
#     here=$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)
# else
#     here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# fi
##
here=/home/pmartire/RICH/build/gnuReleaseMPI
root=/home/pmartire/RICH #$(cd -- "$here/../.." && pwd -P)          # repository root (data/ lives there)
executable="$here/rich"                         # copied here by the user

for f in "$executable" "$SNAPSHOT" "$root/data/STA/MG/frequency_edges.txt" \
         "$root/data/STA/planck.txt" "$root/data/EOS/Tfile.txt"; do
    [[ -e $f ]] || { echo "missing: $f" >&2; exit 2; }
done
[[ -x $executable ]] || { echo "not executable: $executable" >&2; exit 2; }

# The executable links VTK, HDF5 and OpenMPI from the module stack.
# command -v ml >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module restore rich_gnu_2025 # I added this line
ml restore gcc

export RICH_MEASURED_LB_DEBUG_MEMORY=1   # per-rank memory lines in error_<jobid>.txt

echo "exe:         $executable"
echo "snapshot:    $SNAPSHOT"
echo "tasks:       ${SLURM_NTASKS:-1}"
echo "generations: $GENERATIONS  (MG budget $MG_LEARNED_BUDGET, grey $GREY_LEARNED_BUDGET, max $LEARNED_MAX_PER_CELL)"
echo "exploration: MG fraction $MG_EXPLORATION_FRACTION, grey fraction $GREY_EXPLORATION_FRACTION"

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
    --volume-emission.exploration-weight-fraction "$MG_EXPLORATION_FRACTION" \
    --volume-emission.exploration-weight-fraction-grey "$GREY_EXPLORATION_FRACTION" \
    "$@"
