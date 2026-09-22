#!/usr/bin/env bash
#SBATCH --job-name=tde-gray-mg-pol
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=4
#SBATCH --ntasks=768
#SBATCH --exclusive
#SBATCH --partition=genoa
#SBATCH --time=1-10:00:00

#SBATCH --mail-user="martire@strw.leidenuniv.nl"
#SBATCH --mail-type=TIME_LIMIT_50,TIME_LIMIT_90,ALL


set -euo pipefail
module restore rich_gnu_2025 # I added this line
RICH_POSTPROCESS_SNAPSHOT=/home/pmartire/tde_wind/TDE/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/snap_151/snap_151.h5
RICH_EXECUTABLE=/home/pmartire/RICH/build/gnuReleaseMPI/rich

# These are the only user-set numerical controls for the calculation.
LEARNING_ITERATIONS=21
STATISTICS_ITERATIONS=75

require_integer_at_least()
{
    local name=$1
    local value=$2
    local minimum=$3
    if [[ ! $value =~ ^[0-9]+$ ]] || (( value < minimum )); then
        echo "$name must be an integer >= $minimum (got '$value')" >&2
        exit 2
    fi
}

require_integer_at_least LEARNING_ITERATIONS "$LEARNING_ITERATIONS" 2
require_integer_at_least STATISTICS_ITERATIONS "$STATISTICS_ITERATIONS" 1

: "${RICH_POSTPROCESS_SNAPSHOT:?Export RICH_POSTPROCESS_SNAPSHOT with the input Snapshot3D HDF5 path}"
: "${RICH_EXECUTABLE:?Export RICH_EXECUTABLE with the absolute path to the built MPI executable}"

if [[ $RICH_POSTPROCESS_SNAPSHOT != /* ]]; then
    echo "RICH_POSTPROCESS_SNAPSHOT must be an absolute path" >&2
    exit 2
fi
if [[ $RICH_EXECUTABLE != /* ]]; then
    echo "RICH_EXECUTABLE must be an absolute path" >&2
    exit 2
fi

if [[ -n ${SLURM_SUBMIT_DIR:-} ]]; then
    run_directory=$(cd -- "$SLURM_SUBMIT_DIR" && pwd -P)
else
    run_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
fi
if [[ ${run_directory##*/} != imc_postprocess_tde_gray_mg_polarization ]]; then
    echo "Submit this job from runs/imc_postprocess_tde_gray_mg_polarization" >&2
    exit 2
fi
rich_root=$(cd -- "$run_directory/../../../RICH/" && pwd -P) # it was $(cd -- "$run_directory/../../.." && pwd -P)
sta_multigroup="$rich_root/data/STA/MG/" #/gpfs/home3/pmartire/ # it was: "$rich_root/data/STA/MG/"
sta_gray="$rich_root/data/STA/" 
eos_tables="$rich_root/data/EOS/" 

for required_path in \
    "$RICH_POSTPROCESS_SNAPSHOT" \
    "$RICH_EXECUTABLE" \
    "${sta_multigroup}frequency_edges.txt" \
    "${sta_gray}planck.txt"; do
    if [[ ! -e $required_path ]]; then
        echo "Required input does not exist: $required_path" >&2
        exit 2
    fi
done
if [[ ! -x $RICH_EXECUTABLE ]]; then
    echo "RICH_EXECUTABLE is not executable: $RICH_EXECUTABLE" >&2
    exit 2
fi

if [[ -n ${RICH_MPI_LAUNCHER:-} ]]; then
    command -v "$RICH_MPI_LAUNCHER" >/dev/null 2>&1 || {
        echo "RICH_MPI_LAUNCHER is not executable or in PATH: $RICH_MPI_LAUNCHER" >&2
        exit 2
    }
    mpi_launcher=("$RICH_MPI_LAUNCHER")
elif [[ -n ${SLURM_JOB_ID:-} ]]; then
    command -v srun >/dev/null 2>&1 || {
        echo "This Slurm job requires srun in PATH" >&2
        exit 2
    }
    mpi_launcher=(srun)
elif command -v mpirun >/dev/null 2>&1; then
    mpi_launcher=(mpirun)
elif command -v mpiexec >/dev/null 2>&1; then
    mpi_launcher=(mpiexec)
else
    echo "No MPI launcher found; load MPI or submit through Slurm" >&2
    exit 2
fi

mkdir -p "$run_directory/output"
cd "$run_directory"
"${mpi_launcher[@]}" "$RICH_EXECUTABLE" \
    --input.snapshot "$RICH_POSTPROCESS_SNAPSHOT" \
    --input.multigroup-opacity-directory "$sta_multigroup" \
    --input.grey-opacity-directory "$sta_gray" \
    --input.eos-directory "$eos_tables" \
    --adaptive.source.burnin-generations "$LEARNING_ITERATIONS" \
    --transport.generations "$STATISTICS_ITERATIONS"
