#!/usr/bin/env bash

# Fail on errors and make ERR trap work through functions/subshells
set -Ee
set -o pipefail

# Print a helpful message if *anything* fails unexpectedly
trap 'echo "❌ Error on line $LINENO while running: $BASH_COMMAND" >&2' ERR

# ---------------------------------------
# Parse arguments
# ---------------------------------------
DO_CLEAN=false
export GRID_NAME=""
while [[ "${1:-}" != "" ]]; do
  case "$1" in
    -c|--clean)
      DO_CLEAN=true
      ;;
    --grid-name)
      shift
      GRID_NAME="${1:-}"
      if [[ -z "$GRID_NAME" || "$GRID_NAME" == -* ]]; then
        echo "Error: --grid-name requires a non-empty argument." >&2
        exit 2
      fi
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--clean] --grid-name <name> --config-root <name>" >&2
      exit 2
      ;;
  esac
  shift || true
done

# ---------------------------------------
# Enforce required arguments
# ---------------------------------------
if [[ -z "$GRID_NAME" ]]; then
  echo "Error: missing required argument(s)." >&2
  echo "  --grid-name <name> is required." >&2
  echo "Usage: $0 [--clean] --grid-name <name>" >&2
  exit 2
fi

export ROMS_CONFIG_ROOT=$(python -c "import config; print(config.paths.model_config)")
export INPUT_DATA_PATH=$(python -c "import config; print(config.paths.input_data)")/${GRID_NAME}
#export BUILD_MODE=debug

# ---------------------------------------
# roots
# ---------------------------------------
export CODES_ROOT="${CODES_ROOT:-/Users/${USER}/codes}"
export ROMS_ROOT="${ROMS_ROOT:-${CODES_ROOT}/ucla-roms}"
export MARBL_ROOT="${MARBL_ROOT:-${CODES_ROOT}/MARBL}"

# Conda environment name
export ROMS_CONDA_ENV="${ROMS_CONDA_ENV:-roms_env}"

# Logs dir
export LOGS="${PWD}/logs"

mkdir -p "${LOGS}"
mkdir -p "${CODES_ROOT}"

# ---------------------------------------
# Helper: run a command, log stdout+stderr, and fail loudly with context
# usage: run_logged "Label" /path/to/logfile cmd arg1 arg2 ...
# ---------------------------------------
run_logged () {
  local label="$1"; shift
  local logfile="$1"; shift
  echo "[${label}] starting..."
  if ! ("$@" > "${logfile}" 2>&1); then
    echo "❌ ${label} FAILED — see log: ${logfile}" >&2
    echo "---- Last 50 lines of ${logfile} ----" >&2
    tail -n 50 "${logfile}" >&2 || true
    echo "-------------------------------------" >&2
    exit 1
  fi
  echo "[${label}] OK"
}

# ---------------------------------------
# Conda activation inside scripts
# ---------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Install Miniforge/Conda first." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"

# ---------------------------------------
# Get codes
# ---------------------------------------

# ROMS
if [ ! -d "${ROMS_ROOT}/.git" ]; then
  git clone https://github.com/CWorthy-ocean/ucla-roms.git "${ROMS_ROOT}"
fi

# MARBL
if [ ! -d "${MARBL_ROOT}/.git" ]; then
  git clone https://github.com/marbl-ecosys/MARBL.git "${MARBL_ROOT}"
fi
pushd "${MARBL_ROOT}" >/dev/null
git fetch --tags
git checkout marbl0.45.0
popd >/dev/null

# ---------------------------------------
# Create & activate the ROMS conda env
# ---------------------------------------
cd "${ROMS_ROOT}"
if ! conda env list | awk '{print $1}' | grep -qx "${ROMS_CONDA_ENV}"; then
  conda env create -f environments/conda_environment.yml --name "${ROMS_CONDA_ENV}"
fi
conda activate "${ROMS_CONDA_ENV}"

# Early toolchain checks (fail fast if missing)
command -v gfortran >/dev/null 2>&1 || { echo "❌ gfortran not found in env ${ROMS_CONDA_ENV}" >&2; exit 1; }
command -v mpifort  >/dev/null 2>&1 || { echo "❌ mpifort not found in env ${ROMS_CONDA_ENV}"  >&2; exit 1; }

# ---------------------------------------
# Core env vars for ROMS build
# ---------------------------------------
export MPIHOME="${CONDA_PREFIX}"
export NETCDFHOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${NETCDFHOME}/lib"
export PATH="${ROMS_ROOT}/Tools-Roms:${PATH}"

# Double-check ROMS_ROOT points to this clone
if [ ! -d "${ROMS_ROOT}/src" ]; then
  echo "ERROR: ROMS_ROOT does not look correct: ${ROMS_ROOT}" >&2
  exit 1
fi

export MARBL_ROOT  # export for children
if [ ! -d "${MARBL_ROOT}/src" ]; then
  echo "ERROR: MARBL_ROOT/src not found at ${MARBL_ROOT}" >&2
  exit 1
fi

# Choose make target based on compiler detected by mpifort
COMPILER_KIND="gnu"
if mpifort --version 2>/dev/null | grep -qiE "ifx|ifort|Intel"; then
  COMPILER_KIND="intel"
fi

# ---------------------------------------
# Builds (with optional --clean)
# ---------------------------------------

# MARBL
if ${DO_CLEAN}; then
  echo "[Clean] MARBL/src ..."
  make -C "${MARBL_ROOT}/src" clean || true
fi
run_logged "Build MARBL (compiler: ${COMPILER_KIND})" "${LOGS}/build.marbl" \
  make -C "${MARBL_ROOT}/src" "${COMPILER_KIND}" USEMPI=TRUE

# NHMG (optional nonhydrostatic lib)
if ${DO_CLEAN}; then
  echo "[Clean] NHMG/src ..."
  make -C "${ROMS_ROOT}/NHMG/src" clean || true
fi
run_logged "Build NHMG/src" "${LOGS}/build.NHMG" \
  make -C "${ROMS_ROOT}/NHMG/src"

# Tools-Roms (utilities like ncjoin, etc.)
if ${DO_CLEAN}; then
  echo "[Clean] Tools-Roms ..."
  make -C "${ROMS_ROOT}/Tools-Roms" clean || true
fi
run_logged "Build Tools-Roms" "${LOGS}/build.Tools-Roms" \
  make -C "${ROMS_ROOT}/Tools-Roms"

# ROMS Application/Case
if ${DO_CLEAN}; then
  echo "[Clean] ROMS (${ROMS_CONFIG_ROOT}) ..."
  make -C "${ROMS_CONFIG_ROOT}" clean || true
fi
pushd ${ROMS_CONFIG_ROOT} >/dev/null
run_logged "Build ROMS (${ROMS_CONFIG_ROOT})" "${LOGS}/build.ROMS" \
  make
popd >/dev/null

echo
echo "✅ All builds completed."
echo "• ROMS root:        ${ROMS_ROOT}"
echo "• MARBL root:       ${MARBL_ROOT}"
echo "• App root:         ${ROMS_CONFIG_ROOT}"
echo "• Logs:             ${LOGS}"
echo
echo "To run your case:"
echo "  cd \"${ROMS_CONFIG_ROOT}\""
echo "  mpirun -n 6 ./roms <your_input>.in"
