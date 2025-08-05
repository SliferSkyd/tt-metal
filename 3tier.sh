#!/bin/bash
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --partition=debug
#SBATCH --job-name=nano_gpt_job
#SBATCH --output=nano_gpt_%j.out
#SBATCH --error=nano_gpt_%j.err

# Set environmental variables
export WORKER_COUNT=1
export AGG_COUNT=1
export OPT_COUNT=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# Set paths - adjust these as needed
export TT_METAL_HOME="/data/tt-metal"
export CONFIG="training_shakespear_nanogpt_3tier_full_loudbox.yaml"
export BIN_DIR="$TT_METAL_HOME/tt-train/build/sources/examples/nano_gpt"
export CFG_DIR="$TT_METAL_HOME/tt-train/configs"
# Set run flag (empty by default, can be set to specific flags if needed)
export RUN_FLAG=""

# Run the MPI job
mpirun -np "${WORKER_COUNT}" bash -lc "export TT_METAL_HOME='${TT_METAL_HOME}' && export TT_MESH_ID=0 && export TT_HOST_RANK=0  && \"${BIN_DIR}/nano_gpt\" -c \"${CFG_DIR}/${CONFIG}\"" \
       : -np "${AGG_COUNT}" bash -lc "export TT_METAL_HOME='${TT_METAL_HOME}' && export TT_MESH_ID=0 && export TT_HOST_RANK=0  && \"${BIN_DIR}/nano_gpt_aggregator\" -c \"${CFG_DIR}/${CONFIG}\" -d 8" \
       : -np "${OPT_COUNT}" bash -lc "export TT_METAL_HOME='${TT_METAL_HOME}'  && export TT_MESH_ID=0 && export TT_HOST_RANK=0  && \"${BIN_DIR}/nano_gpt_optimizer\" -c \"${CFG_DIR}/${CONFIG}\" -d 8"
