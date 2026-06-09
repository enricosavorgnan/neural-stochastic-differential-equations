#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================

#SBATCH --job-name=LatentSDE
#SBATCH --output=log/latent_sde_train_%j.log
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
## SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# 1. Load system modules (Check if they are needed!)
# module load cuda/11.8
# module load python/3.10

# 2. Navigate to the repository root directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/../.."
fi

echo "Working directory set to: $(pwd)"

# 3. Activate the virtual environment (.venv)
if [ -d ".venv" ]; then
    echo "Activating virtual environment from .venv..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment from venv..."
    source venv/bin/activate
else
    echo "WARNING: Virtual environment (.venv/venv) not found at root! Using system python."
fi

# ==============================================================================
# CONFIGURATION & JOB EXECUTION
# ==============================================================================

# 1) Lorenz Model config
CONFIG_FILE="code/config/latent_sde/config_lorenz.yaml"
# 2) Climate Model config
# CONFIG_FILE="code/config/latent_sde/config_climate.yaml"

echo "Training model using configuration: ${CONFIG_FILE}"

# Run the training script.
python -u code/src/latent_sde.py --config "${CONFIG_FILE}" --train

echo "Job completed successfully."
