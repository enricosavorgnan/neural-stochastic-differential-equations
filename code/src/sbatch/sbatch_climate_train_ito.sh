#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================

#SBATCH --job-name=LatentSDE
#SBATCH --output=log/latent_sde_train_%j.log
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# 1. Load system modules
# IMPORTANT: On most SLURM clusters, you MUST load a Python module before interacting
# with virtual environments. Uncomment and adjust the line below if your cluster requires it.
# module load python/3.10

# 2. Navigate to the repository root directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/../../.."
fi

echo "Working directory set to: $(pwd)"

# 3. Activate the virtual environment and install dependencies
if [ -d ".venv" ]; then
    echo "Activating virtual environment from .venv..."
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r code/src/requirements.txt
elif [ -d "venv" ]; then
    echo "Activating virtual environment from venv..."
    source venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r code/src/requirements.txt
else
    echo "Virtual environment not found. Creating .venv and installing requirements..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r code/src/requirements.txt
fi

# ==============================================================================
# CONFIGURATION & JOB EXECUTION
# ==============================================================================

# 1) Climate Model config
CONFIG_FILE="code/config/latent_sde/config_climate_train_ito.yaml"

echo "Training model using configuration: ${CONFIG_FILE}"

# Run the training script.
# The -u flag ensures Python output is unbuffered, saving your logs even if the job times out.
python -u code/src/latent_sde.py --config "${CONFIG_FILE}" --train

echo "Job completed successfully."