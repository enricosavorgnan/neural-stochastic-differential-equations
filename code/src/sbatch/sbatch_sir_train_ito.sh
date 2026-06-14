#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================

#SBATCH --job-name=LatentSDE
#SBATCH --output=log/latent_sde_train_%j.log
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    cd "$(dirname "$0")/../../.."
fi

echo "Working directory set to: $(pwd)"

# 2. Activate or Create the virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating .venv..."
    # Create venv. We use python3 to ensure it uses the module loaded by the cluster.
    python3 -m venv .venv
    python -m ensurepip --upgrade --default-pip
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r code/src/requirements.txt
fi

echo "Activating virtual environment from .venv..."
source .venv/bin/activate

# ==============================================================================
# CONFIGURATION & JOB EXECUTION
# ==============================================================================

# 1) Lorenz Model config
CONFIG_FILE="code/config/latent_sde/config_sir_train_ito.yaml"

echo "Training model using configuration: ${CONFIG_FILE}"

# Run the training script.
python -u code/src/latent_sde.py --config "${CONFIG_FILE}" --train

echo "Job completed successfully."
