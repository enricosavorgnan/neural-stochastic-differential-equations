#!/bin/bash

# ==============================================================================
# SLURM RESOURCE ALLOCATION (SBATCH Directives)
# ==============================================================================
# All lines starting with '#SBATCH' are active commands interpreted by Slurm.
# Lines starting with '##SBATCH' (double hash) or standard '#' comments are ignored.
# Modify these parameters below to change the resources allocated for your job.

# --job-name: A custom name for your job. It will appear in the output of 'squeue'.
#SBATCH --job-name=train_latent_sde

# --output: File to write standard output (stdout) and standard error (stderr).
# '%j' will be automatically replaced by the unique Slurm Job ID.
#SBATCH --output=latent_sde_train_%j.log

# --partition: The queue/partition you want to submit your job to.
# Commonly 'gpu' on clusters with GPUs. If your cluster has a different partition name,
# change it here (e.g., 'main', 'compute', or a specific GPU partition name).
#SBATCH --partition=gpu

# --nodes: Number of compute nodes to request. For standard single-node training, keep this as 1.
#SBATCH --nodes=1

# --ntasks: Number of tasks to run. Since this is a single process training job, keep this as 1.
#SBATCH --ntasks=1

# --cpus-per-task: Number of CPU cores allocated to this task.
# Pytorch can use multiple cores for data loading (DataLoader num_workers) or CPU operations.
# Modify this depending on the number of workers/cores you need (typically 4 to 8 is a good default).
#SBATCH --cpus-per-task=4

# --gres: Generic Resource Scheduling. Request GPUs here.
# format: gpu:type:number or just gpu:number.
# Here we request 1 GPU. If you don't need a GPU, comment this line out.
#SBATCH --gres=gpu:1

# --mem: Total RAM memory requested for the job.
# Can be specified in Megabytes (e.g., 8000M) or Gigabytes (e.g., 16G).
#SBATCH --mem=16G

# --time: Walltime limit for the job. format: D-HH:MM:SS or HH:MM:SS.
# The job will be terminated if it runs longer than this limit.
# Adjust this based on how long you expect your training to take.
# Here we request 2 hours, which is plenty for 200 iterations.
#SBATCH --time=02:00:00

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

# 1. Load system modules (Uncomment and edit if your cluster requires loading modules)
# e.g., loading Python and CUDA environments:
# module load cuda/11.8
# module load python/3.10

# 2. Navigate to the repository root directory
# Since this script is stored in ./code/src/, we navigate 2 directories up to reach
# the project root. This ensures that relative paths defined inside the YAML config
# files (e.g., "./code/models/...") are resolved correctly.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

echo "Working directory set to: $(pwd)"

# 3. Activate the virtual environment (.venv)
# Check if the virtual environment exists in the root folder and activate it.
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

# Choose the model config to train by uncommenting one of the lines below:
# 1) Lorenz Model config
CONFIG_FILE="code/config/latent_sde/config_lorenz.yaml"
# 2) Climate Model config
# CONFIG_FILE="code/config/latent_sde/config_climate.yaml"

# NOTE ON GPU USAGE:
# To actually utilize the GPU requested via SBATCH above, ensure that the
# 'device' parameter under the 'training' section in the chosen YAML config file
# is set to 'cuda' (or 'cuda:0') instead of 'cpu'.
# You can check or edit this in:
#   - code/config/latent_sde/config_lorenz.yaml
#   - code/config/latent_sde/config_climate.yaml

echo "Training model using configuration: ${CONFIG_FILE}"

# Run the training script.
# We run it relative to the root directory, pointing to the script in code/src/
python code/src/latent_sde.py --config "${CONFIG_FILE}" --train

echo "Job completed successfully."
