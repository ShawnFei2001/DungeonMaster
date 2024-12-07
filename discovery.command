ssh fei.xiao@login.discovery.neu.edu
srun --partition=gpu --nodes=1 --pty --gres=gpu:v100-pcie:1 --ntasks=1 --mem=32GB --time=08:00:00 /bin/bash
module load cuda/12.1
source miniconda3/bin/activate
conda activate finetuning_env