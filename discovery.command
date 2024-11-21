ssh fei.xiao@login.discovery.neu.edu
srun --partition=gpu --nodes=1 --gpus=1 --cpus-per-task=4 --pty /bin/bash
srun --partition=gpu --nodes=1 --gres=gpu:v100-pcie:1 --cpus-per-task=4 --mem=32GB --time=02:00:00 --pty /bin/bash

module load cuda/12.1

source miniconda3/bin/activate
conda activate finetuning_env

pip install torch transformers sqlalchemy redis

python game.py