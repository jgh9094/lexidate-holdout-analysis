#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --cpus-per-task=12
#SBATCH -t 72:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=t-rr
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
###############################################################

# load conda environment
source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.10

# Define the output directory
DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Tournament/learn_50_select_50/
mkdir -p ${DATA_DIR}

# mandatory variables
SPLIT_SELECT=0.50
SCHEME=tournament
SPLIT_OFFSET=1000
EXP_OFFSET=3000
S=481
SEED=$((S + SPLIT_OFFSET + EXP_OFFSET))
TASK_ID=168757
TASK_TYPE=1

# let it rip
python /home/hernandezj45/Repos/lexidate-variation-analysis/Source/experiment.py \
-split_select ${SPLIT_SELECT} \
-scheme ${SCHEME} \
-task_id ${TASK_ID} \
-n_jobs 12 \
-savepath ${DATA_DIR} \
-seed ${SEED} \
-task_type ${TASK_TYPE} \