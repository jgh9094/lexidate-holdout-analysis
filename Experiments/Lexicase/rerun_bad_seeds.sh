#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1,2,3,4
#SBATCH --cpus-per-task=10
#SBATCH -t 72:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=l-rr
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
###############################################################

# load conda environment
source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.10

# check if array task id is 1
if [ $SLURM_ARRAY_TASK_ID -eq 1 ] ; then
    DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Lexicase/learn_05_select_95/
    SPLIT_SELECT=0.95
    SCHEME=lexicase
    SPLIT_OFFSET=0
    EXP_OFFSET=10000
    S=481
    SEED=$((S + SPLIT_OFFSET + EXP_OFFSET))
    TASK_ID=359954

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ] ; then
    DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Lexicase/learn_05_select_95/
    SPLIT_SELECT=0.95
    SCHEME=lexicase
    SPLIT_OFFSET=0
    EXP_OFFSET=10000
    S=482
    SEED=$((S + SPLIT_OFFSET + EXP_OFFSET))
    TASK_ID=359955

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ] ; then
    DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Lexicase/learn_50_select_50/
    SPLIT_SELECT=0.50
    SCHEME=lexicase
    SPLIT_OFFSET=1000
    EXP_OFFSET=10000
    S=481
    SEED=$((S + SPLIT_OFFSET + EXP_OFFSET))
    TASK_ID=168757

elif [ $SLURM_ARRAY_TASK_ID -eq 4 ] ; then
    DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Lexicase/learn_95_select_05/
    SPLIT_SELECT=0.05
    SCHEME=lexicase
    SPLIT_OFFSET=2000
    EXP_OFFSET=10000
    S=481
    SEED=$((S + SPLIT_OFFSET + EXP_OFFSET))
    TASK_ID=168784
fi

# let it rip
python /home/hernandezj45/Repos/lexidate-variation-analysis/Source/experiment.py \
-split_select ${SPLIT_SELECT} \
-scheme ${SCHEME} \
-task_id ${TASK_ID} \
-n_jobs 10 \
-savepath ${DATA_DIR} \
-seed ${SEED} \