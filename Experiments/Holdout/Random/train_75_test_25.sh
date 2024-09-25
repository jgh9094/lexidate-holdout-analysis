#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-160%16
#SBATCH --cpus-per-task=9
#SBATCH -t 48:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=ran-75-25
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
###############################################################

# load conda environment
source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.10

# Define the output directory
DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Holdout/Random/train_75_test_25/
mkdir -p ${DATA_DIR}

# mandatory variables
SPLIT_SELECT=0.25
SCHEME=random
SPLIT_OFFSET=2000
REP_OFFSET=0
SEED=$((SLURM_ARRAY_TASK_ID + SPLIT_OFFSET + REP_OFFSET))

##################################
# Treatments
##################################

TASK_146818_MIN=1
TASK_146818_MAX=20

TASK_168784_MIN=21
TASK_168784_MAX=40

TASK_190137_MIN=41
TASK_190137_MAX=60

TASK_359969_MIN=61
TASK_359969_MAX=80

TASK_359934_MIN=81
TASK_359934_MAX=100

TASK_359945_MIN=101
TASK_359945_MAX=120

TASK_359948_MIN=121
TASK_359948_MAX=140

TASK_359939_MIN=141
TASK_359939_MAX=160

##################################
# Conditions
##################################

if [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_146818_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_146818_MAX} ] ; then
  TASK_ID=146818
  TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_168784_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_168784_MAX} ] ; then
  TASK_ID=168784
  TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_190137_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_190137_MAX} ] ; then
    TASK_ID=190137
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359969_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359969_MAX} ] ; then
    TASK_ID=359969
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359934_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359934_MAX} ] ; then
    TASK_ID=359934
    TASK_TYPE=0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359945_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359945_MAX} ] ; then
    TASK_ID=359945
    TASK_TYPE=0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359948_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359948_MAX} ] ; then
    TASK_ID=359948
    TASK_TYPE=0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359939_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359939_MAX} ] ; then
    TASK_ID=359939
    TASK_TYPE=0
else
  echo "${SEED} from ${TASK_ID} and ${SCHEME} failed to launch" >> /home/hernandezj45/Repos/lexidate-variation-analysis/failtolaunch.txt
fi

# let it rip

python /home/hernandezj45/Repos/lexidate-variation-analysis/Source/hold_out_exp.py \
-split_select ${SPLIT_SELECT} \
-scheme ${SCHEME} \
-task_id ${TASK_ID} \
-n_jobs 9 \
-savepath ${DATA_DIR} \
-seed ${SEED} \
-task_type ${TASK_TYPE} \