#!/bin/bash
########## Define Resources Needed with SBATCH Lines ##########
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-480%16
#SBATCH --cpus-per-task=9
#SBATCH -t 72:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=t-8-2
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040
###############################################################

# load conda environment
source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.10

# Define the output directory
DATA_DIR=/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Tournament/learn_80_select_20/
mkdir -p ${DATA_DIR}

# mandatory variables
SPLIT_SELECT=0.20
SCHEME=tournament
SPLIT_OFFSET=1500
EXP_OFFSET=3000
SEED=$((SLURM_ARRAY_TASK_ID + SPLIT_OFFSET + EXP_OFFSET))

##################################
# Treatments
##################################

TASK_359953_MIN=1
TASK_359953_MAX=30

TASK_146818_MIN=31
TASK_146818_MAX=60

TASK_359954_MIN=61
TASK_359954_MAX=90

TASK_359955_MIN=91
TASK_359955_MAX=120

TASK_190146_MIN=121
TASK_190146_MAX=150

TASK_168757_MIN=151
TASK_168757_MAX=180

TASK_359956_MIN=181
TASK_359956_MAX=210

TASK_359957_MIN=211
TASK_359957_MAX=240

TASK_359958_MIN=241
TASK_359958_MAX=270

TASK_359959_MIN=271
TASK_359959_MAX=300

TASK_2073_MIN=301
TASK_2073_MAX=330

TASK_10090_MIN=331
TASK_10090_MAX=360

TASK_359960_MIN=361
TASK_359960_MAX=390

TASK_168784_MIN=391
TASK_168784_MAX=420

TASK_359961_MIN=421
TASK_359961_MAX=450

TASK_359962_MIN=451
TASK_359962_MAX=480

##################################
# Conditions
##################################

if [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359953_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359953_MAX} ] ; then
  TASK_ID=359953
  TASK_TYPE=1
  echo "TASK_ID: ${TASK_ID}"
  exit 0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_146818_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_146818_MAX} ] ; then
  TASK_ID=146818
  TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359954_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359954_MAX} ] ; then
    TASK_ID=359954
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359955_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359955_MAX} ] ; then
    TASK_ID=359955
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_190146_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_190146_MAX} ] ; then
    TASK_ID=190146
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_168757_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_168757_MAX} ] ; then
    TASK_ID=168757
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359956_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359956_MAX} ] ; then
    TASK_ID=359956
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359957_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359957_MAX} ] ; then
    TASK_ID=359957
    TASK_TYPE=1
    echo "TASK_ID: ${TASK_ID}"
    exit 0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359958_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359958_MAX} ] ; then
    TASK_ID=359958
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359959_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359959_MAX} ] ; then
    TASK_ID=359959
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_2073_MIN} ]   && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_2073_MAX} ] ; then
    TASK_ID=2073
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_10090_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_10090_MAX} ] ; then
    TASK_ID=10090
    TASK_TYPE=1
    echo "TASK_ID: ${TASK_ID}"
    exit 0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359960_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359960_MAX} ] ; then
    TASK_ID=359960
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_168784_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_168784_MAX} ] ; then
    TASK_ID=168784
    TASK_TYPE=1
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359961_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359961_MAX} ] ; then
    TASK_ID=359961
    TASK_TYPE=1
    echo "TASK_ID: ${TASK_ID}"
    exit 0
elif [ ${SLURM_ARRAY_TASK_ID} -ge ${TASK_359962_MIN} ] && [ ${SLURM_ARRAY_TASK_ID} -le ${TASK_359962_MAX} ] ; then
    TASK_ID=359962
    TASK_TYPE=1
else
  echo "${SEED} from ${TASK_ID} and ${SCHEME} failed to launch" >> /home/hernandezj45/Repos/lexidate-variation-analysis/failtolaunch.txt
fi

# let it rip
python /home/hernandezj45/Repos/lexidate-variation-analysis/Source/experiment.py \
-split_select ${SPLIT_SELECT} \
-scheme ${SCHEME} \
-task_id ${TASK_ID} \
-n_jobs 9 \
-savepath ${DATA_DIR} \
-seed ${SEED} \
-task_type ${TASK_TYPE} \