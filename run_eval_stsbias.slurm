#!/bin/sh

##  An SBATCH command is "commented out" (ignored) if line begins with 2 "#"
##  Anything in <angle brackets> needs to be modified by you (remove <> too)

## Specify a name for Slurm to use when displaying your job.
#SBATCH --job-name=stsbias

## Select a partition (queue) to submit your job to.
##SBATCH --partition=normal
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

## Deal with output and errors.  Separate into 2 files (not the default).
## May help to put your result files in a directory: e.g. /scratch/%u/logs/...
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=jhus@gmu.edu           # Put your GMU email address here

## This script will request 1 CPU and 2GB of memory

## Load relevant modules needed for the job
module load gnu10
module load python
module load cuda
module load cudnn

## Run your program or script
source ~/cs782/fuzzy-goofballs/.venv-cs782/bin/activate
cd /home/jhus/cs782/fuzzy-goofballs

set -x;
set -e;
REPO_PATH="/home/jhus/cs782/"
export PYTHONPATH=$REPO_PATH

LOG_DIR="/scratch/jhus/eval-sts.log"

## NLIBias Finetuning
python train_stsbias.py --task eval --model /scratch/jhus/test-trainer-sts/checkpoint-2000/  > $LOG_DIR
