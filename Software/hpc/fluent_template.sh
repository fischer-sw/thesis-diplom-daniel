#!/bin/bash

# Set number of nodes and tasks per node. Should not exceed number of cores for one node.
# CPU Cores available depend on the chosen queue.
# CPUs per node:
# defq 40
# intel_16 16
# intel_32 32

#SBATCH --ntasks-per-node=%tasks_p_node%
#SBATCH --nodes=1

# Set max wallclock time
#SBATCH --time=%wall_time%

# Use partition defq|intel (the latter the includes former hydra nodes)
#SBATCH -p %queue%

# Set account for accessing a specific cpu time contingent
# Options: default, fwd
#SBATCH -A default

# Set name of the job
#SBATCH -J %job_name%

# Mail alert at BEGIN|END|FAIL|ALL
#SBATCH --mail-type=ALL

# Send mail to the following address
#SBATCH --mail-user=d.fischer@hzdr.de

rm cleanup-fluent*.sh
rm case.log
rm *.trn

LOG_FILE=./case.log

function Log {
    local level=$1
    local msg=$2
    echo $(date --rfc-3339=seconds):${level} ${msg} >> ${LOG_FILE}
}

Log INFO "JOB START"
Log INFO "JOB NAME = ${SLURM_JOB_NAME}"

Log INFO "loading modules"
Log INFO "Loading module fluent ..."
module load fluent/22.1 >> ${LOG_FILE} 2>&1

# Change to execution directory
cd $SLURM_SUBMIT_DIR

# Fix distribution of tasks to nodes as workaround for bug in slurm
# Proposed by Henrik Schulz (FWCI)
export SLURM_TASKS_PER_NODE="$((SLURM_NTASKS / SLURM_NNODES))$( for ((i=2; i<=$SLURM_NNODES; i++)); \
do printf ",$((SLURM_NTASKS / SLURM_NNODES))"; done )"

NODES=$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)
Log INFO "allocated nodes ${NODES}"
Log INFO "SLURM_NTASKS = ${SLURM_NTASKS}"

rm *.out
Log INFO "Starting fluent calculation..."
fluent 2ddp -t$SLURM_NTASKS -g -cnf=${NODES} -i case.jou
Log INFO "Ended fluent calculation."
rm cleanup-fluent*.sh

Log INFO "JOB FINISH"