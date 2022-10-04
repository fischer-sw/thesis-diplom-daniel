#!/bin/bash

function do_job {
    cd $1
    dos2unix case.sh
    dos2unix case.jou
    sbatch case.sh
    cd -
}

task=$1

if [ -d "$task" ]
then
    do_job $task
    qstat -u ${USER}
else
    for d in $(find . -type d -maxdepth 1 -mindepth 1)
    do
        do_job $d
    done
    qstat -u ${USER}
fi
