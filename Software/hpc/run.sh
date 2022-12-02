#!/bin/bash

function do_job {
    cd $1
    
	FILES=$(ls ./Data | wc -l)
	if [ $FILES == 0 ]; then

		GZ=$(ls | grep .gz | wc -l)
		N_JOB_ID=$(qstat -u $USER | grep $1 | wc -l)
		if [ $GZ != 2 ]; then
			echo "No .gz files found to run case"
		else
			if [ $N_JOB_ID == 0 ]; then
				dos2unix case.sh
    				dos2unix case.jou
				dos2unix post.sh
				JOB_ID=$(sbatch --parsable case.sh)
      				NAME=$(grep "^#SBATCH -J" ./case.sh | cut -d" "  -f3)
				echo Name = ${NAME}
				echo SIM JOB_ID ${JOB_ID}
				POST_ID=$(sbatch --parsable --dependency=afterok:${JOB_ID} --kill-on-invalid-dep=yes post.sh)
				echo POST ID ${POST_ID}
			else
				echo Job already running or in queue
			fi
		fi
		
	else
		echo "Already calculated data for case $1"
		#dos2unix post.sh
		#echo "Starting post processing"
		#sbatch post.sh
	fi
    cd -
}

function get_job_id {
	 local name=$1
   qstat -u $USER | grep $name | cut -d" " -f1
}

function do_post {
	cd ~/Jobs/$1
    
	FILES=$(ls ./Data | wc -l)
	if [ $FILES != 0 ]; then
		dos2unix post.sh
    		NAME=$(grep "^#SBATCH -J" ./case.sh | cut -d" "  -f3)
    		JOB_ID=$(get_job_id $NAME)
    	if [ "$JOB_ID" = "" ]; then
		CSV=$(ls | grep .csv | wc -l)
		if [ $CSV !=  6 ]; then
        		echo "Removing dependency"
        		grep -v "dependency" ./post.sh > post_tmp.sh
        		mv ./post_tmp.sh ./post.sh
        		sbatch post.sh
		fi
    	else
       		echo "Host Job ${NAME} ${JOB_ID} is still running"     
    	fi
    
		
	else
		echo "No data generated for $1"
	fi
}

function do_option {

if  [ "$#" == 0 ]; then
	echo "Options -c for running a specific case, -a for running all jobs that have not generated data so far and -p for postprocessing of a specific case"
fi

while [ -n "$1" ]; do # while loop starts
	case "$1" in
	-prep) do_prep_all ;;
	-clean) do_clean ;;
	-j) do_job $2 ;;
	-a) do_run_all ;;
	-p) do_post_all  ;;
	*) echo "Option $1 not recognized. -prep, -c, -a and -p are allowed." ;; # In case you typed a different option other than a,b,c
	esac
	shift
done

}

function do_post_all {
	cd ~/Jobs
	JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
	for d in $JOBS
    do
		FILES=$(ls ~/Jobs/$d/Data | wc -l)
		# echo $d $FILES
		if [ $FILES != 0 ]; then
			do_post $d
		fi
    done
	qstat -u ${USER}
}

function do_prep_all {

	echo "Preparing all cases ..."
        cd ~/Jobs
        JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
        for d in $JOBS
    	do
			dos2unix ~/Jobs/$d/post.sh
			dos2unix ~/Jobs/$d/case.sh
    	done
        echo "Preparation finished"

}


function do_clean {
	echo "Cleaning cases ..."
	cd ~/Jobs
	JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
	for d in $JOBS
    	do
		cd ~/Jobs/$d
		FILES=$(ls | wc -l)	
		if [ $FILES == 0 ]; then
			rmdir ~/Jobs/$d
		else
			DATAFILES=$(ls ./Data | wc -l)
			GZ=$(ls | grep .gz | wc -l)
			if [ $DATAFILES == 0 ] || [ $GZ == 0 ]; then
				echo Removing $d
				rm -r ~/Jobs/$d/*
				rmdir ~/Jobs/$d
				echo Finished removing $d
			fi
		fi
		
		done
	echo "Cleaning finished"
}

function do_run_all {

	JOBS=$(find . -nowarn -type d -maxdepth 1 -mindepth 1)
	for d in $JOBS
    do
		FILES=$(ls $d/Data | wc -l)
		# echo $d $FILES
		if [ $FILES == 0 ]; then
			do_job $d
		fi
    done
	qstat -u ${USER}
}

do_option $*
