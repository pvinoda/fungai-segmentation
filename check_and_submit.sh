#!/bin/bash

# A bash script that reads in configuration information from a file, checks a staging directory for files that match
# a given extension, then creates and submits a tailored job submission script for each qualified file

# Derek DeVries - RFS

# 18 April 2024


# Example use line: 

# Save working directory as parent_path to locate configs and companion scripts (refers to true path even through symlinks)
script_path=$(readlink -f "$0")
parent_path=$(dirname "$script_path")

# Set necessary environmental variables
source "$parent_path/config"
source  "/usr/local/lsf/conf/profile.lsf"

# Iterate through list of images in a folder that end with an extension (e.g. PNG, not strictly necessary)
for input_image in $(ls $stage_dir | grep -i pickle)
do
    echo "Processing $input_image"
    job="${input_image%%.*}" # Job name
    job_check=$(bjobs -J $job | wc -l) # Is a job with this name already running? If so, don't submit a new one.
    if [ $job_check == 0 ]; then
    mkdir -p $job_dir/$job # Make a directory for job script and outputs
    jobscript="$job_dir/$job/process_job.sh"
    
    # Lines to be added below last instance of BSUB in your template job script
    addlines="\
#BSUB -J $job\n\
#BSUB -e $job_dir/$job/stderr.%J\n\
#BSUB -o $job_dir/$job/stdout.%J\n\
source $parent_path/config\n\
config_file=\"$parent_path/config\"\n\
FILEVAR=\"$input_image\""
    
    # Some sed magic that adds your addlines after the last line with BSUB in it
    sed -e "$(grep -n 'BSUB' $parent_path/process_job.sh |tail -1|cut -f1 -d':')a $addlines" $parent_path/process_job.sh > $jobscript
    
    # Submit custom script
    bsub < $jobscript
    else
    
    # Message that an identically named job is currently pending/running
    echo "Job with name $job has already been submitted and has not yet completed"
    fi
done

