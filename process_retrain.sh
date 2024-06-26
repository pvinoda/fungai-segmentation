#!/bin/bash
#BSUB -W 10


# A simmple example bash script that takes an image file and gives it to a python script for processing.
# Intended to be executed as part of an lsf job. As originally designed, run-specific lines/arguments
#  will be inserted after the last instance of bsub (above this description)

# Prabhanjan Vinoda Bharadwaj - NCSU CSC

# 20 May 2024

# Example use line: bsub < process_job.sh

# Read in typical bash environment details
source ~/.bashrc
source $config_file # Should be set by wrapper script that modifies this one 


source /rs1/researchers/o/oargell/fungai-rs/myconda/etc/profile.d/conda.sh

conda activate fungenv

echo "Starting Retraining Job for $FILEVAR"
python /rs1/researchers/o/oargell/fungai-rs/independent-seg-fungai/retrain.py --pickle_path /rs1/researchers/o/oargell/fungai-rs/fungai_retrain_info/$FILEVAR 
echo "Job script completed."
