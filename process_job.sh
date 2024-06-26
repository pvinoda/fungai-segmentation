#!/bin/bash
#BSUB -W 10


# A simmple example bash script that takes an image file and gives it to a python script for processing.
# Intended to be executed as part of an lsf job. As originally designed, run-specific lines/arguments
#  will be inserted after the last instance of bsub (above this description)

# Derek DeVries - RFS

# 18 April 2024

# Example use line: bsub < process_job.sh

# Read in typical bash environment details
source ~/.bashrc
source $config_file # Should be set by wrapper script that modifies this one 


source /rs1/researchers/o/oargell/fungai-rs/myconda/etc/profile.d/conda.sh

conda activate fungenv

python /rs1/researchers/o/oargell/fungai-rs/independent-seg-fungai/entrypoint.py --pickle_path /rs1/researchers/o/oargell/fungai-rs/fungai_stage/$FILEVAR 
echo "Job script completed."
