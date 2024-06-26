#!/bin/bash
# Set the script path
script_path="/rs1/researchers/o/oargell/fungai-rs/independent-seg-fungai/check_and_submit.sh"

# Loop to run the job twelve times every minute
for i in {1..12}
do
   # Execute the script
   bash "$script_path"

   # Sleep for 5 seconds before the next run
   sleep 5
done

