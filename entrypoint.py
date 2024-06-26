#!/usr/bin/env python3
import sys
import os
import argparse
import json
import pickle
import time
import numpy as np
import shutil

from utils import save_segmentation_results, show_segmentation, dump_image_into_sqlite
from cellpose import io as cellio, models as cellmodels



def run_segmentation(username: str, action: str, image_object: np.array, timestamp):

    app_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(app_dir, "yeastvision", action, action)

    if "retrained" in action:
        weights_path = f"/rs1/researchers/o/oargell/fungai-rs/fungai_retrain_stage/{action}/{username}/models/retrained_{username}_{action}_{timestamp}"

    if action == "proSeg":
        proseg_run_and_eval(username=username, weights_path=weights_path, timestamp=timestamp, image_object=image_object)
    else:
        action_run_and_eval(username=username, weights_path=weights_path, timestamp=timestamp, image_object=image_object, action=action)

def proseg_run_and_eval(username: str, weights_path: str, timestamp, image_object: np.array):
    print("Performing Proseg Inference and Evaluation...")

    action = "proSeg"
    ims = image_object

    print(f"Initializing CellposeModel and performing evalutaion for {action} model.")

    print("Weights Path:", weights_path)
    model = cellmodels.CellposeModel(gpu=True, model_type=action, pretrained_model=weights_path)
    cmasks, cflows, _ = model.eval(ims[0], diameter=30, channels=[0, 0], cellprob_threshold=0.5,
                                flow_threshold=0.9)
    print("Evaluated" + action + " model.")
    # Save all results
    save_segmentation_results(username, timestamp, action, ims[0], cmasks, cflows)


def action_run_and_eval(username: str, weights_path: str, image_object: np.array, action: str, timestamp):
    print(f"Performing {action} Inference and Evaluation...")
    model = cellmodels.CellposeModel(gpu=True, model_type=action, pretrained_model=weights_path)

    ims = image_object
    diameter, cellprob_threshold, flow_threshold = 0, 0.5, 0.9
    masks, flows, styles = model.eval(ims[0], diameter=diameter, channels=[0, 0], cellprob_threshold=cellprob_threshold,
                                     flow_threshold=flow_threshold)
    print(f"Evaluated {action} model.") 
    # Save all results in output directory
    save_segmentation_results(username, timestamp, action, ims[0], masks, flows) 

def main():
    parser = argparse.ArgumentParser(description='Run segmentation from a pickle file.')
    parser.add_argument('--pickle_path', type=str, help='Path to the pickle file containing all required arguments.')

    # Parse the command line arguments
    args = parser.parse_args()

    print(f"Pickle path is being fetched now for {args.pickle_path}")
###    filepath = "/rs1/researchers/o/oargell/fungai-rs/fungai_stage/"
###    with open(os.path.join(filepath, args.pickle_path), 'rb') as file:
    with open(args.pickle_path, 'rb') as file: 
       data = pickle.load(file)
    
    print("Unpickling successful")
    print(type(data)) # must be of type FileTransferObject

    print(data.image_arr)

    username: str = data.username
    action: str = data.action
    ims: np.array = data.image_arr
    timestamp = data.timestamp

    source = args.pickle_path
    destination = "/rs1/researchers/o/oargell/fungai-rs/moved_stage_files"

    shutil.move(source, destination)
    print("The pickle files have been moved out of the stage directory")

    run_segmentation(username=username, action=action, image_object=ims, timestamp=timestamp)

if __name__ == "__main__":
    main()