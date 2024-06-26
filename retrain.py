import argparse
import pickle
import subprocess
import shutil
import os

def retrain_with_cellpose(username: str, action: str, timestamp: str):
    command = [
        "python", "-m", "cellpose",
        "--train",
        "--dir", f"/rs1/researchers/o/oargell/fungai-rs/fungai_retrain_stage/{action}/{username}/",
        "--pretrained_model", f"/rs1/researchers/o/oargell/fungai-rs/independent-seg-fungai/segmentationModels/{action}",
        "--chan", "1",
        "--learning_rate", "0.1",
        "--weight_decay", "0.0001",
        "--n_epochs", "10",
        "--mask_filter", "_cp_masks",
        "--model_name_out", f"retrained_{username}_{action}",
	    "--min_train_masks", "1",
        "--verbose"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print("Return code:", result.returncode)
    print("Output:", result.stdout)
    print("Errors:", result.stderr)
    
    print("Moving the retrained model into segmentationModels folder....")

    model_source = f"/rs1/researchers/o/oargell/fungai-rs/fungai_retrain_stage/{action}/{username}/models/retrained_{username}_{action}"
    model_destination = "/rs1/researchers/o/oargell/fungai-rs/independent-seg-fungai/segmentationModels"

    shutil.move(model_source, model_destination)

    print(f"Retrained model for {username} and {action} is now inside segmentation models")
    

def main():
    parser = argparse.ArgumentParser(description='Run retraining of a segmodel from a pickle file.')
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


    username: str = data.username
    action: str = data.action
    timestamp = data.timestamp

    source = args.pickle_path
    destination = "/rs1/researchers/o/oargell/fungai-rs/moved_stage_files"

    shutil.move(source, destination)
    print("The pickle files have been moved out of the stage directory.")

    retrain_with_cellpose(username=username, action=action, timestamp=timestamp)

if __name__ == "__main__":
    main()