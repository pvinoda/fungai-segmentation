#!/usr/bin/env python

# Example python script which processes an image and saves output in a directory based on arguments and a config file
# Derek DeVries - NCSU RFS
# 25 April 2024

# Example use: ./fungai.py --file file_to_process.png

# conda install -c pathlib
# pip install Pillow
import argparse
import shutil
import pathlib
import os
import pprint
from PIL import Image, ImageFilter

parser = argparse.ArgumentParser()
parser.add_argument("--file","-f",help="input file that is to be processed - not path, just file name")# hazel
parser.add_argument("--config","-c",default="/home/fungai/fungai_dev/config",help="path to config file, key=value format with double quotes around value")
args = parser.parse_args()

# Print arguments that were supplied to script
print("Printing arguments supplied to script")
print(args)

# Read in config file
with open(args.config,'r') as config_file:
    config_lines = config_file.readlines()
config = {}
for line in config_lines:
    if "=" in line:
        line = line.strip()
        line = line.split("=")
        ckey = line[0].strip('"')
        cval = line[1].strip('"')
        config[ckey] = cval
print("Keys and values from config file: " + args.config)
for key in config:
    print("Key: " + key)
    print("Value: " + config[key])
    print()

# Define variables
stage_path = pathlib.Path(config["stage_dir"]) / args.file
scratch_dir = pathlib.Path(config["scratch_dir"]) / args.file.split(".")[0]
out_dir = pathlib.Path(config["out_dir"]) / args.file.split(".")[0]
print("File start location: " + str(stage_path))
print("File process location: " + str(scratch_dir))
print("Process output location: " + str(out_dir))
print("Making scratch process directory if it does not already exist")
os.makedirs(scratch_dir, exist_ok=True)

# Move input file from staging area to processing area
print("Moving input file from staging area to scratch")
shutil.move(str(stage_path), scratch_dir)
scratch_path = scratch_dir / args.file

# Process file, send outputs to output directory - example is a package that marks edges in white on a black background
print("Procesing input file from scratch and saving output to output directory")
edges = str(scratch_dir) + "/" + args.file.split(".")[0] + "_EDGES." + args.file.split(".")[1]
print("Edges path: " + edges)
image = Image.open(scratch_path)
image = image.convert('RGB')
image = image.filter(ImageFilter.FIND_EDGES)
image.save(edges) 

# Move inputs and outputs from scratch to out
print("Moving outputs from scratch to output directory")
shutil.copytree(str(scratch_dir),out_dir)
shutil.rmtree(str(scratch_dir))
