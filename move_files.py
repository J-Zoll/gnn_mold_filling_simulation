"""Script to move Simulation files to different location"""

import os
from argparse import ArgumentParser
import shutil
from time import sleep

parser = ArgumentParser(
    description='Moves file simulation files to a different location.'
)

parser.add_argument(
    "-i", "--input_dir",
    dest="input_dir",
    type=str,
    help="Directory containing output files.",
    default=os.getcwd(),
    required=False
    )

parser.add_argument(
    "-o", "--output_dir",
    dest="output_dir",
    type=str,
    help="Directory to place the output files in.",
    default=os.getcwd(),
    required=False
)
args = parser.parse_args()

# verify arguments
if not os.path.isdir(args.input_dir):
    raise ValueError('<input_dir> must be a path to a valid directory.')
if not os.path.isdir(args.output_dir):
    raise ValueError('<output_dir> must be a path to a valid directory.')

running = True
while running:
    # get files in input dir
    file_names = os.listdir(os.path.abspath(args.input_dir))
    file_names = sorted(file_names)
    file_names = file_names[:-1]  # exclude folder which might be in use of another script

    # move folders except from one to output destination
    for fn in file_names:
        src = os.path.abspath(os.path.join(args.input_dir, fn))
        dest = os.path.abspath(os.path.join(args.output_dir, fn))
        print(f"moving {fn} ..")
        shutil.move(src, dest)
    
    # go to sleep before next iteration
    print("Sleeping for 30 s ..")
    sleep(30)
