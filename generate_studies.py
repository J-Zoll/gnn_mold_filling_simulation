"""Script that defines a command line program to generate Study objects.

    Usage:
        Use the following command to generate 50 studies in the studies data directory using the
        PlateWithHole generator.
            >> python generate_studies.py PlateWithHoleGenerator -n 50

        Use the following command to get more information about the parameters.
            >> python generate_studies.py -h
"""

from argparse import ArgumentParser
import os.path as osp

import config
from study import generators
from tqdm import tqdm
import pickle


parser = ArgumentParser(
    description='Generates study objects and saves them with pickle in the studies data directory.'
)

parser.add_argument(
    "study_generator",
    type=str,
    help=f"Generator to create studies. Must be on of these: {list(generators.keys())}"
    )
parser.add_argument(
    "-n", "--num_samples",
    default=1,
    type=int,
    required=False,
    help="Number of samples to create.",
    dest="num_samples"
)

# verify arguments
args = parser.parse_args()

if args.study_generator not in generators.keys():
    raise ValueError("<study_generator> must be a valid generator.")

# generate studies
study_generator = generators[args.study_generator]
print("Generating studies..")
for i in tqdm(range(args.num_samples)):
    s = study_generator.generate_study()
    s_name = s.name
    pickle_file_name = f'{s_name}.pickle'
    pickle_file_path = osp.join(config.DIR_DATA_STUDIES, pickle_file_name)
    with open(pickle_file_path, "wb") as f:
        pickle.dump(s, f)
