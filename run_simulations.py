"""Script to simulate studies with Autodesk Moldflow

Command structure:
    >>> python run_simulations <input_dir> <output_dir>
"""

import os
import subprocess
from argparse import ArgumentParser
import pickle

import pythoncom
import win32com.client
from tqdm import tqdm
import madcad

from study import Study

# Path to Autodesk Moldflow
MF = os.path.join("C:/", "Program Files", "Autodesk", "Moldflow Insight 2021.1", "bin")

# Define outputs
OUT = {
    "1610": "fill_time",
    "1653": "weld_surface",
    "1722": "weld_line",
}


parser = ArgumentParser(
    description='Loads pickled Study Objects and simulates them with Moldflow'
)

parser.add_argument(
    "-i", "--input_dir",
    dest="input_dir",
    type=str,
    help="Directory containing pickled Study Objects.",
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

# unpickle all study objects in input_dir
file_names = os.listdir(os.path.abspath(args.input_dir))
pickle_file_names = filter(lambda fn: fn.endswith(".pickle"), file_names)
pickle_file_paths = [os.path.abspath(os.path.join(args.input_dir, pfn)) for pfn in pickle_file_names]
pickle_file_paths = sorted(pickle_file_paths)

# perform simulation with moldflow
for pfp in tqdm(pickle_file_paths):
    with open(pfp, "rb") as pf:
        obj = pickle.load(pf)
        if not isinstance(obj, Study):
            continue
    s = obj

    # Create working directory
    path = os.path.abspath(os.path.join(args.output_dir, s.name))
    os.mkdir(path)

    # Export geometry (needs numpy-stl)
    geo_name = f"{s.name}.stl"
    madcad.write(s.geometry, os.path.join(path, geo_name))

    # Connect to Moldflow Synergy
    Synergy = win32com.client.Dispatch("synergy.Synergy")
    Synergy.SetUnits("Metric")

    # Create project
    Synergy.NewProject(s.name, path)

    # Loop through injection locations
    for location, direction in s.injection_locations:
        # Import stl file
        ImpOpts = Synergy.ImportOptions
        ImpOpts.MeshType = "3D"
        ImpOpts.Units = "mm"
        Synergy.ImportFile(f"{s.name}.stl", ImpOpts, False)

        # Rename study
        study_name = f"{s.name}_{int(location[0])}_{int(location[1])}_study"
        Project = Synergy.Project
        Project.RenameItemByName(f"{s.name}_study", "Study", study_name)

        # Set injection location
        BoundaryConditions = Synergy.BoundaryConditions
        Direction = Synergy.CreateVector
        Direction.SetXYZ(*direction)
        Location = Synergy.CreateVector
        Location.SetXYZ(*location)
        EntList = BoundaryConditions.CreateNDBCAtXYZ(
            Location, Direction, 40000, pythoncom.Nothing
        )

        # Build mesh
        MeshGenerator = Synergy.MeshGenerator
        MeshGenerator.EdgeLength = 2.5
        MeshGenerator.Generate

        # Set number of intermediate results
        PropEd = Synergy.PropertyEditor
        Prop = PropEd.FindProperty(10080, 1)
        DVec = Synergy.CreateDoubleArray
        DVec.AddDouble(50)
        Prop.FieldValues(910, DVec)
        PropEd.CommitChanges("Process Conditions")

        # Save the sdy files
        StudyDoc = Synergy.StudyDoc
        StudyDoc.Save

        # Save mesh as Patran file
        Project = Synergy.Project
        Project.ExportModel(os.path.join(path, study_name + ".pat"))

        # Run the simulation
        p = subprocess.Popen(
            [os.path.join(MF, "runstudy.exe"), study_name + ".sdy",],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=path,
        )
        (output, err) = p.communicate()
        with open(os.path.join(path, study_name + ".log"), "w") as file:
            file.write(output.decode("windows-1252").strip())

        for key, value in OUT.items():
            p = subprocess.Popen(
                [os.path.join(MF, "studyrlt.exe"), study_name + ".sdy", "-xml", key,],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=path,
            )
            (output, err) = p.communicate()
            temp_name = os.path.join(path, f"{study_name}.xml")
            os.rename(temp_name, temp_name.replace(".xml", f"_{value}.xml"))
