import os.path as osp
from pathlib import Path

DIR_BASE = osp.dirname(osp.realpath(__file__))
DIR_DATA = str(Path(DIR_BASE) / "data")
DIR_DATA_RAW = str(Path(DIR_DATA) / "raw")
DIR_DATA_PROCESSED = str(Path(DIR_DATA) / "processed")
DIR_DATA_STUDIES = str(Path(DIR_DATA) / "studies")
DIR_DATA_MOLDFLOW_OUT = str(Path(DIR_DATA) / "moldflow_out")
