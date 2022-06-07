import json
from typing import Dict, Union, Sequence, Tuple
import torch
import shutil
import preprocessing
import copy
from torch import Tensor
import numpy as np
from torch_geometric.data import Data
from pathlib import Path

import config as project_config

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


class Dataset:
    CONFIG_FILE_NAME = "config.json"

    def __init__(self, config: Dict[str, float]):
        self.config = config

        self.raw_dir = Path(project_config.DIR_DATA_RAW)
        self.processed_dir = Path(project_config.DIR_DATA_PROCESSED)

        self._process()
        data_file_names = [str(fn) for fn in self.processed_dir.glob("*.pt")]
        self.file_name_index = sorted(data_file_names)

    def __len__(self):
        return len(self.file_name_index)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self):
            d = self[self.n]
            self.n += 1
            return d
        else:
            raise StopIteration

    def __getitem__(self, idx: Union[int, np.integer, IndexType],) -> Union['Dataset', Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(int(idx))
            return data

        else:
            return self.index_select(idx)

    def _load_params(self):
        config_file_path = self.processed_dir / Dataset.CONFIG_FILE_NAME
        if config_file_path.is_file():
            with open(config_file_path, "r") as config_file:
                config = json.load(config_file)
            return config
        else:
            return None

    def _save_params(self):
        config_file_path = self.processed_dir / Dataset.CONFIG_FILE_NAME
        with open(config_file_path, "w") as config_file:
            json.dump(self.config, config_file)

    def _process(self):
        # check whether config has changed and processing is needed
        old_config = self._load_params()
        if self.config == old_config:
            # config has not changed, no reprocessing necessary
            return

        # clear dir
        shutil.rmtree(self.processed_dir)
        self.processed_dir.mkdir()

        # process
        preprocessing.process_parallel(
            input_dir=str(self.raw_dir),
            output_dir=str(self.processed_dir),
            connection_range=self.config["connection_range"],
            time_step_size=self.config["time_step_size"]
        )
        self._save_params()

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        file_name = self.file_name_index[idx]
        file_path = self.processed_dir / file_name
        data_obj = torch.load(file_path)
        return data_obj

    def index_select(self, idx: IndexType) -> 'Dataset':
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool."""
        indices = list(range(len(self)))

        if isinstance(idx, slice):
            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset.file_name_index = np.array(self.file_name_index)[indices].tolist()
        return dataset

    def shuffle(self, return_perm: bool = False,) -> Union['Dataset', Tuple['Dataset', Tensor]]:
        r"""Randomly shuffles the examples in the dataset.

        Args:
            return_perm (bool, optional): If set to :obj:`True`, will also
                return the random permutation used to shuffle the dataset.
                (default: :obj:`False`)
        """
        perm = torch.randperm(len(self))
        dataset = self.index_select(perm)
        return (dataset, perm) if return_perm is True else dataset
