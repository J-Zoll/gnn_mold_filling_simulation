import numpy as np
import preprocessing
import torch
from typing import List, Union, Tuple
from scipy.spatial import KDTree
from torch_geometric.data import Data
import meshio
from pathlib import Path


Pathlike = Union[str, Path]
Point3D = Tuple[float, float, float]


class NoProgressException(Exception):
    pass


class Rollout:
    def __init__(
            self,
            patran_file_path: Pathlike,
            model_file_path: Pathlike,
            output_file_path: Pathlike,
            connection_range,
            injection_location: Point3D,
            prefill_radius: float
    ):
        self.patran_file_path = patran_file_path
        self.model_file_path = model_file_path
        self.output_file_path = output_file_path
        self.connection_range = connection_range
        self.injection_location = injection_location
        self.prefill_radius = prefill_radius

        # load mesh
        print("loading mesh..")
        self.mesh = meshio.moldflow.read(patran_file_path)
        self.node_positions = self.mesh.points.tolist()
        self.num_nodes = len(self.node_positions)
        self.nodes = list(range(self.num_nodes))

        # load model
        print("loading model..")
        #self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = torch.load(model_file_path, map_location=self.device)

        # preprocess part to get graph
        print("constructing graph..")
        edge_list = preprocessing.get_edges(self.node_positions, connection_range)
        self.edge_index = torch.from_numpy(edge_list).T

        edge_distances = preprocessing.get_distances(np.array(self.node_positions), edge_list)
        self.edge_weight = torch.tensor(edge_distances, dtype=torch.float)
        print()

        # iteration 0: prefill some nodes around the injection location
        print("preparing fill process..")
        self.iteration = 0
        kd_tree = KDTree(self.node_positions)
        prefilled_nodes = kd_tree.query_ball_point(injection_location, prefill_radius, workers=8)  # list of nodes around the injection location
        self.filled = [n in prefilled_nodes for n in self.nodes]  # if a node is filled
        self.fill_time = [self.iteration if self.filled[n] else np.inf for n in self.nodes]  # in which iteration a node was filled
        self.iteration = 1

        print("Initialization finished!")


    def run(self):
        while self.has_unfilled_nodes():
            self.run_next_iteration()
        self.save_output_file()


    def run_next_iteration(self):
        with torch.no_grad():
            x = self.get_encoded_fill_state()
            data = Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight).to(self.device)
            output = self.model(data)  # [num_nodes, 1]-Tensor with entries x ∈ [0, 1]
            new_filled = output.ravel().round().bool().to("cpu").tolist()  # if a node is filled after this iteration

        # stop if rollout made no progress
        if not (sum(new_filled) > sum(self.filled)):
            self.save_output_file()
            raise NoProgressException("rollout made no progress")

        new_fill_time = self.get_new_fill_time(new_filled)
        self.print_status(new_filled)

        # prepare new iteration
        self.filled = new_filled
        self.fill_time = new_fill_time
        self.iteration += 1


    def has_unfilled_nodes(self) -> bool:
        return np.inf in self.fill_time


    def get_encoded_fill_state(self) -> torch.Tensor:
        FILLED = [1.0, 0.0]
        UNFILLED = [0.0, 1.0]
        nodes = list(range(len(self.filled)))
        encoding = [FILLED if self.filled[n] else UNFILLED for n in nodes]
        return torch.tensor(encoding, dtype=torch.float)


    def get_new_fill_time(self, new_filled: List[bool]) -> List[float]:
        num_nodes = len(self.fill_time)
        nodes = list(range(num_nodes))
        new_fill_time = []
        for n in nodes:
            if (self.fill_time[n] == np.inf) and new_filled[n]:
                new_fill_time.append(self.iteration)
            else:
                new_fill_time.append(self.fill_time[n])
        return new_fill_time


    def print_status(self, new_filled: List[bool]):
        percentage_filled = round(sum(new_filled) / self.num_nodes * 100, ndigits=2)
        changed_positively = [(not f) and nf for f, nf in zip(self.filled, new_filled)]
        changed_negatively = [f and (not nf) for f, nf in zip(self.filled, new_filled)]

        num_got_filled = sum(changed_positively)
        num_got_unfilled = sum(changed_negatively)

        iteration_str = str(self.iteration).zfill(3)
        percentage_filled_str = str(percentage_filled).zfill(5)
        print(f"{iteration_str} | {percentage_filled_str}% filled | +{num_got_filled}   -{num_got_unfilled}")


    def save_output_file(self):
        self.mesh.point_data["fill_time"] = np.array(self.fill_time)
        self.mesh.write(self.output_file_path)


def main():
    # fill in parameters to generate a rollout
    PATRAN_FILE_PATH = "path to patran file"
    MODEL_FILE_PATH = "path to trained model"
    OUTPUT_FILE_PATH = "path to output file"
    CONNECTION_RANGE = 0.004
    INJECTION_LOCATION = (0.0007512290069, 0.0006972170961, 0.189)
    PREFILL_RADIUS = 0.05

    rollout = Rollout(
        PATRAN_FILE_PATH,
        MODEL_FILE_PATH,
        OUTPUT_FILE_PATH,
        CONNECTION_RANGE,
        INJECTION_LOCATION,
        PREFILL_RADIUS
    )
    rollout.run()


if __name__ == "__main__":
    main()
