import json

import numpy as np
import torch
from torch.nn import functional as F
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from dataset import Dataset
import config

CONNECTION_RANGE = 0.004
TIME_STEP_SIZE = 0.2
NUM_GNN_LAYERS = 3


def evaluate(model, test_dataset):
    conf_mat = np.zeros(shape=(2, 2))
    for d in tqdm(test_dataset):
        out = model(d).detach()
        y_true = d.y[:, 0].numpy()
        y_pred = F.softmax(out, dim=1).round()[:, 0].numpy()
        cm = confusion_matrix(y_true, y_pred)
        conf_mat += cm
    tn, fp, fn, tp = conf_mat.ravel()
    total = np.sum(conf_mat)
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    ntn, nfp, nfn, ntp = conf_mat.ravel() / total
    return dict(
        total=total,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        ntn=ntn,
        nfp=nfp,
        nfn=nfn,
        ntp=ntp
    )


def main():
    torch.manual_seed(42)

    dataset_config = dict(connection_range=CONNECTION_RANGE, time_step_size=TIME_STEP_SIZE)
    dataset = Dataset(dataset_config)
    test_dataset = dataset.shuffle()

    model_name = sys.argv[1]
    model_path = config.DIR_DATA_TRAINED_MODELS / model_name

    device = torch.device("cpu")
    model = torch.load(model_path, map_location=device)
    metrics = evaluate(model, test_dataset)
    with open("metrics.json", "w") as json_file:
        json.dump(metrics, json_file)


if __name__ == "__main__":
    main()
