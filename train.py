import torch
from torch.nn import functional as F
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from dataset import Dataset
from fillsimnet import FillSimNet
import config
import os.path as osp

CONNECTION_RANGE = 0.004
TIME_STEP_SIZE = 0.2
NUM_GNN_LAYERS = 3

def loss_function(input, target):
    return F.binary_cross_entropy(input, target)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def train_epoch(model, data_loader, criterion, device):
    optimizer = Adam(params=model.parameters(), lr=0.001)
    model = model.to(device)
    loss_sum = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        batch.to(device)
        out = model(batch)
        target = batch.y[:, 0].view(-1, 1)
        loss = criterion(out, target)
        loss_sum += loss.detach().item()
        loss.backward()
        optimizer.step()
    mean_loss = loss_sum / len(data_loader)
    return model, mean_loss


def main():
    torch.manual_seed(42)

    dataset_config = dict(connection_range=CONNECTION_RANGE, time_step_size=TIME_STEP_SIZE)
    dataset = Dataset(dataset_config)
    SAMPLE_SIZE = 800
    train_dataset = dataset.shuffle()[:SAMPLE_SIZE]
    train_loader = DataLoader(train_dataset, batch_size=5, num_workers=1)
    model = FillSimNet(NUM_GNN_LAYERS)
    criterion = loss_function
    device = get_device()

    for epoch in range(10):
        model, loss = train_epoch(model, train_loader, criterion, device)
        print(f"epoch_{epoch}_loss: {loss}\n")
        model_path = osp.join(config.DIR_DATA_TRAINED_MODELS, f"model_epoch_{epoch}.pt")
        torch.save(model, model_path)
        loss_file_path = osp.join(config.DIR_DATA_TRAINED_MODELS, "losses.txt")
        with open(loss_file_path, "a") as loss_file:
            loss_file.write(f"epoch_{epoch}: {loss}\n")


if __name__ == "__main__":
    main()
