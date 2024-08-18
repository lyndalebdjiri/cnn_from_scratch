import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
import tempfile

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(train_dataloader, model, optimizer, loss, device):
    model.to(device)
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_value = loss(output, target)
        loss_value.backward()
        optimizer.step()
        train_loss += (1 / (batch_idx + 1)) * (loss_value.item() - train_loss)
        pred = torch.argmax(output, dim=1)
        correct += torch.sum(pred == target).item()
        total += target.size(0)
    accuracy = 100. * correct / total
    return train_loss, accuracy

def valid_one_epoch(valid_dataloader, model, loss, device):
    model.to(device)
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_value = loss(output, target)
            valid_loss += (1 / (batch_idx + 1)) * (loss_value.item() - valid_loss)
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == target).item()
            total += target.size(0)
    accuracy = 100. * correct / total
    return valid_loss, accuracy

def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot()])
    else:
        liveloss = None
    valid_loss_min = float('inf')
    logs = {}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    patience = 10
    early_stopping_counter = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(data_loaders["train"], model, optimizer, loss, device)
        valid_loss, valid_accuracy = valid_one_epoch(data_loaders["valid"], model, loss, device)
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTraining Accuracy: {:.2f}% \tValidation Accuracy: {:.2f}%".format(
                epoch, train_loss, valid_loss, train_accuracy, valid_accuracy
            )
        )
        if valid_loss < valid_loss_min:
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break
        scheduler.step(valid_loss)
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["accuracy"] = train_accuracy
            logs["val_accuracy"] = valid_accuracy
            logs["lr"] = optimizer.param_groups[0]["lr"]
            liveloss.update(logs)
            liveloss.send()

def one_epoch_test(test_dataloader, model, loss, device):
    test_loss = 0.0
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss_value = loss(logits, target)
            test_loss += (1 / (batch_idx + 1)) * (loss_value.item() - test_loss)
            pred = torch.argmax(logits, dim=1)
            correct += torch.sum(pred == target).item()
            total += data.size(0)
    accuracy = 100. * correct / total
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        accuracy, correct, total))
    return test_loss, accuracy


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        train_loss, _ = train_one_epoch(data_loaders['train'], model, optimizer, loss, device)
        assert not np.isnan(train_loss), "Training loss is nan"



def test_valid_one_epoch(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects

    for _ in range(2):
        valid_loss, _ = valid_one_epoch(data_loaders["valid"], model, loss, device)
        assert not np.isnan(valid_loss), "Validation loss is nan"


def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):
    model, loss, optimizer = optim_objects
    
    for _ in range(2):
        test_loss, _ = one_epoch_test(data_loaders["test"], model, loss, device)
        assert not np.isnan(test_loss), "Test loss is nan"
