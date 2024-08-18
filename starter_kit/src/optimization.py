import torch
import torch.nn as nn
import torch.optim as optim

def get_loss():
    loss = nn.CrossEntropyLoss()
    return loss

def get_optimizer(
    model: nn.Module,
    optimizer: str = "adamw",
    learning_rate: float = 0.0001,
    weight_decay: float = 0.001,
    momentum: float = 0.9,  # Added momentum argument
):
    if optimizer.lower() == "sgd":
        opt = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,  # Use momentum for SGD
            weight_decay=weight_decay
        )
    elif optimizer.lower() == "adam":
        opt = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer.lower() == "adamw":
        opt = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt

def get_scheduler(optimizer, scheduler: str = "reducelronplateau", step_size: int = 10, gamma: float = 0.1):
    if scheduler.lower() == "steplr":
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler.lower() == "reducelronplateau":
        sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5)
    elif scheduler.lower() == "cosineannealinglr":
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size)
    else:
        raise ValueError(f"Scheduler {scheduler} not supported")

    return sched


######################################################################################
#                                     TESTS
######################################################################################
import pytest
import torch
import torch.nn as nn

@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)

def test_get_loss():
    loss = get_loss()
    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"

def test_get_optimizer_type(fake_model):
    opt = get_optimizer(fake_model, optimizer="sgd")  # Specify optimizer type
    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"

def test_get_optimizer_is_linked_with_model(fake_model):
    opt = get_optimizer(fake_model)
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])

def test_get_optimizer_returns_adam(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam")
    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected Adam optimizer, got {type(opt)}"

def test_get_optimizer_sets_learning_rate(fake_model):
    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)
    assert (
        opt.param_groups[0]["lr"] == 0.123
    ), "get_optimizer is not setting the learning rate appropriately. Check your code."

def test_get_optimizer_sets_momentum(fake_model):
    opt = get_optimizer(fake_model, optimizer="sgd", momentum=0.123)
    assert (
        opt.param_groups[0]["momentum"] == 0.123
    ), "get_optimizer is not setting the momentum appropriately. Check your code."

def test_get_optimizer_sets_weight_decay(fake_model):
    opt = get_optimizer(fake_model, optimizer="sgd", weight_decay=0.123)
    assert (
        opt.param_groups[0]["weight_decay"] == 0.123
    ), "get_optimizer is not setting the weight_decay appropriately. Check your code."
