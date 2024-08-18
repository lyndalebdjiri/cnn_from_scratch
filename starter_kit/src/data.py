import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing
import matplotlib.pyplot as plt
import pytest
from .helpers import compute_mean_and_std, get_data_location

def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = 1, limit: int = -1
):
    data_loaders = {"train": None, "valid": None, "test": None}
    base_path = Path(get_data_location())
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        ),
    }

    train_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["train"])
    valid_data = datasets.ImageFolder(base_path / "train", transform=data_transforms["valid"])
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    test_data = datasets.ImageFolder(base_path / "test", transform=data_transforms["test"])
    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return data_loaders

def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # obtain one batch of training images
    dataiter = iter(data_loaders['train'])
    images, labels = next(dataiter)

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # Get class names from the train data loader
    class_names = data_loaders['train'].dataset.classes

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])

@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)

def test_data_loaders_keys(data_loaders):
    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"

def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you forget to resize and/or crop?"

def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.__next__()

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert len(labels) == 2, f"Expected a labels tensor of size 2, got size {len(labels)}"

def test_visualize_one_batch(data_loaders):
    visualize_one_batch(data_loaders, max_n=2)
