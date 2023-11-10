

# Imports
import pathlib
import torch

# import albumentations
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from skimage.transform import resize
from src.dataset import SegmentationDataSet1
from src.transformations import (
    ComposeDouble,
    FunctionWrapperDouble,
    normalize_01,
    create_dense_target,
)
from src.unet import UNet
from src.trainer import Trainer
from src.losses import DiceLoss

# root directory
root = pathlib.Path.cwd() / "Carvana"


def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


# input and target files
inputs = get_filenames_of_path(root / "Input")[:4]
targets = get_filenames_of_path(root / "Target")[:4]

# training transformations and augmentations
transforms_training = ComposeDouble(
    [
        FunctionWrapperDouble(
            resize, input=True, target=False, output_shape=(512, 512, 3)
        ),
        FunctionWrapperDouble(
            resize,
            input=False,
            target=True,
            output_shape=(512, 512),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ),
        # AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

# validation transformations
transforms_validation = ComposeDouble(
    [
        FunctionWrapperDouble(
            resize, input=True, target=False, output_shape=(512, 512, 3)
        ),
        FunctionWrapperDouble(
            resize,
            input=False,
            target=True,
            output_shape=(512, 512),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        ),
        FunctionWrapperDouble(create_dense_target, input=False, target=True),
        FunctionWrapperDouble(
            np.moveaxis, input=True, target=False, source=-1, destination=0
        ),
        FunctionWrapperDouble(normalize_01),
    ]
)

# random seed
random_seed = 42

# split dataset into training set and validation set
train_size = 0.8  # 80:20 split

inputs_train, inputs_valid = train_test_split(
    inputs, random_state=random_seed, train_size=train_size, shuffle=True
)

targets_train, targets_valid = train_test_split(
    targets, random_state=random_seed, train_size=train_size, shuffle=True
)

# dataset training
dataset_train = SegmentationDataSet1(
    inputs=inputs_train, targets=targets_train, transform=transforms_training
)

dataset_train = SegmentationDataSet1(
    inputs=inputs_train, targets=targets_train, transform=transforms_training
)

# dataset validation
dataset_valid = SegmentationDataSet1(
    inputs=inputs_valid, targets=targets_valid, transform=transforms_validation
)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train, batch_size=2, shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid, batch_size=2, shuffle=True)

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model
model = UNet(
    in_channels=3,
    out_channels=2,
    n_blocks=4,
    start_filters=32,
    activation="relu",
    normalization="batch",
    conv_mode="same",
    dim=2,
).to(device)

# criterion
# criterion = torch.nn.CrossEntropyLoss()
criterion = DiceLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# trainer
trainer = Trainer(
    model=model,
    device=device,
    criterion=criterion,
    optimizer=optimizer,
    training_dataloader=dataloader_training,
    # validation_dataloader=dataloader_validation,
    lr_scheduler=None,
    epochs=1,
    epoch=0,
)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()
