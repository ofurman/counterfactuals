import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image

from counterfactuals.datasets.moons import MoonsDataset
from counterfactuals.cf_methods.ppcef import PPCEF
from counterfactuals.losses import MulticlassDiscLoss
from counterfactuals.discriminative_models import LogisticRegression
from counterfactuals.generative_models import MaskedAutoregressiveFlow


def plot_model_distribution(model):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 12)

    xline = torch.linspace(-0.25, 1.25, 200)
    yline = torch.linspace(-0.25, 1.25, 200)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    with torch.no_grad():
        zgrid0 = model(xyinput, torch.zeros(40000, 1)).exp().reshape(200, 200)
        zgrid1 = model(xyinput, torch.ones(40000, 1)).exp().reshape(200, 200)

    zgrid0 = zgrid0.numpy()
    zgrid1 = zgrid1.numpy()

    _ = ax.contour(
        xgrid.numpy(),
        ygrid.numpy(),
        zgrid0,
        levels=10,
        cmap="Greys",
        linewidths=0.4,
        antialiased=True,
    )
    _ = ax.contour(
        xgrid.numpy(),
        ygrid.numpy(),
        zgrid1,
        levels=10,
        cmap="Oranges",
        linewidths=0.4,
        antialiased=True,
    )
    return ax


def plot_all_cf(X_test, X_cf):
    assert X_test.shape[1] == X_cf.shape[1], "Sizes of test set and counterfactuals is not equal."

    ax = plot_model_distribution(flow)

    # Classifier Line
    w1, w2 = list(disc_model.parameters())[0][0].detach().cpu().numpy()
    b = list(disc_model.parameters())[1].detach().cpu().numpy().item()
    c = -b / w2
    m = -w1 / w2
    xmin, xmax = -0, 1.0
    ymin, ymax = -0.5, 1.5
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, "red", lw=2.0, ls="dashed")
    plt.axis("off")

    # Classifier Line
    w1, w2 = list(disc_model.parameters())[0][0].detach().cpu().numpy()
    b = list(disc_model.parameters())[1][0].detach().cpu().numpy().item()
    c = -b / w2
    m = -w1 / w2
    xmin, xmax = -0, 1.0
    ymin, ymax = -0.5, 1.5
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, "black", lw=2.0, ls="dashed")
    plt.axis("off")

    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=matplotlib.colormaps["tab10"], s=50, alpha=0.8)
    ax.scatter(X_cf[:, 0], X_cf[:, 1], c="orange", s=50, alpha=0.8)

    for i in range(len(X_test)):
        ax.arrow(
            X_test[i, 0],
            X_test[i, 1],
            X_cf[i, 0] - X_test[i, 0],
            X_cf[i, 1] - X_test[i, 1],
            width=0.001,
            lw=0.001,
            length_includes_head=True,
            alpha=0.5,
            color="k",
        )

    return ax


def create_grid_image(folders, output_filename):
    """Creates a grid image combining images from multiple folders, with automatic grid calculation.

    Args:
        folders (list): A list of paths to folders containing images.
        output_filename (str): Name of the output image file.
    """

    images = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith((".png", ".jpg")):
                images.append(Image.open(os.path.join(folder, filename)))

    # Calculate grid dimensions
    num_images = len(images)
    rows = math.ceil(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    # Calculate image dimensions (assuming all have reasonably similar size)
    image_width, image_height = images[0].size

    grid_width = image_width * cols
    grid_height = image_height * rows
    grid_image = Image.new('RGB', (grid_width, grid_height))

    index = 0
    for row in range(rows):
        for col in range(cols):
            if index < num_images:  # Prevent going out of bounds
                grid_image.paste(images[index], (col * image_width, row * image_height))
            index += 1

    grid_image.save(output_filename)


methods = ['artelth20', 'cbce', 'CEGP', 'CEM', 'ppcef', 'wach']
dataset_name = "MoonsDataset"
classifier = "LogisticRegression"

dataset = MoonsDataset(file_path="data/moons.csv")
datasets = iter(dataset.get_cv_splits())

for i in range(5):
    X_train, X_test, y_train, y_test = next(datasets)  # yields splits one by one

    disc_model = LogisticRegression(dataset.X_train.shape[1], 1)
    disc_model.load(f"models/MoonsDataset/disc_model_{i}_{classifier}.pt")

    flow = MaskedAutoregressiveFlow(
        dataset.X_train.shape[1],
        hidden_features=4,
        num_blocks_per_layer=2,
        num_layers=5,
        context_features=1,
    )
    flow.load(f"models/MoonsDataset/gen_model_{i}_MaskedAutoregressiveFlow.pt")

    cf = PPCEF(
        gen_model=flow,
        disc_model=disc_model,
        disc_model_criterion=MulticlassDiscLoss(),
        neptune_run=None,
    )

    median_prob = (
        flow.predict_log_prob(dataset.train_dataloader(batch_size=64, shuffle=False))
        .median()
        .item()
    )

    for method in methods:
        X_cf = pd.read_csv(
            f"models/{dataset_name}/{method}/counterfactuals_{classifier}_{i}.csv"
        ).values

        print(i, method)
        print(X_cf.shape, X_test.shape)

        ax = plot_all_cf(X_test, X_cf)
        ax.set_title(method)
        plt.tight_layout()
        plt.savefig(
            f"models/{dataset_name}/{method}/visualization_{i}.png"
        )
        plt.close()

folders = [f"models/{dataset_name}/{method}" for method in methods]
output_filename = f"models/{dataset_name}/counterfactuals_{classifier}_comparison.png"

create_grid_image(folders, output_filename)
