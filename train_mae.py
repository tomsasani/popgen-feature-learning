import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import stdpopsim
import msprime
import wandb
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from sklearn.metrics import DistanceMetric
from sklearn.covariance import ShrunkCovariance


import glob
import os
import skimage.io as io

import models
from generator_fake import prep_simulated_region
import util
import transforms
import torchvision.transforms.v2 as v2
import global_vars
import losses
from scripts.real_data_iterator import RealData
import params
import demographies
import scipy.stats as ss
import torchvision


def train_loop(
    model,
    minibatch,
    loss_fn,
    optimizer,
    
):
    model.train()

    B, C, H, W = minibatch.shape

    pred, mask = model(minibatch.to(DEVICE))

    loss = loss_fn(pred, minibatch.to(DEVICE), mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    return loss.item()

# def linear_probe(
#     model,
#     minibatch,
#     loss_fn,
#     optimizer,

# ):
#     model.train()

#     B, C, H, W = minibatch.shape

#     pred, mask = model(minibatch.to(DEVICE))

#     loss = loss_fn(pred, minibatch.to(DEVICE), mask)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


#     return loss.item()


def plot_example(
    model,
    batch,
    plot_name: str,
):
    plt.rc("font", size=16)
    f, axarr = plt.subplots(1, 2, figsize=(24, 8))
    B, C, H, W = batch.shape

    # print (batch.shape)

    model.eval()
    with torch.no_grad():
        pred, mask = model(batch.to(DEVICE))
    # pred shape is B, H, CW
    pred = pred.reshape((B, H, C, W)).permute((0, 2, 1, 3))

    b1 = np.transpose(batch[0].cpu().numpy(), (1, 2, 0))
    p1 = np.transpose(pred[0].cpu().numpy(), (1, 2, 0))

    axarr[0].imshow(b1)
    axarr[1].imshow(p1)

    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()


EPOCHS = 21
DEVICE = torch.device("cuda")
N_CLUSTERS = 1

def decay(step: int):
    return 0.9 ** (step / EPOCHS)


def warmup_plus_decay(step: int):
    warmup_steps = int(0.1 * EPOCHS)
    rest_steps = EPOCHS - warmup_steps
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return 0.9 ** (step / rest_steps)


def main(config=None):

    # start a new wandb run to track this script
    # with wandb.init(project="spectrum-popgen", config=config):

    # config = wandb.config

    # batch_size = config["batch_size"]
    # lr = config["lr"]
    # representation_dim = config["representation_dim"]
    # projector_dim = config["projector_dim"]
    # n_snps = config["n_snps"]
    # agg = config["agg"]
    # mask_frac = config["mask_frac"]
    # representation_layer = config["representation_layer"]
    # stride = config["stride"]
    # lmbda = config["lmbda"]
    # hidden_layers = config["hidden_layers"]
    # kernel_size = config["kernel_size"]
    # use_ssdk = config["use_ssdk"]

    model = models.MAE(
        width=32,
        height=33,
        in_channels=1,
        num_heads=8,
        depth=1,
        mlp_ratio=2,
        hidden_size=128,
    )

    orig_tfms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Grayscale(),
        v2.Resize(size=(33, 32)),
    ]

    rng = np.random.default_rng(1234)

    # NOTE: train on just the neutral data
    train = torchvision.datasets.MNIST(
        root="mnist",
        download=True,
        train=True,
        transform=v2.Compose(orig_tfms),
    )

    test = torchvision.datasets.MNIST(
        root="mnist",
        download=True,
        train=False,
        transform=v2.Compose(orig_tfms),
    )

    train_loader = DataLoader(
        dataset=train,
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test,
        batch_size=256,
        shuffle=True,
        drop_last=False,
    )

    model = model.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print (pytorch_total_params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1.5e-6,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=decay,
    )

    # loss_fn = torch.nn.MSELoss()
    loss_fn = losses.MAELoss()
    res = []
    for epoch in tqdm.tqdm(range(EPOCHS)):
        model.eval()

        epoch_train_loss = 0

        for bi, (batch, _) in enumerate(train_loader):

            if epoch % 5 == 0 and bi == 0:
                plot_example(model, batch, f"fig/{epoch}.png")
            train_loss = train_loop(
                model,
                batch,
                loss_fn,
                optimizer,
                
            )
            epoch_train_loss += train_loss

        print (epoch_train_loss / len(train_loader))
        if scheduler is not None:
            scheduler.step()


if __name__ == "__main__":


    # sweep_configuration = {
    #     "method": "grid",
    #     "name": "sweep",
    #     "metric": {"goal": "maximize", "name": "salient_silhouette"},
    #     "parameters": {
    #         "batch_size": {"values": [512]},
    #         "lmbda": {"values": [5e-3]},
    #         "agg": {"values": ["max"]},
    #         "hidden_layers": {"values": [2]},
    #         "mask_frac": {"values": [0.25]},
    #         "lr": {"values": [1e-3]},
    #         "padding": {"values": [False]},
    #         "representation_dim": {"values": [32]},
    #         "n_snps": {"values": [32]},
    #         "projector_dim": {"values": [512]},
    #         "include_dists": {"values": [False]},
    #         "representation_layer": {"values": [False]},
    #         "use_ssdk": {"values": [True]},
    #         "stride": {"values": [1]},
    #         "kernel_size": {"values": [5]},
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="barlow-popgen")

    # wandb.agent(sweep_id, function=main)
    main()
