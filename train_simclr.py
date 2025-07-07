from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, STL10
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from PIL import Image, ImageOps, ImageFilter
import random

import models
import losses
import generator_fake
import demographies
import params
import util

import wandb


# def simulate_batch(engine, n_smps: int = 32, n_snps: int = 32, batch_size: int = 32):

#     param_vals = [
#         SIM_PARAMS.N1.proposal(9_000, 1, RNG),
#         SIM_PARAMS.N2.proposal(5_000, 1, RNG),
#         SIM_PARAMS.T1.proposal(1_500, 1, RNG),
#         SIM_PARAMS.T2.proposal(350, 1, RNG),
#     ]
#     counted = 0

#     minibatch = np.empty((batch_size, 3, n_smps * 2, n_snps))

#     while counted < batch_size:

#         region = engine.sample_fake_region(
#             [n_smps],
#             param_values=param_vals,
#         )
#         n_batches_zero_padded = util.check_for_missing_data(region)
#         if n_batches_zero_padded > 0:
#             minibatch = np.empty((batch_size, 3, n_smps * 2, n_snps))
#             param_vals = [
#                 SIM_PARAMS.N1.proposal(9_000, 1, RNG),
#                 SIM_PARAMS.N2.proposal(5_000, 1, RNG),
#                 SIM_PARAMS.T1.proposal(1_500, 1, RNG),
#                 SIM_PARAMS.T2.proposal(350, 1, RNG),
#             ]
#             continue
#         else:
#             minibatch[counted] = region
#             counted += 1

#     return torch.from_numpy(minibatch)


class MajorMinorTransform(torch.nn.Module):

    def __init__(self):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        pass

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        # convert image so that 0s are -1 and 1s stay the same, but
        # only in the first channel
        new_x = torch.clone(x)
        new_x[new_x == 0] = -1
        new_x[1:, :, :] = x[1:, :, :]

        return new_x


class RandomSiteMaskingTransform(torch.nn.Module):
    def __init__(self, rng: np.random.default_rng, mask_ratio=0.5):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        # draw from uniform distribution to figure out how many sites
        # to mask, a la
        self.mask_ratio = rng.uniform(0.1, mask_ratio)

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        C, H, W = x.shape
        n_sites_W = int(self.mask_ratio * W)

        # Randomly select pixels to mask
        mask_sites_W = torch.randperm(W, dtype=int)[:n_sites_W]
        mask = torch.ones((H, W), dtype=torch.int8)
        
        mask[:, mask_sites_W] = 0
        masked_x = x * mask  # Broadcast across channels

        return masked_x

class RandomRepolarizationTransform(torch.nn.Module):
    def __init__(self, rng: np.random.default_rng, frac_sites: float = 0.25):
        """
        Args:
            mask_ratio (float): Fraction of pixels to mask (set to zero).
        """
        self.frac_sites = rng.uniform(0.1, frac_sites)

    def __call__(self, x):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (C, H, W).
        Returns:
            masked_x (torch.Tensor): Image with random pixels zeroed out.
            mask (torch.Tensor): Mask tensor of shape (1, H, W) with 1s where pixels are kept, 0s where masked.
        """
        C, H, W = x.shape

        # randomly select sites to repolarize
        n_sites = int(W * self.frac_sites)
        mask_sites = torch.randperm(W, dtype=int)[:n_sites]
       
        # # create copy of the input array
        x_copy = torch.clone(x)
        x_copy[0, :, mask_sites] = 1 - x[0, :, mask_sites]
    
        return x_copy


class ChannelSubset(torch.nn.Module):
    def __init__(self, n: int = 2):
        self.n = n

    def __call__(self, x):
        if self.n == 1:
            return torch.unsqueeze(x[0, :, :], dim=0)
        else:
            return x[:self.n, :, :]

class PermuteHaplotypes(torch.nn.Module):
    def __init__(self, ):
        pass

    def __call__(self, x):
        C, H, W = x.shape
        yi = torch.randperm(H)
        return x[:, yi, :]


class RandomlySizedCrop(torch.nn.Module):
    def __init__(self, width: int = 32, min_height: int = 32):
        self.width = width
        self.min_height = min_height

    def __call__(self, x):
        
        B, C, H, W = x.shape
        # get random x coordinate
        xi = torch.randint(W - self.width, size=(1,))[0]
        yi = torch.randint(H - self.min_height, size=(1,))[0]
        return x[:, :, yi:, xi:xi + self.width]


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


def train_linear_clf(model, loader, encoding_dim: int = 128, n_classes: int = 3):

    # initialize a simple linear classifier
    clf = torch.nn.Sequential(torch.nn.Linear(encoding_dim, n_classes))
    clf = clf.to(DEVICE)

    optimizer = torch.optim.Adam(clf.parameters(), lr=LR)

    loss_fn = torch.nn.CrossEntropyLoss()

    clf.train()
    model.eval()

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        # get encoded data from model
        enc, proj = model(batch_x)
        # feed encoded data into the simple classifier
        preds = clf(enc)
        loss = loss_fn(preds, torch.unsqueeze(batch_y, dim=1).to(DEVICE).float())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


    return clf

def test_linear_clf(model, loader):

    clf = LogisticRegression(random_state=42, max_iter=1_000)
    model.eval()

    reps, labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            # get encoded data from model
            enc, proj = model(batch_x)
            # feed encoded data into the simple classifier
            reps.append(enc.cpu())
            labels.append(batch_y.cpu())
            
    reps = np.concatenate(reps)
    labels = np.concatenate(labels)

    X_train, X_test, y_train, y_test = train_test_split(reps, labels)
    X_train = preprocessing.StandardScaler().fit_transform(X_train)
    X_test = preprocessing.StandardScaler().fit_transform(X_test)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return reps, labels, score

def train_loop(
    model,
    dataloader,
    loss_fn,
    optimizer,
):

    model.train()

    n_batches = len(dataloader)
    total_loss = 0
    total_acc = 0
    batch_size = None
    for batch_idx, (batch, _) in tqdm.tqdm(enumerate(dataloader)):

        b1, b2 = batch

        if batch_size is None:
            batch_size = b1.shape[0]

        #tfm = RandomlySizedCrop()

        b1 = b1.to(DEVICE)
        b2 = b2.to(DEVICE)

        e1, z1 = model(b1)
        e2, z2 = model(b2)

        loss = loss_fn(z1, z2)

        optimizer.zero_grad()

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / n_batches


def test_loop(model, dataloader, loss_fn):
    model.eval()

    n_batches = len(dataloader)
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        batch_size = None
        for batch_idx, (batch, _) in tqdm.tqdm(enumerate(dataloader)):

            b1, b2 = batch

            if batch_size is None:
                batch_size = b1.shape[0]

            # tfm = RandomlySizedCrop()

            b1 = b1.to(DEVICE)
            b2 = b2.to(DEVICE)

            e1, z1 = model(b1)
            e2, z2 = model(b2)

            loss = loss_fn(z1, z2)
            total_loss += loss.item()

    return total_loss / n_batches# , total_acc / n_batches


def plot_example(
    model,
    dataloader,
    plot_name: str,
    in_channels: int = 3,
):

    f, axarr = plt.subplots(2, 4, sharex=True, figsize=(12, 8))
    model.eval()
    dataloader_iter = iter(dataloader)
    with torch.no_grad():

        for i in range(2):
            # grab first example
            xs, ys = next(dataloader_iter)
            # x1 = np.expand_dims(dataloader[0][0], axis=0)
            # x2 = np.expand_dims(dataloader[1][0], axis=0)
            x_view_1, x_view_2 = xs[0][0], xs[1][0]
            
            tfm = RandomlySizedCrop()
            x_view_1 = torch.unsqueeze(x_view_1, dim=0).cpu().numpy()
            # x_view_1 = tfm(x_view_1).cpu().numpy()
            x_view_2 = torch.unsqueeze(x_view_2, dim=0).cpu().numpy()
            # x_view_2 = tfm(x_view_2).cpu().numpy()
            
            if x_view_1.shape[1] < CHANNELS:
                sns.heatmap(x_view_1[0, 0, :, :], ax=axarr[i, 0])
                # sns.heatmap(x_view_1[0, 1, :, :], ax=axarr[i, 1], vmin=0, vmax=1)
                sns.heatmap(x_view_2[0, 0, :, :], ax=axarr[i, 2])
                # sns.heatmap(x_view_2[0, 1, :, :], ax=axarr[i, 3], vmin=0, vmax=1)
                axarr[i, 0].set_ylim(0, 200)
                axarr[i, 2].set_ylim(0, 200)
            else:
                x1 = np.transpose(x_view_1, (0, 2, 3, 1))
                x2 = np.transpose(x_view_2, (0, 2, 3, 1))

                axarr[i, 0].imshow(x1[0])
                axarr[i, 1].imshow(x2[0])

            for j in (0, 1):
                axarr[i, j].set_xticks([])
                axarr[i, j].set_yticks([])

    f.tight_layout()
    plt.subplots_adjust(wspace=0)
    f.savefig(plot_name, dpi=200)
    plt.close()


def collect_transformations(
    rng,
    height: int = 32,
    width: int = 32,
    channels_to_use: int = 3,
    major_minor: bool = False,
    repolarize_frac: float = 0.5,
    mask: float = 0.5,
):

    img_tfms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            PermuteHaplotypes(),
        ]

    if repolarize_frac > 0:
        img_tfms.append(
            RandomRepolarizationTransform(rng, frac_sites=repolarize_frac),
        )

    if major_minor:
        img_tfms.append(MajorMinorTransform())

    if mask > 0:
        img_tfms.append(RandomSiteMaskingTransform(rng, mask_ratio=mask))

    if channels_to_use < 3:
        img_tfms.append(ChannelSubset(n=channels_to_use))

    return transforms.Compose(img_tfms)



CHANNELS = 3
USE_FIRST_N = 2
LR = 1e-4
EPOCHS = 50
DEVICE = torch.device("cuda")

N_SNPS = 64
N_HAPS = 128

def main(config=None):

    # start a new wandb run to track this script
    with wandb.init(project="simclr-popgen", config=config):

        config = wandb.config

        batch_size = config["batch_size"]
        projection_dimension = config["projection_dimension"]
        encoding_dimension = config["encoding_dimension"]
        mask_fraction = config["mask_fraction"]
        agg = config["agg"]
        pool = config["pool"]
        shrink_kernel = config["shrink_kernel"]
        kernel_size = config["kernel_size"]
        stride = config["stride"]

        rng = np.random.default_rng(42)

        train_img_tfms = collect_transformations(
            rng,
            height=N_HAPS,
            width=N_SNPS,
            channels_to_use=1,
            major_minor=True,
            repolarize_frac=mask_fraction,
            mask=mask_fraction,
        )
        val_img_tfms = collect_transformations(
            rng,
            height=N_HAPS,
            width=N_SNPS,
            channels_to_use=1,
            major_minor=True,
            repolarize_frac=0,
            mask=0,
        )        

        # create train/test/val dataloaders
        train = torchvision.datasets.ImageFolder(
            "data/real/train/",
            transform=ContrastiveTransformations(train_img_tfms, n_views=2),
        )
        test = torchvision.datasets.ImageFolder(
            "data/real/test/",
            transform=ContrastiveTransformations(train_img_tfms, n_views=2),
        )
        val = torchvision.datasets.ImageFolder(
            "data/real/validation/",
            transform=val_img_tfms,
        )

        train_loader = DataLoader(
            dataset=train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_loader = DataLoader(
            dataset=test,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        # hidden_dims = [32, 64]

        # model = models.ContrastiveEncoder(
        #     in_channels=1,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     hidden_dims=hidden_dims,
        #     agg=agg,
        #     shrink_kernel=shrink_kernel,
        #     pool=pool,
        #     dropout=False,
        #     projection_dim=projection_dimension,
        #     encoding_dim=encoding_dimension,
        #     width = int(0.5 * N_SNPS),
        # )



        # print (model)

        model = model.to(DEVICE)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print (pytorch_total_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        loss_fn = losses.BarlowTwinsLoss(projection_dim=projection_dimension)

        reps, labels, acc = test_linear_clf(model, val_loader,)
        wandb.log(
            {
                "epoch": 0,
                "validation_acc": acc,
            }
        )

        for epoch in tqdm.tqdm(range(EPOCHS)):

            if epoch == 0:
                plot_example(
                    model,
                    test_loader,
                    plot_name="fig/reconstructions/0.png",
                    in_channels=USE_FIRST_N if USE_FIRST_N < CHANNELS else CHANNELS,
                )

            train_loss = train_loop(
                model,
                train_loader,
                loss_fn,
                optimizer,
            )

            test_loss = test_loop(
                model,
                test_loader,
                loss_fn,
            )

            reps, labels, acc = test_linear_clf(model, val_loader,)            
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    # "train_acc": train_acc,
                    # "test_acc": test_acc,
                    "validation_acc": acc,

                })
            
                

if __name__ == "__main__":

    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "salient_silhouette"},
        "parameters": {
            "mask_fraction": {"values": [0.5]},
            "batch_size": {"values": [128]},
            "projection_dimension": {"values": [128, 256]},
            "encoding_dimension": {"values": [128]},
            "agg": {"values": ["mean"]},
            "temperature": {"values": [0.1]},
            "pool": {"values": [True]},
            "shrink_kernel": {"values": [True]},
            "kernel_size": {"values": [7]},
            "stride": {"values": [2]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="simclr-popgen")

    wandb.agent(sweep_id, function=main)
