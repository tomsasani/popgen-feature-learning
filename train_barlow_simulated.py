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

MUT_PROBS = [0.05, 0.35, 0.1, 0.05, 0.05, 0.4,]
MUT_PROBS = [0.5, 0.5]
# MUT_PROBS = [0.17] * 4 + [0.16] * 2
N_MUTS = len(MUT_PROBS)


def assign_biased_mutations(rng, region):
    # get allele counts at each site in the region
    ac = np.sum(region[0, :, :] == 1, axis=0)
    # assert ac.shape[0] == 36
    # figure out low ACs
    low_ac_sites = np.where(ac <= 1)[0]
    # at low AC sites, what's the prob we make it a
    # single mutation type only?
    assign_prob = 1
    # randomly assign mtuations to start
    mutations = rng.choice(N_MUTS, size=ac.shape[0], p=MUT_PROBS)

    # then figure out the prob that we switch C>As
    switch_probs = rng.uniform(size=low_ac_sites.shape[0])
    sites_to_change = low_ac_sites[switch_probs <= assign_prob]
    mutations[sites_to_change] = 0
    return mutations

def get_outlier_score(cluster_center, cluster_cov, embeddings):
    dist = DistanceMetric.get_metric('mahalanobis', V=cluster_cov)
    return dist.pairwise(cluster_center, embeddings)


def train_loop(
    model,
    minibatch,
    loss_fn,
    optimizer,
    tfms,
    
):

    model.train()

    assert minibatch.min() == 0 and minibatch.max() == 1

    B, C, H, W = minibatch.shape

    # get 2 views for each image
    v1 = tfms(minibatch).to(DEVICE)
    v2 = tfms(minibatch).to(DEVICE)

    assert v1.max() == 1 and v1.min() == 0

    z1 = model(v1)
    z2 = model(v2)

    loss = loss_fn(z1, z2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

    return loss.item()


def plot_example(
    batch,
    tfms,
    plot_name: str,
):
    plt.rc("font", size=16)
    f, axarr = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(24, 8))
    B, C, H, W = batch.shape

    # get 2 views for each image
    v1 = tfms(batch)
    v2 = tfms(batch)
    
    for bi, b in enumerate(v1):
        if bi > 0: break
        sns.heatmap(v1[0, 0, :, :].cpu(), ax=axarr[0])
        sns.heatmap(v2[0, 0, :, :].cpu(), ax=axarr[1])

    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()

class RealDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, regions, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform

        self.region_list = []
        with open(regions, "r") as infh:
            for l in infh:
                self.region_list.append(l.strip())

        path_list = glob.glob(image_dir + "*")
        self.path_list = [p.split("/")[-1] for p in path_list]
        #print (path_list)
        self.path2region = dict(zip(path_list, self.region_list))

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir,
                                self.path_list[idx])
        image = io.imread(img_name)

        if self.transform:
            image = self.transform(image)

        return image, self.region_list[idx]


EPOCHS = 101
DEVICE = torch.device("cuda")


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
    with wandb.init(project="spectrum-popgen", config=config):

        config = wandb.config

        batch_size = config["batch_size"]
        lr = config["lr"]
        representation_dim = config["representation_dim"]
        projector_dim = config["projector_dim"]
        n_snps = config["n_snps"]
        agg = config["agg"]
        mask_frac = config["mask_frac"]
        representation_layer = config["representation_layer"]
        stride = config["stride"]
        lmbda = config["lmbda"]
        hidden_layers = config["hidden_layers"]
        kernel_size = config["kernel_size"]
        use_ssdk = config["use_ssdk"]

        hidden_dims = [32]
        for _ in range(hidden_layers - 1):
            cur_dim = hidden_dims[-1]
            hidden_dims.append(cur_dim * 2)

        model = models.DeFinetti(
            in_channels=1,
            kernel=(1, kernel_size),
            hidden_dims=hidden_dims,
            stride=stride,
            agg=agg,
            width=n_snps,
            projector_dim=projector_dim,
            representation_dim=representation_dim,
            fc_representation=representation_layer,

        )


        orig_tfms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            transforms.ChannelSubset(),
        ]

        rng = np.random.default_rng(1234)

        barlow_tfms = v2.Compose(
            [
                transforms.RandomlySizedCrop(min_width=n_snps, on_batch=True),
                transforms.RandomRepolarizationTransform(rng, frac_sites=mask_frac),
            ]
        )

        ssdk_tfms = v2.Compose(
            [
                transforms.RandomlySizedCrop(min_width=n_snps, on_batch=True),
                transforms.RandomRepolarizationTransform(rng, frac_sites=mask_frac),
            ]
        )

        # create train/test/val dataloaders
        train = torchvision.datasets.ImageFolder(
            "data/simulated/validation/",
            transform=v2.Compose(orig_tfms),
        )

        val = torchvision.datasets.ImageFolder(
            "data/simulated/validation/",
            transform=v2.Compose(
                orig_tfms
                + [transforms.RandomlySizedCrop(min_width=n_snps, on_batch=False)],
            ),
        )

        ood = torchvision.datasets.ImageFolder("data/simulated/ood/", transform=v2.Compose(orig_tfms))

        train_loader = DataLoader(
            dataset=train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_loader = DataLoader(
            dataset=val,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # for few-shot SSD
        ood_loader = DataLoader(
            dataset=ood,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        model = model.to(DEVICE)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1.5e-6,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=decay,
        )

        loss_fn = losses.BarlowTwinsLoss(lmbda=lmbda)

        res = []
        for epoch in tqdm.tqdm(range(EPOCHS)):
            model.eval()

            if epoch % 50 == 0:
                X, y = [], []
                for bi, (batch, labels) in enumerate(val_loader):
                    with torch.no_grad():
                        embeddings, _ = model(batch.to(DEVICE), return_embeds=True)
                        X.append(embeddings)
                        y.append(labels)
                X, y = torch.cat(X).cpu().numpy(), torch.cat(y).cpu().numpy()

                # get the covariance matrix of the in-distribution data
                in_distro_idxs = np.where(y == 0)
                in_distro = X[in_distro_idxs]
                in_distro_cov = np.cov(in_distro, rowvar=False)
                # fit a k-means on the in-distribution data
                kmeans = KMeans(n_clusters=1, random_state=1234)
                kmeans.fit(in_distro)
                # get the k-means cluster centers
                center = kmeans.cluster_centers_
                scores = get_outlier_score(center, in_distro_cov, X)[0]

                # if we want to use few-shot SSD, we need to also calculate the distance
                # between each point and the sample mean of the known OOD data. since we only
                # have a few samples in the OOD data, we use a shrunk covariance matrix.
                X_ood = []
                for bi, (batch, labels) in enumerate(ood_loader):
                    # apply random transformations
                    batch = ssdk_tfms(batch)
                    with torch.no_grad():
                        embeddings, _ = model(batch.to(DEVICE), return_embeds=True)
                        X_ood.append(embeddings)
                X_ood = torch.cat(X_ood).cpu().numpy()

                shrunk_ood_cov = ShrunkCovariance().fit(X_ood).covariance_

                # fit a k-means on the OOD data
                kmeans = KMeans(n_clusters=1, random_state=1234)
                kmeans.fit(X_ood)
                # get the k-means cluster centers
                center_ood = kmeans.cluster_centers_

                # compute scores between all input points and the OOD
                scores_ood = get_outlier_score(center_ood, shrunk_ood_cov, X)[0]
                assert scores.shape[0] == scores_ood.shape[0]

                if use_ssdk:
                    scores = scores - scores_ood

                f, axarr = plt.subplots(2, sharex=True)
                _min, _max = np.min(scores), np.max(scores)
                for l in np.unique(y):
                    li = np.where(y == l)
                    axarr[l].hist(scores[li], bins=np.linspace(_min, _max, 100), label=l)
                f.savefig(f"fig/{wandb.run.name}.{epoch}.hist.png")

                # boostrap resample ROC
                f, ax = plt.subplots()
                roc_bs = []
                for trial in range(1_000):
                    bs_idxs = rng.choice(
                        y.shape[0],
                        size=y.shape[0],
                        replace=True,
                    )

                    roc_auc = roc_auc_score(y[bs_idxs].reshape(-1, 1), scores[bs_idxs].reshape(-1, 1))
                    fpr, tpr, _ = roc_curve(y[bs_idxs].reshape(-1, 1), scores[bs_idxs].reshape(-1, 1))
                    if trial % 10 == 0:
                        ax.plot(fpr, tpr, c="gainsboro", alpha=0.25)
                    roc_bs.append(roc_auc)

                    d = {
                        "run_name": wandb.run.name,
                        "epoch": epoch,
                        "trial": trial,
                        "roc_auc": roc_auc,
                    }
                    d.update(config)
                    res.append(d)
                ax.axline((0, 0), slope=1, c="k", ls=":")
                f.savefig(f"fig/{wandb.run.name}.{epoch}.roc.png")
                wandb.log(
                    {
                        "epoch": epoch,
                        "roc_auc": np.mean(roc_bs),
                        
                    }
                )

            epoch_train_loss = 0

            for bi, (batch, _) in enumerate(train_loader):

                if epoch == 0 and bi == 0:
                    plot_example(batch, barlow_tfms, "fig/0.png")
                train_loss = train_loop(
                    model,
                    batch,
                    loss_fn,
                    optimizer,
                    barlow_tfms,
                    
                )
                epoch_train_loss += train_loss

            if scheduler is not None:
                scheduler.step()
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss / len(train_loader),
                    "last_lr": scheduler.get_last_lr()[0] if scheduler is not None else lr,
                }
            )
        res = pd.DataFrame(res)
        res.to_csv(f"csv/{wandb.run.name}.tsv", sep="\t", index=False)


if __name__ == "__main__":


    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "salient_silhouette"},
        "parameters": {
            "batch_size": {"values": [512]},
            "lmbda": {"values": [5e-3]},
            "agg": {"values": ["max"]},
            "hidden_layers": {"values": [2, 4]},
            "mask_frac": {"values": [0.25]},
            "lr": {"values": [1e-3]},
            "padding": {"values": [False]},
            "representation_dim": {"values": [32]},
            "n_snps": {"values": [32]},
            "projector_dim": {"values": [128]},
            "include_dists": {"values": [False]},
            "representation_layer": {"values": [True]},
            "use_ssdk": {"values": [True, False]},
            "stride": {"values": [1]},
            "kernel_size": {"values": [5, 7]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="barlow-popgen")

    wandb.agent(sweep_id, function=main)
