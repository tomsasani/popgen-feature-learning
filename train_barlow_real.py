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
from line_profiler import LineProfiler
from sklearn.covariance import ShrunkCovariance

from collections import defaultdict
from bx.intervals.intersection import Interval, IntervalTree

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

        path_list = glob.glob(image_dir + "*.png")
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

    # read in selection data
    selected_regions = pd.read_excel(
        "data/1-s2.0-S0092867413000871-mmc1.xlsx", sheet_name="CMSGW"
    ).rename(
        columns={
            "Unnamed: 9": "chrom",
            "Unnamed: 10": "start",
            "Unnamed: 11": "end",
        }
    )
    selected_regions = selected_regions[selected_regions["Population"].str.contains("CEU")]

    # store regions we've classified
    selection_tree = defaultdict(IntervalTree)
    for i, row in selected_regions.iterrows():
        chrom, start, end = row["chrom"], row["start"], row["end"]
        interval = Interval(int(start), int(end))
        selection_tree[chrom.lstrip("chr")].insert_interval(interval)

    # start a new wandb run to track this script
    # with wandb.init(project="spectrum-popgen", config=config):

    # config = wandb.config

    batch_size = 512#config["batch_size"]
    lr = 1e-3#config["lr"]
    representation_dim = 32#config["representation_dim"]
    projector_dim = 512#config["projector_dim"]
    n_snps = 32#config["n_snps"]
    agg = "max"#config["agg"]
    mask_frac = 0.25#config["mask_frac"]
    representation_layer = True#config["representation_layer"]
    stride = 1#config["stride"]

    model = models.DeFinetti(
        in_channels=1,
        kernel=(1, 5),
        hidden_dims=[32, 64],
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

    # train on real data
    train = RealDataset(
        "data/real/CEU/2/64/",
        "data/real/CEU/2/64/regions.txt",
        transform=v2.Compose(orig_tfms),
    )
    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # train on real data
    val = RealDataset(
        "data/real/CEU/2/64/",
        "data/real/CEU/2/64/regions.txt",
        transform=v2.Compose(
            orig_tfms + [transforms.RandomlySizedCrop(min_width=n_snps, on_batch=False)]
        ),
    )
    val_loader = DataLoader(
            dataset=val,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

    # small OOD of simulated data with selection
    ood = torchvision.datasets.ImageFolder(
        "data/slim/CEU/64/",
        transform=v2.Compose(orig_tfms),
    )
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
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=decay,
    )

    loss_fn = losses.BarlowTwinsLoss()

    res = []
    for epoch in tqdm.tqdm(range(EPOCHS)):

        if epoch % 20 == 0:

            all_regions = []

            model.eval()

            # get array of embeddings for IID data
            X, is_selected = [], []
            for bi, (batch, region) in tqdm.tqdm(enumerate(val_loader)):
                with torch.no_grad():
                    embeddings, _ = model(batch.to(DEVICE), return_embeds=True)
                    X.append(embeddings)
                    for reg in region:
                        chrom, se = reg.split(":")
                        start, end = list(map(int, se.split("-")))
                        overlaps_selected = len(selection_tree[chrom].find(start, end)) > 0
                        is_selected.append(int(overlaps_selected))

            X = torch.cat(X).cpu().numpy()
            is_selected = np.array(is_selected)

            # kmeans outlier detection
            in_distro_cov = np.cov(X, rowvar=False)

            kmeans = KMeans(n_clusters=1, random_state=0)
            kmeans.fit(X)
            # get cluster center
            center = kmeans.cluster_centers_
            # calculate distances between every point and the IID
            # cluster center from kmeans
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

            ssdk_scores = scores - scores_ood

            hst_fig, hst_axarr = plt.subplots(2, 2)
            for mi, metric in enumerate((scores, ssdk_scores)):
                _min, _max = np.min(metric), np.max(metric)
                bins = np.linspace(_min, _max, 100)
                for l in np.unique(is_selected):
                    li = np.where(is_selected == l)
                    hst_axarr[mi, li].hist(metric[li], bins=bins, histtype="step")
            hst_fig.tight_layout()
            hst_fig.savefig(f"fig/real/{epoch}.hist.png", dpi=200)

            roc_fig, roc_axarr = plt.subplots(2)
            # calculate ROC AUC using either raw scores or ssdk scores
            for mi, (metric_name, metric) in enumerate(zip(("ssd", "ssdk"), (scores, ssdk_scores))):

                roc_bs = []
                for trial in range(1_000):
                    # get bootstrapping indexes
                    bs_idxs = rng.choice(
                        metric.shape[0],
                        size=metric.shape[0],
                        replace=True,
                    )

                    roc_auc = roc_auc_score(is_selected[bs_idxs].reshape(-1, 1), metric[bs_idxs].reshape(-1, 1))
                    fpr, tpr, _ = roc_curve(is_selected[bs_idxs].reshape(-1, 1), metric[bs_idxs].reshape(-1, 1))
                    if trial % 10 == 0:
                        roc_axarr[mi].plot(fpr, tpr, c="gainsboro", alpha=0.25)
                    roc_bs.append(roc_auc)

                roc_axarr[mi].set_title(np.mean(roc_bs))
                roc_axarr[mi].axline((0, 0), slope=1, c="k", ls=":")
            roc_fig.tight_layout()
            roc_fig.savefig(f"fig/real/{epoch}.roc.png", dpi=200)

        epoch_train_loss = 0
        print ("Training on in-distribution data")
        for bi, (batch, _) in tqdm.tqdm(enumerate(train_loader)):
            print (f"Done with {bi + 1} of {len(train_loader)}")
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
        print (f"On epoch {epoch}, train loss is {epoch_train_loss / len(train_loader)}")
        if scheduler is not None:
            scheduler.step()

    res = pd.DataFrame(res)


if __name__ == "__main__":


    # sweep_configuration = {
    #     "method": "grid",
    #     "name": "sweep",
    #     "metric": {"goal": "maximize", "name": "salient_silhouette"},
    #     "parameters": {
    #         "batch_size": {"values": [512]},
    #         "agg": {"values": ["sum"]},
    #         "mask_frac": {"values": [0.25]},
    #         "lr": {"values": [1e-3]},
    #         "padding": {"values": [False]},
    #         "representation_dim": {"values": [32]},
    #         "n_snps": {"values": [32]},
    #         "projector_dim": {"values": [512]},
    #         "include_dists": {"values": [False]},
    #         "representation_layer": {"values": [True]},
    #         "stride": {"values": [1, 2]},
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="barlow-popgen")

    # wandb.agent(sweep_id, function=main)
    # lp = LineProfiler()
    # lp_wrapper = lp(main)
    # lp_wrapper()
    # lp.print_stats()
    main()
