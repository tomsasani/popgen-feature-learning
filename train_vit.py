import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision
from torcheval.metrics import functional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from transformers import ViTModel, ViTConfig
from transformers import get_cosine_schedule_with_warmup, RoFormerConfig, RoFormerModel
import transforms
import models
from typing import Union, List, Tuple
import math
import wandb

from itertools import chain
from generator_fake import prep_simulated_region
import stdpopsim
import msprime
import util

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
    # dataloader,
    batch_x,
    batch_y,
    loss_fn,
    optimizer,
    scheduler
):

    model.train()

    # n_batches = len(dataloader)
    # total_loss = 0
    # total_acc = 0
    # batch_size = None
    # for batch_idx, (batch, y) in enumerate(dataloader):

    # if batch_size is None:
    batch_size = batch_x.shape[0]

    batch_x = batch_x.to(DEVICE)
    # B, C, H, W
    z1 = model(batch_x)

    loss = loss_fn(z1, batch_y.to(DEVICE))
    acc = torch.sum(torch.argmax(z1, dim=1) == batch_y.to(DEVICE)) / batch_size

    optimizer.zero_grad()

    # total_loss += loss.item()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item(), acc.item()


def test_loop(model, dataloader, loss_fn, crop_test_set: bool = False,):
    model.eval()

    n_batches = len(dataloader)
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        batch_size = None
        for batch_idx, (batch, y) in enumerate(dataloader):


            if batch_size is None:
                batch_size = batch.shape[0]

            batch = batch.to(DEVICE)
            if crop_test_set:
                batch = transforms.RandomlySizedCrop()(batch)

            z1 = model(batch)

            acc = torch.sum(torch.argmax(z1, dim=1) == y.to(DEVICE)) / batch_size
            f1 = functional.multiclass_f1_score(z1, y.to(DEVICE))
            total_acc += acc.item()
            loss = loss_fn(z1, y.to(DEVICE))

            total_loss += loss.item()

    return total_loss / n_batches, total_acc / n_batches


def plot_example(
    minibatch,
    plot_name: str,
    crop: bool = False,
):
    plt.rc("font", size=16)
    f, axarr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 8))

    idxs = [0, 80]
    for axi, i in enumerate(idxs):

        x = torch.unsqueeze(minibatch[i, :, :, :], dim=0).cpu().numpy()
        x = 1 - x

        x = np.transpose(x, (0, 2, 3, 1)) * 255
        axarr[axi].imshow(x[0], cmap="grey")

        axarr[axi].set_xticks([])
        axarr[axi].set_yticks([])
        axarr[axi].set_xlabel("SNPs", size=14)
        axarr[axi].set_ylabel("Haplotypes", size=14)

        axarr[axi].set_title(
            r"$\rho = 1 \times 10^{-7}$" if axi == 0 else r"$\rho = 1 \times 10^{-8}$"
        )
    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()




# class ColumnViTClassifier(torch.nn.Module):
#     def __init__(
#         self,
#         vit_model_name: str = "facebook/deit-tiny-patch16-224",
#         # vit_model_name: str = "google/vit-base-patch16-224",
#         width: int = 32,
#         in_channels: int = 1,
#         layers_to_unfreeze: int = 2,
#         num_classes: int = 2,
#         attention_pool: bool = False,
#     ):
#         super().__init__()

#         self.vit = ViTModel.from_pretrained(vit_model_name)
#         for param in self.vit.parameters():
#             param.requires_grad = False

#         self.hidden_size = self.vit.config.hidden_size
#         self.pooling = AttentionPooling(self.hidden_size)
#         self.attention_pool = attention_pool

#         # we use a "custom" tokenizer that takes
#         # patches of size (C * W), where W is the
#         # number of SNPs
#         self.tokenizer = ColumnTokenizer(
#             input_channels=in_channels,
#             input_width=width,
#             hidden_size=self.hidden_size,
#         )

#         # transformer encoder blocks
#         self.encoder = self.vit.encoder
#         self.norm = self.vit.layernorm

#         # linear classifier head
#         self.classifier = torch.nn.Linear(
#             self.hidden_size,
#             num_classes,
#         )

#         if layers_to_unfreeze == -1:
#             for layer in self.encoder.layer:
#                 for param in layer.parameters():
#                     param.requires_grad = True
#             for param in self.norm.parameters():
#                 param.requires_grad = True
#         elif layers_to_unfreeze > 0:
#             for layer in self.encoder.layer[-1 * layers_to_unfreeze:]:
#                 for param in layer.parameters():
#                     param.requires_grad = True
#             for param in self.norm.parameters():
#                 param.requires_grad = True
#         else:
#             pass

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.tokenizer(x)  # (B, H, hidden_size)

#         # pass through transformer encoder
#         x = self.encoder(x)[0]
#         # pass through final layernorm
#         x = self.norm(x)

#         # classification head on the average or attention-pooled
#         # final embeddings (no CLS token)
#         if self.attention_pool:
#             cls_output = self.pooling(x)
#         else:
#             cls_output = torch.mean(x, dim=1)

#         logits = self.classifier(cls_output)  

#         return logits




ITERATIONS = 1_000
N_SMPS = 32
DEVICE = torch.device("cuda")

def main(config=None):

    # Choose species and model
    species = stdpopsim.get_species("HomSap")
    demography = species.get_demographic_model("OutOfAfrica_3G09")
    chrom = species.genome.get_chromosome("chr22")

    RHO = [1.08e-9, 1.17e-8, 1.34e-7]

    contigs = [
        species.get_contig(
            # chromosome=chrom,
            length=50_000,
            recombination_rate=r,
            mutation_rate=2.35e-8,
        )
        for r in RHO
    ]

    samples_per_population = [0, 0, 0]
    # samples_per_population[pcs] = N_SMPS
    samples_per_population[1] = N_SMPS
    samples = dict(zip(["YRI", "CEU", "CHB"], samples_per_population))

    # Simulate tree sequence
    engine = stdpopsim.get_default_engine()

    # start a new wandb run to track this script
    with wandb.init(project="tfmr-popgen", config=config):

        config = wandb.config

        batch_size = config["batch_size"]
        agg = config["agg"]
        kernel_size = config["kernel_size"]
        n_hidden = config["n_hidden"]
        use_vit = config["use_vit"]
        lr1, lr2 = config["lr1"], config["lr2"]
        batch_norm = config["batch_norm"]
        pool = config["pool"]
        hidden_size = config["hidden_size"]
        n_heads = config["n_heads"]
        depth = config["depth"]
        n_snps = config["n_snps"]

        if use_vit:
            
            model = models.BabyTransformer(
                width=n_snps,
                in_channels=1,
                num_classes=3,
                hidden_size=hidden_size,
                num_heads=n_heads,
                depth=depth,
                attention_pool=False,
                mlp_ratio=2,
            )

        else:
            hidden_dims = []
            cur_dim = 32
            for _ in range(n_hidden):
                hidden_dims.append(cur_dim)
                cur_dim *= 2

            kernel = (1, kernel_size)

            model = models.BasicPredictor(
                in_channels=1,
                kernel=kernel,
                hidden_dims=hidden_dims,
                agg=agg,
                encoding_dim=hidden_size,
                projection_dim=3,
                width = n_snps,
                pool=pool,
                batch_norm=batch_norm,
                padding=0,
            )

        model = model.to(DEVICE)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print (pytorch_total_params)


        optimizer = torch.optim.AdamW(model.parameters(), lr=lr1, weight_decay=0)
        scheduler = None
        
        loss_fn = torch.nn.CrossEntropyLoss()

        res = []
        for iteration in tqdm.tqdm(range(ITERATIONS)):

            batch_x, batch_y = [], []

            # create a batch
            for ci, contig in enumerate(contigs):

                counted = 0
                while counted < (batch_size // 3):

                    ts = engine.simulate(
                        demography,
                        contig=contig,
                        samples=samples,
                    )

                    X, positions = prep_simulated_region(
                        ts,
                        filter_singletons=False,
                    )

                    X = util.major_minor(X.T)

                    ref_alleles = np.zeros(X.shape[1])

                    region = util.process_region(
                        X,
                        positions,
                        ref_alleles,
                        convert_to_rgb=True,
                        n_snps=n_snps,
                        norm_len=1_000,
                        convert_to_diploid=False,
                    ) 

                    n_batches_zero_padded = util.check_for_missing_data(np.expand_dims(region, axis=0))
                    if n_batches_zero_padded > 0:
                        continue

                    batch_x.append(np.expand_dims(region[0, :, :], axis=0))
                    batch_y.append(ci)
                    counted += 1

            batch_x = np.concatenate(batch_x)
            batch_y = np.array(batch_y)

            batch_x = torch.from_numpy(batch_x).unsqueeze(dim=1)
            batch_y = torch.from_numpy(batch_y)

            train_loss, train_acc = train_loop(
                model,
                batch_x,
                batch_y,
                loss_fn,
                optimizer,
                scheduler,
            )
            if iteration == 0:
                plot_example(
                    batch_x,
                    plot_name="fig/reconstructions/0.png",
                    
                )

            
            if iteration % 5 == 0:
                wandb.log(
                    {
                        "iteration": iteration,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "n_params": pytorch_total_params,
                    }
                )
            d = {
                    "iteration": iteration,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "n_params": pytorch_total_params,
                    "run_name": wandb.run.name,
                }
            d.update(config)
            res.append(d)
        res = pd.DataFrame(res)
        res.to_csv(f"csv/{wandb.run.name}.tsv", sep="\t", index=False)


if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-use_vit", action="store_true")
    args = p.parse_args()

    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "salient_silhouette"},
        "parameters": {
            "use_vit": {"values": [args.use_vit]},
            "n_snps": {"values": [36]},
            "batch_size": {"values": [240]},
            "n_hidden": {"values": [None] if args.use_vit else [2]},
            "hidden_size": {"values": [192]},
            "n_heads": {"values": [6, 12] if args.use_vit else [None]},
            "depth": {"values": [2] if args.use_vit else [None]},
            "pool": {"values": [None] if args.use_vit else [True]},
            "batch_norm": {"values": [None] if args.use_vit else [False, True]},
            "agg": {"values": ["mean"]},
            "major_minor": {"values": [False]},
            "shrink_kernel": {"values": [False]},
            "kernel_size": {"values": [None] if args.use_vit else [5]},
            "stride": {"values": [2]},
            "lr1": {"values": [1e-3]},
            "lr2": {"values": [3e-5] if args.use_vit else [None]},
            "crop_test_set": {"values": [False]},
            # "attention_pool": {"values": [False] if args.use_vit else [None]}
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="tfmr-popgen")

    wandb.agent(sweep_id, function=main)
