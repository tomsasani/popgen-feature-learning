import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import stdpopsim
import math
import wandb

import models
from generator_fake import prep_simulated_region
import util


def train_loop(
    model,
    batch_x,
    batch_y,
    loss_fn,
    optimizer,
    scheduler
):

    model.train()

    batch_size = batch_x.shape[0]

    batch_x = batch_x.to(DEVICE)

    z1 = model(batch_x)
    loss = loss_fn(z1, batch_y.to(DEVICE))
    acc = torch.sum(torch.argmax(z1, dim=1) == batch_y.to(DEVICE)) / batch_size

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item(), acc.item()


def plot_example(
    minibatch,
    plot_name: str,
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
            r"$\rho = 2 \times 10^{-9}$" if axi == 0 else r"$\rho = 1 \times 10^{-8}$"
        )
    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()


ITERATIONS = 1_000
N_SMPS = 32
DEVICE = torch.device("cuda")

def main(config=None):

    # Choose species and model
    species = stdpopsim.get_species("HomSap")
    demography = species.get_demographic_model("OutOfAfrica_3G09")

    RHO = [1e-9, 1e-8, 1e-7]

    contigs = [
        species.get_contig(
            length=50_000,
            recombination_rate=r,
            mutation_rate=2.35e-8,
        )
        for r in RHO
    ]

    samples_per_population = [0, 0, 0]
    # use a CEU population
    samples_per_population[1] = N_SMPS
    samples = dict(zip(["YRI", "CEU", "CHB"], samples_per_population))

    # Simulate tree sequence
    engine = stdpopsim.get_default_engine()

    # start a new wandb run to track this script
    with wandb.init(project="tfmr-popgen", config=config):

        config = wandb.config

        batch_size = config["batch_size"]
        kernel_size = config["kernel_size"]
        n_hidden = config["n_hidden"]
        use_vit = config["use_vit"]
        lr = config["lr"]
        batch_norm = config["batch_norm"]
        hidden_size = config["hidden_size"]
        n_heads = config["n_heads"]
        depth = config["depth"]
        n_snps = config["n_snps"]
        use_padding = config["use_padding"]
        init_conv_dim = config["init_conv_dim"]

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
            for _ in range(n_hidden):
                hidden_dims.append(init_conv_dim)
                init_conv_dim *= 2

            model = models.BasicPredictor(
                in_channels=1,
                kernel=(1, kernel_size),
                hidden_dims=hidden_dims,
                agg="mean",
                encoding_dim=hidden_size,
                num_classes=3,
                width=n_snps,
                batch_norm=batch_norm,
                padding=math.floor(kernel_size / 2) if use_padding else 0,
            )

        model = model.to(DEVICE)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0,
        )
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
                        norm_len=1,
                        convert_to_diploid=False,
                    )

                    n_batches_zero_padded = util.check_for_missing_data(
                        np.expand_dims(region, axis=0)
                    )
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
            "lr": {"values": [1e-3]},
            "hidden_size": {"values": [48, 96, 192]},
            "init_conv_dim": {"values": [32, 64]},
            "n_hidden": {"values": [None] if args.use_vit else [2]},
            "n_heads": {"values": [6] if args.use_vit else [None]},
            "depth": {"values": [1, 2] if args.use_vit else [None]},
            "batch_norm": {"values": [None] if args.use_vit else [False, True]},
            "kernel_size": {"values": [None] if args.use_vit else [5]},
            "use_padding": {"values": [None] if args.use_vit else [True, False]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="tfmr-popgen")

    wandb.agent(sweep_id, function=main)
