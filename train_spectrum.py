import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import stdpopsim
import msprime
import wandb

import models
from generator_fake import prep_simulated_region
import util
import global_vars
import losses
from real_data_iterator import RealData

MUT_PROBS = [0.05, 0.35, 0.1, 0.05, 0.05, 0.4,]


def create_minibatch(
    engine,
    demography,
    contig,
    samples,
    n_snps: int = 36,
):
    # simulate training data
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
        norm_len=global_vars.L,
        convert_to_diploid=False,
    )
    return region


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
    mutations = rng.choice(6, size=ac.shape[0], p=MUT_PROBS)

    # then figure out the prob that we switch C>As
    switch_probs = rng.uniform(size=low_ac_sites.shape[0])
    sites_to_change = low_ac_sites[switch_probs <= assign_prob]
    mutations[sites_to_change] = 1
    return mutations


def create_mutation_views(batch_x, batch_y):
    # figure out indices of each mutation type in each
    # training example in the batch
    views = []
    for mut in torch.arange(6):#torch.unique(batch_y.flatten()):
        bi, mi = torch.where(batch_y != mut)
        x_copy = batch_x.detach().clone()
        x_copy[bi, :, :, mi] = 0
        views.append(x_copy)
    return torch.cat(views)


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

    B, C, H, W = batch_x.shape

    # get 6 views for each image
    views = create_mutation_views(batch_x, batch_y)

    e, z = model(views, return_embeds=True)
    _, D = z.shape
    # reshape so there are 6 views per embedding
    z = z.reshape((B, 6, D))

    loss = loss_fn(z)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()

def test_loop(
    model,
    batch_x,
    batch_y,
    loss_fn,
):

    model.eval()

    batch_x = batch_x.to(DEVICE)

    B, C, H, W = batch_x.shape

    # get 6 views for each image
    views = create_mutation_views(batch_x, batch_y)

    e, z = model(views, return_embeds=True)
    _, D = z.shape
    # reshape so there are 6 views per embedding
    z = z.reshape((B, 6, D))

    loss = loss_fn(z)

    mid = B // 2

    return loss#loss[:mid].cpu(), loss[mid:].cpu()

def plot_example(
    model,
    batch_x,
    batch_y,
    plot_name: str,
):
    plt.rc("font", size=16)
    f, axarr = plt.subplots(1, 6, sharex=True, sharey=False, figsize=(24, 8))
    model.eval()
    with torch.no_grad():
        for bi, b in enumerate(batch_x):
            if bi > 0: break
            views = create_mutation_views(
                b.unsqueeze(dim=0),
                batch_y[bi].unsqueeze(dim=0),
            )
            for ci in range(6):
                sns.heatmap(views[ci, 0, :, :], ax=axarr[ci])
            # z = model(views)
            # loss = loss_fun

    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()


ITERATIONS = 1_000
DEVICE = torch.device("cuda")


def decay(step: int):
    return 0.9 ** (step / ITERATIONS)


def warmup_plus_decay(step: int):
    warmup_steps = int(0.1 * ITERATIONS)
    rest_steps = ITERATIONS - warmup_steps
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return 0.9 ** (step / rest_steps)


VCF_FH = "/scratch/ucgd/lustre-core/common/data/1KG_VCF/1KG.chrALL.phase3_v5a.20130502.genotypes.vcf.gz"
PED_FH = "data/igsr_samples.tsv"
BED_FH = "data/LCR-hs37d5.bed.gz"

def main(config=None):

    # start a new wandb run to track this script
    with wandb.init(project="spectrum-popgen", config=config):

        config = wandb.config

        batch_size = config["batch_size"]
        lr = config["lr"]
        hidden_size = config["hidden_size"]
        n_heads = config["n_heads"]
        depth = config["depth"]
        n_snps = config["n_snps"]
        n_smps = config["n_smps"]
        agg = config["agg"]

        # batch_size = 32
        # n_snps = 36
        # n_smps = 100
        # hidden_size = 128
        # n_heads = 8
        # depth = 1
        # agg = "max"
        # lr = 1e-3

        model = models.TinyTransformer(
            width=n_snps,
            in_channels=1,
            num_classes=128,
            hidden_size=hidden_size,
            num_heads=n_heads,
            depth=depth,
            mlp_ratio=2,
            agg=agg,
        )

        real_data = RealData(
            vcf_fh=VCF_FH,
            ped_fh=PED_FH,
            bed_fh=BED_FH,
            pop_code="CEU",
            convert_to_rgb=True,
            n_snps=n_snps,
            sort=False,
        )

        # Choose species and model
        species = stdpopsim.get_species("HomSap")
        demography = species.get_demographic_model("OutOfAfrica_3G09")

        contig = species.get_contig(
                length=50_000,
                recombination_rate=1e-8,
                mutation_rate=2.35e-8,
            )

        engine = stdpopsim.get_default_engine()

        samples_per_population = [0, 0, 0]
        # use a CEU population
        samples_per_population[1] = n_smps
        samples = dict(
            zip(
                ["YRI", "CEU", "CHB"],
                samples_per_population,
            )
        )

        length = real_data.chrom2len["22"]

        model = model.to(DEVICE)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay)
        loss_fn = losses.SpectrumVICReg()

        rng = np.random.default_rng(1234)

        

        res = []
        for iteration in tqdm.tqdm(range(ITERATIONS)):

            batch_x, batch_y = [], []

            # create a batch
            counted = 0
            while counted < batch_size:

                # region, positions, mutations = real_data.sample_real_region(
                #     "22",
                #     rng.integers(0, length),
                # )
                # if region is None:
                #     continue

                region = create_minibatch(
                    engine,
                    demography,
                    contig,
                    samples,
                    n_snps=n_snps,
                )

                # randomly sample mutations
                mutations = rng.choice(6, size=n_snps, p=MUT_PROBS)
                n_batches_zero_padded = util.check_for_missing_data(
                    np.expand_dims(region, axis=0)
                )

                if n_batches_zero_padded > 0:
                    continue
                # batch_x.append(np.expand_dims(region[:2], axis=0))
                batch_x.append(np.expand_dims(region[0], axis=(0, 1)))

                batch_y.append(mutations)
                counted += 1

            batch_x = np.concatenate(batch_x)
            batch_y = np.array(batch_y)

            batch_x = torch.from_numpy(batch_x).to(torch.float32)
            batch_y = torch.from_numpy(batch_y)

            # if iteration == 0:
            #     plot_example(model, batch_x, batch_y, f"fig/predictions/{iteration}.png")

            train_loss = train_loop(
                model,
                batch_x,
                batch_y,
                loss_fn,
                optimizer,
                scheduler,
            )

            # DO TESTING #

            batch_x, batch_y = [], []

            # create a batch
            counted = 0
            while counted < batch_size:

                region = create_minibatch(
                    engine,
                    demography,
                    contig,
                    samples,
                    n_snps=n_snps,
                )

                if rng.uniform() < 1:
                    mutations = assign_biased_mutations(rng, region)
                else:
                    mutations = rng.choice(6, size=n_snps, p=MUT_PROBS)
                n_batches_zero_padded = util.check_for_missing_data(
                    np.expand_dims(region, axis=0)
                )

                if n_batches_zero_padded > 0:
                    continue
                # batch_x.append(np.expand_dims(region[:2], axis=0))
                batch_x.append(np.expand_dims(region[0], axis=(0, 1)))

                batch_y.append(mutations)
                counted += 1

            batch_x = np.concatenate(batch_x)
            batch_y = np.array(batch_y)

            batch_x = torch.from_numpy(batch_x).to(torch.float32)
            batch_y = torch.from_numpy(batch_y)

            test_loss = test_loop(
                model,
                batch_x,
                batch_y,
                loss_fn,
            )
            if iteration % 10 == 0 and iteration > 0:
                wandb.log(
                    {
                        "iteration": iteration,
                        "train_loss": train_loss,
                        "test_loss": test_loss,
                    }
                )


if __name__ == "__main__":

    
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "salient_silhouette"},
        "parameters": {
            "batch_size": {"values": [64]},
            "agg": {"values": ["max"]},
            "lr": {"values": [1e-3]},
            "hidden_size": {"values": [128]},
            "n_heads": {"values": [8]},
            "depth": {"values": [1]},
            "n_snps": {"values": [64]},
            "n_smps": {"values": [100]},
            "include_dists": {"values": [False]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="simclr-popgen")

    wandb.agent(sweep_id, function=main)
