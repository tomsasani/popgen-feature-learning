import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import stdpopsim
import msprime

import models
from generator_fake import prep_simulated_region
import util
import demographies
import params
import global_vars

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
    batch_x, batch_y,
    plot_name: str,
):
    plt.rc("font", size=16)
    f, axarr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))

    pos, neg = np.where(batch_y == 1)[0][0], np.where(batch_y == 0)[0][0]
    for axi, i in enumerate((pos, neg)):

        x = torch.unsqueeze(batch_x[i, :, :, :], dim=0).cpu().numpy()
        # x = 1 - x

        # x = np.transpose(x, (0, 2, 3, 1)) * 255
        for ci in range(2):
            sns.heatmap(x[0, ci, :, :], ax=axarr[axi, ci], cmap="grey")
        # axarr[axi].set_xticks([])
        # axarr[axi].set_yticks([])
        # axarr[axi].set_xlabel("SNPs", size=14)
        # axarr[axi].set_ylabel("Haplotypes", size=14)

        # axarr[axi].set_title(
        #     r"$\rho = 1 \times 10^{-9}$" if axi == 0 else r"$\rho = 1 \times 10^{-8}$"
        # )
    f.tight_layout()
    f.savefig(plot_name, dpi=200)
    plt.close()


ITERATIONS = 1_000
DEVICE = torch.device("mps")

CONFIG = {
    "batch_size": 128,
    "kernel_size": 5,
    "conv_layers": 2,
    "lr": 1e-3,
    "batch_norm": False,
    "hidden_size": 192,
    "n_heads": 6,
    "depth": 1,
    "n_snps": 36,
    "n_smps": 99,
    "stride": 1,
    "pool": True,
    "agg": "max",
    "init_conv_dim": 32,
}


def decay(step: int):
    return 0.9 ** (step / ITERATIONS)


def warmup_plus_decay(step: int):
    warmup_steps = int(0.1 * ITERATIONS)
    rest_steps = ITERATIONS - warmup_steps
    if step < warmup_steps:
        return step / warmup_steps
    else:
        return 0.9 ** (step / rest_steps)


def main(args):

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

    engine = stdpopsim.get_default_engine()
    parameters = params.ParamSet()

    batch_size = CONFIG["batch_size"]
    kernel_size = CONFIG["kernel_size"]
    conv_layers = CONFIG["conv_layers"]
    use_vit = args.use_vit
    lr = CONFIG["lr"]
    batch_norm = CONFIG["batch_norm"]
    hidden_size = CONFIG["hidden_size"]
    n_heads = CONFIG["n_heads"]
    depth = CONFIG["depth"]
    n_snps = CONFIG["n_snps"]
    n_smps = CONFIG["n_smps"]
    stride = CONFIG["stride"]
    pool = CONFIG["pool"]
    agg = CONFIG["agg"]
    init_conv_dim = CONFIG["init_conv_dim"]

    samples_per_population = [0, 0, 0]
    # use a CEU population
    samples_per_population[1] = n_smps
    samples = dict(
        zip(
            ["YRI", "CEU", "CHB"],
            samples_per_population,
        )
    )

    if use_vit:

        model = models.BabyTransformer(
            width=n_snps,
            in_channels=2,
            num_classes=2,
            hidden_size=hidden_size,
            num_heads=n_heads,
            depth=depth,
            attention_pool=False,
            mlp_ratio=2,
        )

    else:
        hidden_dims = []
        for _ in range(conv_layers):
            hidden_dims.append(init_conv_dim)
            init_conv_dim *= 2

        model = models.BasicPredictor(
            in_channels=2,
            kernel=(1, kernel_size),
            hidden_dims=hidden_dims,
            agg="max",
            encoding_dim=hidden_size,
            num_classes=2,
            width=n_snps,
            batch_norm=batch_norm,
            padding=0,
            stride=stride,
            pool=pool,
        )

    model = model.to(DEVICE)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0,
    )

    scheduler = None
    if use_vit:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_plus_decay)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay)

    rng = np.random.default_rng(42)

    loss_fn = torch.nn.CrossEntropyLoss()

    res = []
    for iteration in tqdm.tqdm(range(ITERATIONS)):

        batch_x, batch_y = [], []

        # create a batch
        counted = 0
        while counted < batch_size:
            # pick random demo
            ci = rng.choice(2)

            background = rng.uniform(5e-9, 2e-8)
            heat = 1 if ci == 0 else rng.uniform(10, 100)

            hs_s, hs_e = (
                (global_vars.L - global_vars.HOTSPOT_L) / 2,
                (global_vars.L + global_vars.HOTSPOT_L) / 2,
            )
            rho_map = msprime.RateMap(
                position=[
                    0,
                    hs_s,
                    hs_e,
                    global_vars.L,
                ],
                rate=[background, background * heat, background],
            )

            ts = demographies.simulate_exp(
                parameters,
                [n_smps],
                rho_map,
                rng,
                seqlen=global_vars.L,
                plot=False,
            )

            # ts = msprime.simulate(
            #     sample_size=n_smps * 2,
            #     Ne=1e4,
            #     recombination_map=rho_map,
            #     mutation_rate=1.1e-8,
            # )

            # ts = engine.simulate(
            #     demography,
            #     contig=contigs[ci],
            #     samples=samples,
            # )

            X, positions = prep_simulated_region(
                ts,
                filter_singletons=False,
            )

            # figure out if we have enough SNPs on either side of the hotspot
            if ci == 1:
                hs_idxs = np.where((positions >= hs_s) & (positions <= hs_e))
                # need at least 5 SNPs with hotspot
                if hs_idxs[0].shape[0] < 1:
                    # print (hs_s, hs_e)
                    # print (positions)
                    continue

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

            n_batches_zero_padded = util.check_for_missing_data(
                np.expand_dims(region, axis=0)
            )
            if n_batches_zero_padded > 0:
                continue
            # batch_x.append(np.expand_dims(region[0, :, :], axis=0))
            batch_x.append(np.expand_dims(region[:2, :, :], axis=0))
            batch_y.append(ci)
            counted += 1

        batch_x = np.concatenate(batch_x)
        batch_y = np.array(batch_y)

        batch_x = torch.from_numpy(batch_x)# .unsqueeze(dim=1)
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
                batch_x, batch_y,
                plot_name="fig/reconstructions/0.png",
            )
        if iteration % 10 == 0:
            print (f"on iteration {iteration}, loss = {train_loss} and acc = {train_acc}")

        d = {
                "iteration": iteration,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "n_params": pytorch_total_params,
            }

        res.append(d)
    res = pd.DataFrame(res)

    res.to_csv(f"csv/o.{args.use_vit}.tsv", sep="\t", index=False)

    f, ax = plt.subplots()
    sns.scatterplot(data=res, x="iteration", y="train_acc", ax=ax)
    f.tight_layout()
    f.savefig(f"fig/{args.use_vit}.png")

if __name__ == "__main__":

    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-use_vit", action="store_true")
    args = p.parse_args()

    main(args)
