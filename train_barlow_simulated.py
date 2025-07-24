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
from sklearn.metrics import roc_auc_score

import models
from generator_fake import prep_simulated_region
import util
import transforms
import torchvision.transforms.v2 as v2
import global_vars
import losses
from real_data_iterator import RealData
import params
import demographies
import scipy.stats as ss

MUT_PROBS = [0.05, 0.35, 0.1, 0.05, 0.05, 0.4,]
MUT_PROBS = [0.5, 0.5]
# MUT_PROBS = [0.17] * 4 + [0.16] * 2
N_MUTS = len(MUT_PROBS)


def calculate_scon_dist(train_embeds: np.ndarray, test_embeds: np.ndarray):
    vals = []
    cossims = cosine_similarity(test_embeds, train_embeds)
    np.fill_diagonal(cossims, 0)

    for ei, e in enumerate(test_embeds):
        cur_max = 0
        # get cossims to this embedding
        e_cossim = cossims[ei]
        # get the norm
        norm = np.linalg.norm(e)
        # get max of the cossim * norms
        vals.append(np.max(e_cossim * norm))
    return vals


def get_crop_idx(rng, width: int = 200, min_w: int = 32):
    return rng.integers(0, width - min_w)

def get_random_w_idxs(rng, width: int = 200, frac: float = 0.5):
    return rng.choice(width, replace=False, size=int(width * frac))


def transform_minibatch(
    rng,
    minibatch,
    mutations_onehot,
    n_snps: int = 32,
    mask_frac: float = 0.5,
):
    assert minibatch.min() == 0 and minibatch.max() == 1
    X, y = [], []
    for bi, b in enumerate(minibatch):
        m = mutations_onehot[bi]
        # 1. randomly subset the region
        wi = get_crop_idx(rng, width=b.shape[-1], min_w=n_snps)
        _b, _m = b[:, :, wi:wi+n_snps], m[wi:wi+n_snps, :]
        # 2. randomly repolarize sites -- different sites in each example
        # repolarize_idxs = get_random_w_idxs(rng, width=n_snps, frac=mask_frac)
        # _b[0, :, repolarize_idxs] = 1 - _b[0, :, repolarize_idxs]
        # 3. convert 0s to -1s
        _b[_b == 0] = -1
        # 4. randomly mask sites
        mask_idxs = get_random_w_idxs(rng, width=n_snps, frac=mask_frac)
        _b[:, :, mask_idxs] = 0
        # _m[mask_idxs, :] = 0
        X.append(_b.unsqueeze(0))
        y.append(_m.unsqueeze(0))
    return torch.cat(X), torch.cat(y)


def create_minibatch(
    rng,
    # demographic_params,
    rho_map,
    n_snps: int = 36,
    n_smps: int = 100,
    batch_size: int = 64,
):
    counted = 0

    X_batch = []

    while counted < batch_size:

        # ts = demographies.simulate_exp(
        #     demographic_params,
        #     [n_smps],
        #     rho_map,
        #     rng,
        #     seqlen=global_vars.L,
        #     plot=False,
        # )
        ts = msprime.simulate(
            sample_size=n_smps * 2,
            Ne=1e4,
            recombination_map=rho_map,
            mutation_rate=1.1e-8,
            random_seed=rng.integers(0, 2**32),
        )
        # simulate training data
        # ts = engine.simulate(
        #     demography,
        #     contig=contig,
        #     samples=samples,
        #     mutation_model=msprime.BinaryMutationModel(state_independent=False),
        # )

        X, positions = prep_simulated_region(
            ts,
            filter_singletons=False,
        )

        X, _ = util.major_minor(X.T)

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

        # region = util.sort_min_diff_numpy(region)

        X_batch.append(np.expand_dims(region[0], axis=(0, 1)))
        counted += 1

    X_batch = np.concatenate(X_batch)
    return X_batch


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


def make_batch(
    rng,
    # demographic_params,
    rho_map,
    batch_size: int = 32,
    biased_mutations: bool = False,
    n_snps: int = 36,
    n_smps: int = 100,
):
    ### NORMAL ###
    # samples_per_population = [0, 0, 0]
    # samples_per_population[pop_idx] = n_smps
    # samples = dict(
    #     zip(
    #         ["YRI", "CEU", "CHB"],
    #         samples_per_population,
    #     )
    # )

    minibatch = create_minibatch(
            rng,
            # demographic_params,
            rho_map,
            n_smps=n_smps,
            n_snps=n_snps,
            batch_size=batch_size,
        )

    mutations = []
    for bi in range(batch_size):
        _batch = minibatch[bi]
        if biased_mutations:
            _mutations = assign_biased_mutations(rng, _batch)
        else:
            _mutations = rng.choice(N_MUTS, size=n_snps, p=MUT_PROBS)
        mutations.append(_mutations)

    mutations = np.vstack(mutations)
    mutations = torch.from_numpy(mutations)

    # one-hot encode the mutations
    mutations_onehot = torch.nn.functional.one_hot(mutations, num_classes=N_MUTS)
    minibatch = torch.from_numpy(minibatch)

    assert minibatch.max() == 1 and minibatch.min() == 0
    return minibatch.to(torch.float32), mutations_onehot.to(torch.float32)


def train_loop(
    rng,
    model,
    minibatch,
    mutations_onehot,
    loss_fn,
    optimizer,
    scheduler,
    mask_frac: float = 0.5,
    n_snps: int = 32,
):

    model.train()

    minibatch = minibatch
    mutations_onehot = mutations_onehot.to(torch.float32)

    assert minibatch.min() == 0 and minibatch.max() == 1

    B, C, H, W = minibatch.shape

    # get 2 views for each image
    v1, m1 = transform_minibatch(
        rng,
        minibatch.detach().clone(),
        mutations_onehot.detach().clone(),
        n_snps=n_snps,
        mask_frac=mask_frac,
    )

    v2, m2 = transform_minibatch(
        rng,
        minibatch.detach().clone(),
        mutations_onehot.detach().clone(),
        n_snps=n_snps,
                mask_frac=mask_frac,

    )

    assert v1.max() == 1 and v1.min() == -1

    z1 = model(v1, m1)
    z2 = model(v2, m2)

    loss = loss_fn(z1, z2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.item()


# def test_loop(
#         rng,
#     model,
#     batch_x,
#     batch_y,
#     loss_fn,
# ):

#     model.eval()

#     batch_x = batch_x.to(DEVICE)

#     B, C, H, W = batch_x.shape
#     (v1, v2), (m1, m2) = create_mutation_views_barlow(rng, batch_x, batch_y)

#     #e, z = model(views, return_embeds=True)
#     e1, z1 = model(v1, return_embeds=True)
#     e2, z2 = model(v2, return_embeds=True)

#     loss = loss_fn(z1, z2)

#     mid = B // 2

#     return loss


def plot_example(
    rng,
    batch_x,
    batch_y,
    plot_name: str,
    n_snps: int = 32,
            mask_frac: float = 0.5,

):
    plt.rc("font", size=16)
    f, axarr = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(24, 8))

    # get 2 views for each image
    v1, m1 = transform_minibatch(
        rng,
        batch_x.detach().clone(),
        batch_y.detach().clone(),
        n_snps=n_snps,
        mask_frac=mask_frac,
    )
    v2, m2 = transform_minibatch(
        rng,
        batch_x.detach().clone(),
        batch_y.detach().clone(),
        n_snps=n_snps,
                mask_frac=mask_frac,

    )
    for bi, b in enumerate(v1):
        if bi > 0: break
        sns.heatmap(batch_x[0, 0, :, :], ax=axarr[0, 0])
        sns.heatmap(b[0, :, :], ax=axarr[1, 0])
        sns.heatmap(m1[0, :, :].T, ax=axarr[1, 1])
        sns.heatmap(v2[0, 0, :, :], ax=axarr[2, 0])
        sns.heatmap(m2[0, :, :].T, ax=axarr[2, 1])

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
        representation_dim = config["representation_dim"]
        projector_dim = config["projector_dim"]
        n_heads = config["n_heads"]
        depth = config["depth"]
        n_snps = config["n_snps"]
        n_smps = config["n_smps"]
        agg = config["agg"]
        mask_frac = config["mask_frac"]

        model = models.TinyTransformer(
            width=n_snps,
            height=n_smps * 2,
            in_channels=1,
            num_classes=N_MUTS,
            hidden_size=128,
            representation_dim=representation_dim,
            projector_dim=projector_dim,
            num_heads=n_heads,
            depth=depth,
            mlp_ratio=2,
            agg=agg,
            how="row"
        )

        # model = models.DeFinetti(
        #     in_channels=1,
        #     kernel=(1, 5),
        #     hidden_dims=[32, 64],
        #     agg="max",
        #     width=n_snps,
        #     num_classes=128,
        #     fc_dim=128,
        # )

        demographic_params = params.ParamSet() 

        rng = np.random.default_rng(1234)

        ### MAKE SOME TEST DATA ###
        rho_map = msprime.RateMap(
                    position=[0, global_vars.L // 2],
                    rate=[1e-8],
                )

        test_minibatch_normal, test_mutations_normal = make_batch(
            rng,
            # demographic_params,
            rho_map,
            batch_size=batch_size * 2,
            biased_mutations=False,
            n_snps=n_snps,
            n_smps=n_smps,
        )

        hs_s, hs_e = (
            (global_vars.L // 2 - global_vars.HOTSPOT_L) / 2,
            (global_vars.L // 2 + global_vars.HOTSPOT_L) / 2,
        )
        rho_map = msprime.RateMap(
            position=[
                0,
                hs_s,
                hs_e,
                global_vars.L // 2,
            ],
            rate=[1e-8, 5e-8, 1e-8],
        )

        test_minibatch_weird, test_mutations_weird = make_batch(
            rng,
            # demographic_params,
            rho_map,
            batch_size=batch_size * 2,
            biased_mutations=False,
            n_snps=n_snps,
            n_smps=n_smps,
        )
        ####

        model = model.to(DEVICE)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay)
        loss_fn = losses.BarlowTwinsLoss()
        # loss_fn = losses.SimCLRLoss()

        res = []
        for iteration in tqdm.tqdm(range(ITERATIONS)):

            if iteration % 50 == 0:

                normal_embeddings, weird_embeddings = None, None
                model.eval()
                with torch.no_grad():
                    normal_embeddings, _ = model(test_minibatch_normal.to(DEVICE), test_mutations_normal.to(DEVICE), return_embeds=True)
                    weird_embeddings, _ = model(test_minibatch_weird.to(DEVICE), test_mutations_weird.to(DEVICE), return_embeds=True)

                embeddings = torch.cat((normal_embeddings, weird_embeddings), dim=0).cpu().numpy()
                labels = torch.cat((torch.zeros(batch_size * 2), torch.ones(batch_size * 2)))

                f, ax = plt.subplots()
                clf = PCA()
                X_new = clf.fit_transform(embeddings)
                for l in torch.unique(labels):
                    li = torch.where(labels == l)
                    ax.scatter(X_new[li, 0], X_new[li, 1], label=l)
                f.legend()
                f.savefig("pca.png")

                # X_train, X_test, y_train, y_test = train_test_split(
                #     embeddings,
                #     labels,
                #     random_state=42,
                # )

                clf = LogisticRegression(max_iter=1_000)
                # clf.fit(X_train, y_train)
                # roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

                cv_results = cross_validate(clf, embeddings, labels, cv=5)
                print (cv_results["test_score"])
                for si, s in enumerate(cv_results["test_score"]):
                    wandb.log(
                        {
                            "iteration": iteration,
                            f"score_{si}": s,
                        }
                    )

            rho_map = msprime.RateMap(
                position=[0, global_vars.L],
                rate=[1e-8],
            )

            train_minibatch, train_mutations = make_batch(
                rng,
                # demographic_params,
                rho_map,
                batch_size=batch_size,
                biased_mutations=False,
                n_snps=n_snps * 2,
                n_smps=n_smps,
            )
            assert train_minibatch.min() == 0 and train_minibatch.max() == 1

            if iteration == 0:
                plot_example(rng, train_minibatch, train_mutations, "fig/0.png", mask_frac=mask_frac,)

            train_loss = train_loop(
                rng,
                model,
                train_minibatch.to(DEVICE),
                train_mutations.to(DEVICE),
                loss_fn,
                optimizer,
                scheduler,
                n_snps=n_snps,
                mask_frac=mask_frac,
            )

            # if iteration % 50 == 0:
            #     f, ax = plt.subplots()
            #     ax.scatter(X_new[:half_n, 0], X_new[:half_n, 1])
            #     ax.scatter(X_new[half_n:, 0], X_new[half_n:, 1])
            #     f.savefig(f"fig/ttest/{iteration}.png")
            #     plt.close()
            if iteration % 10 == 0:
                wandb.log(
                    {
                        "iteration": iteration,
                        "train_loss": train_loss,
                    }
                )


if __name__ == "__main__":

    
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "salient_silhouette"},
        "parameters": {
            "batch_size": {"values": [128]},
            "agg": {"values": ["max"]},
            "mask_frac": {"values": [0.5]},
            "lr": {"values": [1e-3]},
            "representation_dim": {"values": [32]},
            "n_heads": {"values": [8]},
            "depth": {"values": [1]},
            "n_snps": {"values": [32]},
            "n_smps": {"values": [100]},
            "projector_dim": {"values": [512]},
            "include_dists": {"values": [False]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="barlow-popgen")

    wandb.agent(sweep_id, function=main)
