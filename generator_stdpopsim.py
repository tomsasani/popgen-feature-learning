"""
Generator class for pg-gan.
Author: Sara Matheison, Zhanpeng Wang, Jiaping Wang, Rebecca Riley
Date: 9/27/22
"""

# python imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# our imports
import global_vars
import params
import demographies
import util
from typing import List, Union


class Generator:
    def __init__(
        self,
        simulator,
        param_names,
        seed,
        convert_to_rgb: bool = False,
        permute: bool = False,
        n_snps: int = 32,
        seqlen: int = global_vars.L,
        filter_singletons: bool = False,
        convert_to_diploid: bool = False,
        sort: bool = False,
        n_pops: int = 1,
    ):
        self.simulator = simulator
        self.param_names = param_names
        self.rng = np.random.default_rng(seed)
        self.convert_to_rgb = convert_to_rgb
        self.n_snps = n_snps
        self.seqlen = seqlen
        self.filter_singletons = filter_singletons
        self.convert_to_diploid = convert_to_diploid
        self.curr_params = None
        self.sort = sort
        self.n_pops = n_pops

    def sample_fake_region(
        self,
        sample_sizes: List[int],
        param_values: List[Union[float, int]] = [],
        treat_as_real: bool = False,
    ):

        region = None

        ts = self.simulator(
            sample_sizes,
            seqlen=self.seqlen,
        )

        X, positions = prep_simulated_region(
            ts,
            filter_singletons=self.filter_singletons,
        )

        region = util.process_region(
            X,
            positions,
            convert_to_rgb=self.convert_to_rgb,
            n_snps=self.n_snps,
            norm_len=global_vars.L,
            convert_to_diploid=self.convert_to_diploid,
        )

        if self.sort:
            region = util.sort_min_diff_numpy(
                region,
            )

        # region = util.normalize_ndarray(region)

        region = np.expand_dims(region, axis=0)

        return region


def prep_simulated_region(ts, filter_singletons: bool = False) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    # n_snps x n_haps
    X = ts.genotype_matrix().astype(np.int8)

    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)

    seg = util.find_segregating_idxs(X, filter_singletons=filter_singletons)

    X = X[seg, :]
    positions = positions[seg]

    assert positions.shape[0] == X.shape[0]

    return X, positions


# testing
if __name__ == "__main__":
    batch_size = 10
    parameters = params.ParamSet()

    # quick test
    generator = Generator(
        demographies.simulate_exp,
        ["rho"],
        42,
        convert_to_diploid=True,
        filter_singletons=False,
        n_snps=32,
        sort=True,
        permute=False,
        convert_to_rgb=True,
        seqlen=50_000,
    )

    region = generator.sample_fake_region(
        [64],
        param_values=[1e-8],
    )
    print("x", region.shape)

    f, axarr = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(8, 8))
    for channel_i in range(3):
        sns.heatmap(
            region[0, channel_i, :, :],
            ax=axarr[channel_i, 0],
        )

    for channel_i in range(3):
        sns.heatmap(
            region[0, channel_i, :, :],
            ax=axarr[channel_i, 1],
        )
    f.tight_layout()
    f.savefig("test.png", dpi=200)
