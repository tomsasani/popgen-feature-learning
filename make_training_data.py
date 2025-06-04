import demographies
import generator_fake
import params
import util

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


PARAM_NAMES = ["rho"]
PARAM_NAMES = ["N1", "N2", "T1", "T2"]

# initialize basic engine
engine = generator_fake.Generator(
    demographies.simulate_exp,
    PARAM_NAMES,
    42,
    convert_to_rgb=True,
    permute=False,
    n_snps=64,
    convert_to_diploid=False,
    seqlen=50_000,
    sort=True,
    filter_singletons=False,
)

sim_params = params.ParamSet()
rng = np.random.default_rng(42)

PARAM_VALUES = [
    [23_231, 29_962, 4_870, 581],  # YRI
    [22_552, 3_313, 3_589, 1_050],  # CEU
    # [9_000, 5_000, 1_500, 350],  # CHB
]

# PARAM_VALUES = [[1e-8], [1e-9], [5e-9]]


N_SMPS = 32
total_regions = 15_000 / len(PARAM_VALUES)

for model_i in range(len(PARAM_VALUES)):
    counted = 0
    while counted < total_regions:

        param_values = PARAM_VALUES[model_i]

        region = engine.sample_fake_region(
            [N_SMPS],
            param_values=param_values,
        )

        n_batches_zero_padded = util.check_for_missing_data(region)
        if n_batches_zero_padded > 0:
            continue

        if counted % 100 == 0:
            print(counted)

        if counted == 0:
            f, axarr = plt.subplots(3)
            sns.heatmap(region[0, 0, :, :], ax=axarr[0])
            sns.heatmap(region[0, 1, :, :], ax=axarr[1])
            sns.heatmap(region[0, 2, :, :], ax=axarr[2])
            f.savefig("region.png")
            plt.close()

        region = np.transpose(region[0, :, :, :], (1, 2, 0))
        region = region[:, :, 0]
        region = np.uint8(region * 255)

        img = Image.fromarray(region, mode="L")

        if counted == 0:
            f, axarr = plt.subplots(3)
            sns.heatmap(img.getchannel("L"), ax=axarr[0])
            f.savefig("region.img.png")
            plt.close()

        if counted < int(total_regions * 0.8):
            img.save(f"data/simulated/train/{model_i}/{counted}.png")
        else:
            img.save(f"data/simulated/test/{model_i}/{counted}.png")

        counted += 1
