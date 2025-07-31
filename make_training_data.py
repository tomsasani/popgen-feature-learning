import demographies
import generator_fake
import params
import util
import global_vars
from generator_fake import prep_simulated_region

import msprime

import os
import glob
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import stdpopsim

def ts2img(ts):
    
    X, positions = prep_simulated_region(
        ts,
        filter_singletons=False,
    )

    # X, _ = util.major_minor(X.T)
    X = X.T
    ref_alleles = np.zeros(X.shape[1])

    region = util.process_region(
        X,
        positions,
        ref_alleles,
        convert_to_rgb=True,
        n_snps=N_SNPS,
        norm_len=global_vars.L,
        convert_to_diploid=DIPLOID,
    )

    n_batches_zero_padded = util.check_for_missing_data(
        np.expand_dims(region, axis=0)
    )
    if n_batches_zero_padded > 0:
        return None

    region = np.transpose(region, (1, 2, 0))
    region = np.uint8(region * 255)

    if RGB:
        img = Image.fromarray(region, mode="RGB")
    else:
        img = Image.fromarray(region[:, :, 0], mode="L")
    return img

N_REGIONS = 5_000

N_SMPS = 100
N_SNPS = 64
RGB = False
DIPLOID = False

# Choose species and model
species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("OutOfAfrica_3G09")

# Get a chromosome (e.g., chromosome 22)
chrom = species.genome.get_chromosome("chr22")

normal_rho_map = msprime.RateMap(
                position=[0, global_vars.L * 2],
                rate=[1e-8],
            )
hs_s = (global_vars.L // 2) - global_vars.HOTSPOT_L
hs_e = (global_vars.L // 2) + global_vars.HOTSPOT_L

weird_rho_map = msprime.RateMap(
                position=[0, hs_s, hs_e, global_vars.L],
                rate=[1e-8, 1e-8 * 50, 1e-8],
            )

normal_contig = species.get_contig(
    # chromosome=chrom,
    length=global_vars.L,
    recombination_rate=1e-8,
    mutation_rate=2.35e-8,
) 
weird_contig = species.get_contig(
    # chromosome=chrom,
    length=global_vars.L,
    recombination_rate=1e-7,
    mutation_rate=2.35e-8,
) 

samples_per_population = [0, 0, 0]
# samples_per_population[pcs] = N_SMPS
samples_per_population[1] = N_SMPS
normal_samples = dict(zip(["YRI", "CEU", "CHB"], samples_per_population))
samples_per_population = [0, 0, 0]
# samples_per_population[pcs] = N_SMPS
samples_per_population[0] = N_SMPS
weird_samples = dict(zip(["YRI", "CEU", "CHB"], samples_per_population))

# Simulate tree sequence
engine = stdpopsim.get_default_engine()  # msprime by default


rng = np.random.default_rng(42)

for fh in tqdm.tqdm(glob.glob(f"data/simulated/train/0/*.png")):
    os.remove(fh)
for fh in tqdm.tqdm(glob.glob(f"data/simulated/test/0/*.png")):
    os.remove(fh)

counted = 0
while counted < N_REGIONS:
    ts = engine.simulate(model, contig=normal_contig, samples=normal_samples)

    img = ts2img(ts)
    if img is None: continue
    
    img.save(f"data/simulated/train/0/{counted}.png")

    counted += 1

for i in (0, 1):
    for fh in tqdm.tqdm(glob.glob(f"data/simulated/validation/{i}/*.png")):
        os.remove(fh)


counted = 0
while counted < N_REGIONS:

    class_label = int(counted % 10 == 0)

    ts = engine.simulate(
        model,
        contig=normal_contig if class_label == 0 else weird_contig,
        samples=normal_samples,
    )

    img = ts2img(ts)
    if img is None: continue

    img.save(f"data/simulated/validation/{class_label}/{counted}.png")

    counted += 1


# OPTIONAL: make few-shot examples for SSD

counted = 0

while counted < 10:

    ts = engine.simulate(
        model,
        contig=weird_contig,
        # contig=normal_contig,
        samples=normal_samples,
    )

    img = ts2img(ts)
    if img is None: continue

    img.save(f"data/simulated/ood/0/{counted}.png")

    counted += 1

