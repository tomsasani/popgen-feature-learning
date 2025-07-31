import util


import numpy as np
from PIL import Image
import stdpopsim


def prep_simulated_region(
    ts,
    filter_singletons: bool = False,
) -> np.ndarray:
    """Gets simulated data ready. Returns a matrix of size
    (n_haps, n_sites, 6)"""

    # n_snps x n_haps
    X = ts.genotype_matrix().astype(np.int8)

    site_table = ts.tables.sites
    positions = site_table.position.astype(np.int64)

    seg = util.find_segregating_idxs(
        X,
        filter_singletons=filter_singletons,
    )

    X = X[seg, :]
    positions = positions[seg]

    assert positions.shape[0] == X.shape[0]

    return X, positions

# Choose species and model
species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("OutOfAfrica_3G09")

# Get a chromosome (e.g., chromosome 22)
# chrom = species.genome.get_chromosome("chr22")

samples_per_population = [0, 0, 0]
# samples_per_population[pcs] = N_SMPS

pop2i = dict(zip(["YRI", "CEU", "CHB"], range(3)))
popi = pop2i[snakemake.wildcards.population]

samples_per_population[popi] = 100
samples = dict(zip(["YRI", "CEU", "CHB"], samples_per_population))

# Simulate tree sequence
engine = stdpopsim.get_default_engine()  # msprime by default

rng = np.random.default_rng(42)


counted = 0

outfh = open(snakemake.output.fh, "w")

while counted < 20_000:

    contig = species.get_contig(
        length=100_000,
        recombination_rate=rng.uniform(1e-8, 1.5e-8),
        mutation_rate=2.35e-8,
    )

    ts = engine.simulate(model, contig=contig, samples=samples)

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
        n_snps=int(snakemake.wildcards.n_snps),
        norm_len=50_000,
        convert_to_diploid=False,
    )

    n_batches_zero_padded = util.check_for_missing_data(
        np.expand_dims(region, axis=0)
    )
    if n_batches_zero_padded > 0:
        continue

    if counted % 100 == 0:
        print(counted)

    region = np.transpose(region, (1, 2, 0))
    region = np.uint8(region * 255)

    img = Image.fromarray(region, mode="RGB")

    img.save(f"{snakemake.params.pref}/{snakemake.wildcards.population}/{snakemake.wildcards.chrom}/{snakemake.wildcards.n_snps}/{counted}.png")

    print (f"{counted}", file=outfh)

    counted += 1
