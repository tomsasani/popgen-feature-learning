import util

import numpy as np
from PIL import Image
import global_vars
import stdpopsim
import matplotlib.pyplot as plt
import seaborn as sns

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

    # seg = util.find_segregating_idxs(
    #     X,
    #     filter_singletons=filter_singletons,
    # )

    # X = X[seg, :]
    # positions = positions[seg]

    assert positions.shape[0] == X.shape[0]

    return X, positions

# whether to simulate selection at all
simulate_sweep = True if int(snakemake.wildcards.raw_coef) > 0 else False

rng = np.random.default_rng(int(snakemake.params.seed))

# Choose species and model
species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("OutOfAfrica_3G09")

chrom = snakemake.params.chrom

coef = int(snakemake.wildcards.raw_coef) / 1_000

# dfe = species.get_dfe("Gamma_K17")
# exons = species.get_annotations("ensembl_havana_104_exons")
# exon_intervals = exons.get_chromosome_annotations(chrom)
# contig.add_dfe(intervals=exon_intervals, DFE=dfe)
engine = stdpopsim.get_engine("slim") 

extended_events = None
if simulate_sweep:
    debug = model.model.debug()
    coal_T = debug.mean_coalescence_time({snakemake.wildcards.population: 2})
    gamma = coal_T * coef #2Ns
    T_f = (
        4 * (np.log(gamma) + 0.5772 - (1 / gamma)) / coef
    )  
    print (T_f, coef)
    extended_events = stdpopsim.selection.selective_sweep(
        single_site_id="sweep",
        population="CEU",
        selection_coeff=coef,
        mutation_generation_ago=T_f,
        min_freq_at_end=0.5,
    )

pop_sizes = [0, 0, 0]
pop2i = {"YRI": 0, "CEU": 1, "CHB": 2}
pop_sizes[pop2i[snakemake.wildcards.population]] = int(snakemake.params.n_haps)
samples = dict(zip(pop2i.keys(), pop_sizes))
print (samples)

outfh = open(snakemake.output.fh, "w")

counted = 0
while counted < int(snakemake.params.n_replicates):

    left = rng.integers(0, 200_000_000)
    right = left + 100_000
    coord = (left + right) // 2

    contig = species.get_contig(
        chrom,
        left=left,
        right=right,
        genetic_map="HapMapII_GRCh37",
        mutation_rate=2.35e-8,
    )
    if simulate_sweep:
        # add the site
        contig.add_single_site(id="sweep", coordinate=coord)

    engine_seed = int(rng.integers(0, 2**32))

    # https://github.com/popsim-consortium/analysis2/blob/main/workflows/sweep_simulate.snake#L662
    # https://github.com/popsim-consortium/analysis2/blob/20136e89ef279ed28fff503e5377ee1f88461712/workflows/config/snakemake/sweep_config.yaml#L37
    ts = engine.simulate(
        model,
        contig,
        samples,
        extended_events=extended_events,
        seed=engine_seed,
        slim_scaling_factor=2,
        slim_burn_in=2,
    )

    X, positions = prep_simulated_region(
        ts,
        filter_singletons=False,
    )

    X[X > 1] = 1


    print (positions.shape, np.where(positions == coord))

    # f, ax = plt.subplots()
    # sns.heatmap(X.T, ax=ax)
    # f.savefig(f"{snakemake.params.pref}/{counted}.raw.png")

    dists = util.inter_snp_distances(positions, norm_len=1)

    ref_alleles = np.zeros_like(dists)

    region = util.process_region(
        X.T,
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

    region = np.transpose(region, (1, 2, 0))
    region = np.uint8(region * 255)

    img = Image.fromarray(region, mode="RGB")

    print (f"{chrom}:{left}-{right}\t{counted}", file=outfh)

    img.save(f"{snakemake.params.pref}/{counted}.png")

    counted += 1
