from scripts.real_data_iterator import RealData
import util

import numpy as np
from PIL import Image


rng = np.random.default_rng(123)

engine = RealData(
    snakemake.input.vcf,
    snakemake.input.ped,
    snakemake.input.exclude,
    pop_code=snakemake.wildcards.population,
    convert_to_diploid=False,
    convert_to_rgb=True,
    sort=False,
    seqlen=50_000,
    n_snps=int(snakemake.wildcards.n_snps),
    rng=rng,
    filter_singletons=False,
)

outfh = open(snakemake.output.fh, "w")

start = 1
# if we need to zero pad a region, it's because we ran out of sites
zero_padded = False
while not zero_padded:

    region, positions, mutations = engine.sample_real_region(
        snakemake.wildcards.chrom,
        start,
        end=None,
        keep=None,
    )
    # print (region)
    if region is None:
        start = positions[-1]
        continue

    n_batches_zero_padded = util.check_for_missing_data(
        np.expand_dims(region, axis=0),
    )
    if n_batches_zero_padded > 0: 
        zero_padded = True

    region = np.transpose(region, (1, 2, 0))
    region = np.uint8(region * 255)

    img = Image.fromarray(region, mode="RGB")

    region_str = f"{snakemake.wildcards.chrom}_{start}_{positions[-1]}"

    img.save(f"{snakemake.params.pref}/{snakemake.wildcards.population}/{snakemake.wildcards.chrom}/{snakemake.wildcards.n_snps}/{region_str}.png")

    print (f"{snakemake.wildcards.chrom}:{start}-{positions[-1]}", file=outfh)

    start = positions[-1]
