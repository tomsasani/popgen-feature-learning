CHROMS = list(map(str, range(1, 23)))
CHROMS = [f"chr{c}" for c in CHROMS]

POPS = ["CEU"]
N_SNPS = [32, 64]

rule all:
    input:
        # raw coef will be divided by 1_000
        expand("data/slim/CEU/64/{raw_coef}/regions.txt", raw_coef = [10, 25, 50])

rule create_slim_chrom_images:
    input:
    output: fh = "data/slim/{population}/{n_snps}/{raw_coef}/regions.txt"
    params:
        seed = 1234,
        chrom = "chr2",
        n_haps = 100,
        pref = "data/slim",
        n_replicates = 10

    script:
        "scripts/make_training_data_slim.py"

