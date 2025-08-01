CHROMS = list(map(str, range(1, 23)))
CHROMS = [f"chr{c}" for c in CHROMS]

POPS = ["CEU"]
N_SNPS = [32, 64]

rule all:
    input:
        "data/training/CEU/64/neutral/0/regions.txt",
        "data/ood/CEU/64/25/regions.txt",
        "data/training/CEU/64/selection/25/regions.txt",

# images to "contaminate" real data with
rule create_selection_training_images:
    input:
    output: fh = "data/training/{population}/{n_snps}/selection/{raw_coef}/regions.txt"
    params:
        seed = 42,
        chrom = "chr2",
        n_haps = 100,
        pref = lambda wildcards: f"data/training/{wildcards.population}/{wildcards.n_snps}/selection/{wildcards.raw_coef}",
        n_replicates = 100

    script:
        "scripts/make_training_data_slim.py"

# images for ssdk
rule create_selection_ood_images:
    input:
    output: fh = "data/ood/{population}/{n_snps}/{raw_coef}/regions.txt"
    params:
        seed = 1234,
        chrom = "chr2",
        n_haps = 100,
        pref = lambda wildcards: f"data/ood/{wildcards.population}/{wildcards.n_snps}/{wildcards.raw_coef}",
        n_replicates = 10

    script:
        "scripts/make_training_data_slim.py"

# baseline real data
rule create_neutral_training_images:
    input:
    output: fh = "data/training/{population}/{n_snps}/neutral/0/regions.txt"
    params:
        seed = 1234,
        chrom = "chr2",
        n_haps = 100,
        pref = lambda wildcards: f"data/training/{wildcards.population}/{wildcards.n_snps}/neutral/0",
        n_replicates = 5_000

    script:
        "scripts/make_training_data_fake.py"

