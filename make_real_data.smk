CHROMS = list(map(str, range(1, 23)))
CHROMS = ["2"]

POPS = ["CEU"]
CHROMS = ["2"]
N_SNPS = [32, 64]

rule all:
    input:
        "data/real/CEU/2/64/regions.txt",


rule create_real_chrom_images:
    input:
        vcf = "/scratch/ucgd/lustre-core/common/data/1KG_VCF/1KG.chrALL.phase3_v5a.20130502.genotypes.vcf.gz",
        ped = "data/igsr_samples.tsv",
        exclude = "data/LCR-hs37d5.bed.gz",
    output: fh = "data/real/{population}/{chrom}/{n_snps}/regions.txt"
    params:
        pref = "data/real",
    script:
        "scripts/make_training_data_real.py"


rule create_fake_chrom_images:
    input:
    output: fh = "data/fake/{population}/{chrom}/{n_snps}/regions.txt"
    params:
        pref = "data/fake",
    script:
        "scripts/make_training_data_fake.py"