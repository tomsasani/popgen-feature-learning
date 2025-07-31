# python imports
from collections import defaultdict
import numpy as np
from bx.intervals.intersection import Interval, IntervalTree
import gzip
import csv
import tqdm
from cyvcf2 import VCF
import pandas as pd
from typing import Union
import cyvcf2
import util
import matplotlib.pyplot as plt
import seaborn as sns
import global_vars
import torch
from mutyper import Ancestor

def read_exclude(fh: str) -> IntervalTree:
    """
    Read in a BED file containing genomic regions from which we want
    to exclude potential variants. Riley et al. 2023 use a callability mask,
    but for my purposes I'll stick to a known file (Heng Li's LCR file for 
    hg19/hg38).

    Args:
        fh (str): Path to filename containing regions. Must be BED-formatted. Can be \
            uncompressed or gzipped.

    Returns:
        tree (Dict[IntervalTree]): Dictionary of IntervalTree objects, with one key per \
            chromosome. Each IntervalTree containing the BED regions from `fh`, on which we can \
            quickly perform binary searches later.
    """

    tree = defaultdict(IntervalTree)
    is_zipped = fh.endswith(".gz")

    print("BUILDING EXCLUDE TREE")

    with gzip.open(fh, "rt") if is_zipped else open(fh, "rt") as infh:
        csvf = csv.reader(infh, delimiter="\t")
        for l in tqdm.tqdm(csvf):
            if l[0].startswith("#") or l[0] == "chrom":
                continue
            chrom, start, end = l
            interval = Interval(int(start), int(end))
            tree[chrom].insert_interval(interval)

    return tree


class RealData(object):
    def __init__(
        self,
        vcf_fh: str,
        ped_fh: Union[str, None],
        bed_fh: Union[str, None],
        pop_code: Union[str, None] = None,
        superpop_code: Union[str, None] = None,
        filter_singletons: bool = False,
        convert_to_rgb: bool = False,
        sort: bool = False,
        convert_to_diploid: bool = False,
        seqlen: int = global_vars.L,
        n_snps: int = 64,
        rng: np.random.default_rng = None,
    ):

        callset = VCF(vcf_fh, gts012=True)

        self.exclude_tree = read_exclude(bed_fh) if bed_fh is not None else None
        
        if ped_fh is not None:
            ped = pd.read_csv(ped_fh, sep="\t").dropna()
            ped = ped[ped["Superpopulation code"].str.len() == 3]
            ped = ped[ped["Data collections"].str.contains("1000 Genomes phase 1 release")]
            if superpop_code is not None:
                ped = ped[ped["Superpopulation code"] == superpop_code]
            if pop_code is not None:
                ped = ped[ped["Population code"] == pop_code]

        samples2use = ped["Sample name"].to_list()

        callset.set_samples(samples2use)
        self.callset = callset
        self.n_samples = len(callset.samples)
        self.n_haps = len(callset.samples) * 2
        self.sample2idx = dict(zip(callset.samples, range(len(callset.samples),)))
        self.filter_singletons = filter_singletons
        self.convert_to_rgb = convert_to_rgb
        self.sort = sort
        self.convert_to_diploid = convert_to_diploid
        self.seqlen = seqlen
        self.n_snps = n_snps
        self.rng = rng
        chroms = callset.seqnames
        lengths = callset.seqlens
        self.chrom2len = dict(zip(chroms, lengths))

        AUTOSOMES = list(map(str, range(1, 23)))
        self.autosomes = AUTOSOMES

        self.mut_dict = {
            "A>C": 0,
            "A>G": 1,
            "A>T": 2,
            "C>A": 3,
            "C>G": 4,
            "C>T": 5,
        }
        self.nuc_dict = {
            "A": 0,
            "C": 1,
            "G": 2,
            "T": 3,
        }
        self.revcomp = {
            "A": "T",
            "T": "A",
            "C": "G",
            "G": "C",
        }

    def excess_overlap(self, chrom: str, start: int, end: int) -> bool:
        """
        Given an interval, figure out how much it overlaps an exclude region
        (if at all). If it overlaps the exclude region by more than 50% of its
        length, ditch the interval.

        Args:
            chrom (str): Chromosome of query region.
            start (int): Starting position of query region.
            end (int): Ending position of query region.
            thresh (float, optional): Fraction of base pairs of overlap between query region \
                and exclude regions, at which point we should ditch the interval. Defaults to 0.5.

        Returns:
            overlap_is_excessive (bool): Whether the query region overlaps the exclude \
                regions by >= 50%.
        """
        total_bp_overlap = 0

        if self.exclude_tree is None:
            return False

        else:
            overlaps = self.exclude_tree[chrom].find(start, end)
            for inter in overlaps:
                total_bp_overlap += inter.end - inter.start

            overlap_pct = total_bp_overlap / (end - start)
            if overlap_pct > 0.5:
                return True
            else:
                return False


    def filter_site(
        self,
        v: cyvcf2.Variant,
        keep_idxs: np.ndarray,
    ):
        # filter SNP
        if v.var_type != "snp":
            return (None, None, None)
        # filter multi-allelic
        if len(v.ALT) > 1:
            return (None, None, None)

        ref, alt = v.REF, v.ALT[0]
        

        haplotypes = np.array(v.genotypes)[:, :-1].reshape(1, -1)

        haplotypes = haplotypes[:, keep_idxs]

        # remove unknown
        if np.any(haplotypes == 3):
            return (None, None, None)

        # ignore singletons if desired
        if self.filter_singletons:
            if np.sum(haplotypes) == 1 or np.sum(haplotypes) == keep_idxs.shape[0] - 1:
                return (None, None, None)
            
        # ignore fixed sites
        if np.sum(haplotypes) == keep_idxs.shape[0]:
            return (None, None, None)
        if np.sum(haplotypes) == 0:
            return (None, None, None)

        return haplotypes, ref, alt

    def sample_real_region(
        self,
        chrom: str,
        start: int,
        end: Union[int, None] = None,
        keep: Union[int, None] = None,

    ) -> np.ndarray:
        """Sample a random "real" region of the genome from the provided VCF file,
        and use the variation data in that region to produce an ndarray of shape
        (n_haps, n_sites), where n_haps is the number of haplotypes in the VCF,
        n_sites is the NUM_SNPs we want in both a simulated or real region (defined
        in global_vars).

        Args:
            start_pos (int, optional): Starting position of the desired region. Defaults to None.

        Returns:
            np.ndarray: np.ndarray of shape (n_haps, n_sites).
        """
        # loop over all entries in the VCF by default, but stop when we
        # reach n_snps. if an end coordinate is provided, use that.
        if end is None:
            end = self.chrom2len[chrom]
        region = f"{chrom}:{start}-{end}"

        X = []
        positions = []
        refs, alts = [], []

        if keep is None:
            idxs = np.arange(self.n_haps)
        else:
            idxs = self.rng.choice(self.n_haps, size=keep)

        counted = 0
        for v in self.callset(region):
            if counted >= self.n_snps:
                break
            haplotypes, ref, alt = self.filter_site(v, keep_idxs=idxs)
            if haplotypes is None:
                continue
            # add entries to arrays
            X.append(haplotypes)
            positions.append(v.start)
            refs.append(ref)
            alts.append(alt)
            counted += 1

        X = np.squeeze(np.array(X))
        positions = np.array(positions)

        if counted == 1:
            X = np.expand_dims(X, axis=0)

        if counted == 0:
            return (None, None, None)
        
        # repolarize the haplotype array
        X, switch_idxs = util.major_minor(X.T)
        
        # at the idxs we need to repolarize, swap the ref
        # and alt alleles
        new_refs, new_alts = [], []
        for i, (r, a) in enumerate(zip(refs, alts)):
            if i in switch_idxs:
                new_refs.append(a)
                new_alts.append(r)
            else:
                new_refs.append(r)
                new_alts.append(a)
        
        new_mutations = []
        for r, a in zip(new_refs, new_alts):
            if r not in ("A", "C"):
                r, a = self.revcomp[r], self.revcomp[a]
            mut_i = self.mut_dict[">".join([r, a])]
            new_mutations.append(mut_i)
        new_mutations = np.array(new_mutations)

        # process the region
        region = util.process_region(
            X,
            positions,
            new_mutations,
            norm_len=global_vars.L,
            convert_to_rgb=self.convert_to_rgb,
            n_snps=self.n_snps,
            convert_to_diploid=self.convert_to_diploid,
        )

        if self.sort:
            region = util.sort_min_diff_numpy(region)

        # check for excess overlap with exclude file once we've finished cataloguing
        # the SNPs of interest.
        exclude = False
        if self.exclude_tree is not None:
            excess_overlap = self.excess_overlap(chrom, start, positions[-1])
            if excess_overlap:
                exclude = True

        if not exclude:
            return region, positions, new_mutations
        else:
            return (None, positions, new_mutations)


if __name__ == "__main__":

    VCF_FH = "/scratch/ucgd/lustre-core/common/data/1KG_VCF/1KG.chrALL.phase3_v5a.20130502.genotypes.vcf.gz"
    PED_FH = "data/igsr_samples.tsv"
    BED_FH = None  # "data/LCR-hs37d5.bed.gz"

    rng = np.random.default_rng(42)

    # read in VCF
    real_data = RealData(
        VCF_FH,
        PED_FH,
        BED_FH,
        pop_code="CHB",
        filter_singletons=False,
        convert_to_diploid=True,
        n_snps=32,
        convert_to_rgb=True,
        sort=True,
        rng=rng,
        
    )

    chrom2len = real_data.chrom2len
    AUTOSOMES = list(map(str, range(1, 23)))
    AUTOSOMES = [f"chr{c}" for c in AUTOSOMES]

    rng = np.random.default_rng(42)

    skipped = 0
    plotted = False
    for i in range(10):
        chrom = "21"  # np.random.choice(AUTOSOMES)
        start = rng.integers(1, chrom2len[chrom])
        region, positions, mutations = real_data.sample_real_region(chrom, start, end=None, keep=128)

        if region is None:
            skipped += 1
            continue

        minibatch = torch.from_numpy(region).unsqueeze(0)

        if i == 0:

            f, axarr = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(8, 8))
            for channel_i in range(3):
                sns.heatmap(
                    minibatch[0, channel_i, :, :],
                    ax=axarr[channel_i],
                )

            
            f.tight_layout()
            f.savefig("real.png", dpi=200)
