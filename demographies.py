import msprime
import numpy as np
import demesdraw
import matplotlib.pyplot as plt

import math

import global_vars


def simulate_exp(
    params,
    sample_sizes,
    recombination_map,
    rng: np.random.default_rng,
    seqlen: int = global_vars.L,
    plot: bool = False,
    plot_title: str = "demography.png",
):
    # get random seed for msprime simulations
    seed = rng.integers(1, 2**32)

    # get demographic parameters
    N1, N2 = params.N1.value, params.N2.value
    T1, T2 = params.T1.value, params.T2.value
    growth = params.growth.value

    N0 = N2 / math.exp(-growth * T2)

    demography = msprime.Demography()

    # at present moment, create population A with the size it should be
    # following its period of exponential growth
    demography.add_population(
        name="A",
        initial_size=N0,
        growth_rate=growth,
    )
    # T2 generations in the past, change the population size to be N2
    demography.add_population_parameters_change(
        population="A",
        time=T2,
        initial_size=N2,
        growth_rate=0,
    )

    # T1 generations in the past, change the population size to be N1
    demography.add_population_parameters_change(
        population="A",
        time=T1,
        initial_size=N1,
        growth_rate=0,
    )

    if plot:
        graph = msprime.Demography.to_demes(demography)
        f, ax = plt.subplots()
        demesdraw.tubes(graph, ax=ax, seed=1)
        f.savefig(plot_title, dpi=200)

    # sample sample_sizes diploids from the diploid population
    ts = msprime.sim_ancestry(
        samples=sum(sample_sizes),
        demography=demography,
        sequence_length=seqlen,
        recombination_rate=recombination_map,
        discrete_genome=False,
        random_seed=seed,
        ploidy=2,
    )

    mts = msprime.sim_mutations(
        ts,
        rate=params.mu.value,
        model=msprime.BinaryMutationModel(state_independent=False),
        random_seed=seed,
        discrete_genome=False,
    )

    return mts


if __name__ == "__main__":
    import params
    params = params.ParamSet()
    rng = np.random.default_rng(42)

    L = 25_000

    rate_map = msprime.RateMap(
        position=[0, (L - 2_000) // 2, (L + 2_000) // 2, L],
        rate=[1e-8, 1e-8 * 10, 1e-8],
    )
    for _ in range(10):
        ts = simulate_exp(params, [50], rate_map, rng, plot=True, seqlen=L)
        print (ts.genotype_matrix().shape)
