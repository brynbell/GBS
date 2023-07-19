from models import BorealisExperiment
import samplers
from verification import TotalNVerifier, TotalDistributionVerifier, NCumulantVerifier
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    exp = BorealisExperiment(r'C:\Users\bryn_\PycharmProjects\xanadu-qca-data\qca-data\fig2')
    state = exp.calc_output_state()
    sampler = samplers.MSourceSampler(exp, cutoff=4)

    result = sampler.run(100000)
    print(f'Mean total photon number: {np.mean([sum(sample.det_pattern) for sample in result.sample_list])}')
    print(f'Mean sample time: {np.mean([sample.time for sample in result.sample_list])}')

    TotalNVerifier()(result.sample_list, state, log=False)
    TotalDistributionVerifier()(result.sample_list, state, [1, 2, 3, 4])
    n_mode_verifier = NCumulantVerifier(state, result.sample_list)
    n_mode_verifier(n=2)
    n_mode_verifier(n=3)
    n_mode_verifier(n=4)
