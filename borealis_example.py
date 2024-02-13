from models import BorealisExperiment
import samplers
from verification import TotalNVerifier, TotalDistributionVerifier, NCumulantVerifier
import numpy as np
import matplotlib.pyplot as plt

folder = r"C:\Users\bryn_\PycharmProjects\xanadu-qca-data\qca-data\\"

if __name__ == "__main__":
    # exp = BorealisExperiment(folder + 'fig2')
    # exp = BorealisExperiment(folder + 'fig3b')
    # exp = BorealisExperiment(folder + 'fig3a')
    exp = BorealisExperiment(folder + 'fig4')
    state = exp.calc_output_state()
    sampler = samplers.FasterExactSampler(exp, cutoff=6)
    print(state.modes)

    result = sampler.run(1)
    # result = samplers.load_result('2023_07_24_10-58-06_FasterExactSampler.p')
    print(result.state.modes)
    print(f'Mean total photon number: {np.mean([sum(sample.det_pattern) for sample in result.sample_list])}')
    print(f'Mean sample time: {np.mean([sample.time for sample in result.sample_list])}')

    TotalNVerifier()(result.sample_list, state, log=True)
    # TotalDistributionVerifier()(result.sample_list, state, [1, 2, 3, 4])
    # n_mode_verifier = NCumulantVerifier(state, result.sample_list)
    # n_mode_verifier(n=2)
    # n_mode_verifier(n=3)
    # n_mode_verifier(n=4)
