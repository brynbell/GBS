from models import GBSExperiment
import samplers
from verification import TotalNVerifier, TotalDistributionVerifier, TwoModeCorrelatorVerifier, XEntropyVerifier, NCumulantVerifier
import numpy as np

if __name__ == '__main__':
    m = 6
    experiment = GBSExperiment(modes=m, sources=m // 2, squeezing=1.1, transmission=0.5)
    state = experiment.calc_output_state()
    sampler = samplers.IPSSampler(experiment, pure_state_sampler=samplers.PureStateSamplerMinSqueeze(experiment))

    result = sampler.run(10000)
    print(f'Mean total photon number: {np.mean([sum(sample.det_pattern) for sample in result.sample_list])}')
    print(f'Mean sample time: {np.mean([sample.time for sample in result.sample_list])}')

    TotalNVerifier()(result.sample_list, state, log=False)
    TotalDistributionVerifier()(result.sample_list, state, [1, 2, 3, 4, 5, 6])
    n_mode_verifier = NCumulantVerifier(state, result.sample_list)
    n_mode_verifier(n=2)
    n_mode_verifier(n=3)
    n_mode_verifier(n=4)
