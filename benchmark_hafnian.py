from models import GBSExperiment
import samplers
from thewalrus.loop_hafnian_batch import loop_hafnian_batch
from thewalrus._hafnian import loop_hafnian
from verification import TotalNVerifier, TotalDistributionVerifier, NCumulantVerifier
import numpy as np
import matplotlib.pyplot as plt
import datetime

folder = r"C:\Users\bryn_\PycharmProjects\xanadu-qca-data\qca-data\\"

if __name__ == "__main__":
    m = 90
    n = 34
    cutoff = 6
    exp = GBSExperiment(m, m, 1.0, 0.3)
    state = exp.calc_output_state()
    # sampler = samplers.IPSSampler(exp, pure_state_sampler=samplers.PureStateSamplerMinSqueeze(exp))
    # sample = sampler.get_sample()
    # det_pattern = sample.det_pattern
    det_pattern = np.random.multinomial(n, [1/m]*m)
    det_pattern[-1] = cutoff
    loop_hafnian(state.get_B()[:3, :3], state.get_gamma()[:3], det_pattern[:3], glynn=False)
    print(np.sum(det_pattern))
    now = datetime.datetime.now()
    loop_hafnian(state.get_B(), state.get_gamma()[:m], det_pattern, glynn=False)
    print((datetime.datetime.now() - now).total_seconds())
    now = datetime.datetime.now()
    loop_hafnian_batch(state.get_B(), state.get_gamma()[:m], det_pattern[:-1], 6, glynn=False)
    print((datetime.datetime.now()-now).total_seconds())