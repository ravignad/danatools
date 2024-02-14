import pytest
import numpy as np
import danatools
from scipy.stats import norm

def test_density_histogram():
    rng = np.random.default_rng(seed=6870)
    a = norm.rvs(size=1000, random_state=rng)
    density, xbin = danatools.histogram(a, bins=20, range=(-1, 1), density=True)
    pdf = norm.pdf(xbin)
    rms = danatools.array_rms(density - pdf)
    assert rms < 0.1