"""Continuous flow grids preserve exact step counts and model-owned shifts."""
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy import model_sampling, samplers


@pytest.mark.parametrize('steps', [1, 7, 100, 333, 500])
@pytest.mark.parametrize('shift', [.5, 1., 5.])
def test_grid_matches_moss_reference_scheduler(steps, shift):
    sampling = model_sampling.ModelSamplingDiscreteFlow()
    sampling.set_parameters(shift=shift, multiplier=1000)
    # Pinned DiffSynth FlowMatchScheduler: extra_one_step=True, sigma_min=0,
    # sigma_max=1; its final Euler update jumps to zero.
    reference = torch.linspace(1, 0, steps + 1)[:-1]
    reference = shift * reference / (1 + (shift - 1) * reference)
    reference = torch.cat((reference, torch.zeros(1)))
    actual = samplers.calculate_sigmas(sampling, 'linear_flow', steps)
    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-7)
    assert len(actual) == steps + 1 and actual[-1] == 0 and actual[0] == 1
    assert (actual[:-1] >= actual[1:]).all()


def test_requires_flow_sampling_and_leaves_existing_scheduler_unchanged():
    with pytest.raises(ValueError, match='discrete flow'):
        samplers.calculate_sigmas(SimpleNamespace(), 'linear_flow', 7)
    sampling = model_sampling.ModelSamplingDiscreteFlow()
    expected = samplers.simple_scheduler(sampling, 7)
    torch.testing.assert_close(samplers.calculate_sigmas(sampling, 'simple', 7), expected, rtol=0, atol=0)
