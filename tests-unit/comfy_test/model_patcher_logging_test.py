"""Exercise dynamic preparation logging without allocating CUDA memory."""
import logging
from types import SimpleNamespace
import uuid

import pytest
import torch

from comfy import model_patcher


@pytest.fixture
def make_patcher(monkeypatch):
    patchers = []
    monkeypatch.setattr(model_patcher.comfy_aimdo.host_buffer, 'HostBuffer', lambda *args: SimpleNamespace(size=0))

    def make(device='cuda:0'):
        patcher = model_patcher.ModelPatcherDynamic(torch.nn.Module(), torch.device(device), torch.device('cpu'))
        patcher.model.dynamic_pins[patcher.load_device]['hostbufs_initialized'] = True
        monkeypatch.setattr(patcher, '_vbar_get', lambda create=False: None)
        patchers.append(patcher)
        return patcher

    yield make
    for patcher in patchers:
        patcher.detach()
        patcher.model.dynamic_patchers.discard(id(patcher))


def prepare_levels(caplog):
    return [record.levelno for record in caplog.records if 'prepared for dynamic VRAM loading' in record.message]


@pytest.mark.parametrize('progress_active', [False, True])
def test_repeated_preparation_keeps_admission_callbacks_and_demotes_logs(make_patcher, monkeypatch, caplog, progress_active):
    monkeypatch.setattr(model_patcher.tqdm.tqdm, '_instances', {object()} if progress_active else set())
    patcher = make_patcher()
    callbacks = []
    patcher.add_callback(model_patcher.CallbacksMP.ON_LOAD, lambda *args: callbacks.append(args))

    with caplog.at_level(logging.DEBUG):
        for _ in range(4):
            patcher.load(patcher.load_device)

    assert prepare_levels(caplog) == [logging.INFO, logging.DEBUG, logging.DEBUG, logging.DEBUG]
    assert len(callbacks) == 4


def test_preparation_suppression_is_private_to_each_patcher(make_patcher, caplog):
    first, second = make_patcher(), make_patcher()
    second.patches_uuid = first.patches_uuid
    with caplog.at_level(logging.DEBUG):
        first.load(first.load_device)
        second.load(second.load_device)
        first.load(first.load_device)
        second.load(second.load_device)
    assert prepare_levels(caplog) == [logging.INFO, logging.INFO, logging.DEBUG, logging.DEBUG]


def test_changed_patches_are_visible(make_patcher, caplog):
    patcher = make_patcher()
    with caplog.at_level(logging.DEBUG):
        patcher.load(patcher.load_device)
        patcher.patches_uuid = uuid.uuid4()
        patcher.load(patcher.load_device)
        patcher.load(patcher.load_device)
    assert prepare_levels(caplog) == [logging.INFO, logging.INFO, logging.DEBUG]


def test_changed_device_is_visible(make_patcher, caplog):
    patcher = make_patcher()
    with caplog.at_level(logging.DEBUG):
        patcher.load(patcher.load_device)
        patcher.load_device = torch.device('cuda:1')
        patcher.register_load_device(patcher.load_device)
        patcher.model.dynamic_pins[patcher.load_device]['hostbufs_initialized'] = True
        patcher.load(patcher.load_device)
        patcher.load(patcher.load_device)
    assert prepare_levels(caplog) == [logging.INFO, logging.INFO, logging.DEBUG]


@pytest.mark.parametrize('unpatch_weights', [False, True])
def test_only_full_unload_resets_preparation_logging(make_patcher, caplog, unpatch_weights):
    patcher = make_patcher()
    with caplog.at_level(logging.DEBUG):
        patcher.load(patcher.load_device)
        patcher.unpatch_model(unpatch_weights=unpatch_weights)
        patcher.load(patcher.load_device)
        patcher.load(patcher.load_device)
    assert prepare_levels(caplog) == [logging.INFO, logging.INFO if unpatch_weights else logging.DEBUG, logging.DEBUG]
