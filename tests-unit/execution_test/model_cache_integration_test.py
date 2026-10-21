"""Exercise real ComfyUI execution caches and safetensors with isolated CPU fixtures."""
import json
from pathlib import Path

import pytest
from comfy.cli_args import args
args.cpu = True

import nodes
import folder_paths
import comfy.model_cache as cache_runtime
from comfy.model_cache import store
from comfy.utils import load_torch_file
from execution import PromptExecutor, CacheType


class LoadFixture:
    calls = 0
    paths = []
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'name': ('STRING',)}}
    RETURN_TYPES = ('FIXTURE',)
    FUNCTION = 'load'
    def load(self, name):
        type(self).calls += 1
        path = folder_paths.get_full_path_or_raise('diffusion_models', name)
        type(self).paths.append(path)
        return (load_torch_file(path),)


class ConsumeFixture:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('FIXTURE',), 'nonce': ('INT',)}}
    RETURN_TYPES = ('FIXTURE',)
    FUNCTION = 'consume'
    OUTPUT_NODE = True
    def consume(self, model, nonce):
        return (model,)


class Independent:
    calls = 0
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}}
    RETURN_TYPES = ('INT',)
    FUNCTION = 'run'
    OUTPUT_NODE = True
    def run(self):
        type(self).calls += 1
        return (42,)


class Server:
    client_id = None
    def send_sync(self, *args, **kwargs):
        pass


@pytest.fixture
def managed(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'diffusion_models').mkdir(parents=True)
    path = archive / 'diffusion_models' / 'model.safetensors'
    header = json.dumps({'tensor': {'dtype': 'F32', 'shape': [1024], 'data_offsets': [0,4096]}}).encode()
    path.write_bytes(len(header).to_bytes(8,'little') + header + bytes(4096))
    config = dict(archive_root=str(archive), archive_uuid='fixture', cache_root=str(tmp_path/'cache'), max_bytes=10**6, reserve_bytes=0, idle_seconds=1800)
    monkeypatch.setattr(store,'check_archive',lambda config: Path(config['archive_root']))
    monkeypatch.setattr(folder_paths,'check_archive',store.check_archive)
    manager = store.ModelCache(config)
    monkeypatch.setattr(cache_runtime,'manager',manager)
    for name, node in [('CacheLoadFixture', LoadFixture), ('CacheConsumeFixture',ConsumeFixture), ('CacheIndependent',Independent)]:
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS,name,node)
    LoadFixture.calls = Independent.calls = 0
    LoadFixture.paths = []
    yield manager
    manager.close()


def prompt(nonce=1):
    return {
        '1': {'class_type':'CacheLoadFixture','inputs':{'name':'archive/model.safetensors'}},
        '2': {'class_type':'CacheConsumeFixture','inputs':{'model':['1',0],'nonce':nonce}},
        '3': {'class_type':'CacheIndependent','inputs':{}},
    }


@pytest.mark.parametrize('cache_type',[CacheType.CLASSIC,CacheType.LRU,CacheType.RAM_PRESSURE,CacheType.NONE])
def test_native_load_reuse_activity_selective_release(managed, cache_type):
    executor = PromptExecutor(Server(),cache_type=cache_type,cache_args={'ram':0,'ram_inactive':0,'lru':20})
    assert 'archive/model.safetensors' in folder_paths.get_filename_list('diffusion_models')
    source = folder_paths.get_full_path('diffusion_models','archive/model.safetensors')
    assert str(managed.archive) in source
    assert not list(managed.files.iterdir()), 'Discovery must not copy weights'
    executor.execute(prompt(), 'first',execute_outputs=['2','3'])
    assert executor.success
    row = managed.db.execute('SELECT * FROM entries').fetchone()
    key = row['key']
    assert all(str(managed.files) in p for p in LoadFixture.paths)
    assert row['active'] == 0
    managed.monotonic_used[key] -= 100
    executor.execute(prompt(2),'second',execute_outputs=['2','3'])
    assert executor.success
    assert LoadFixture.calls == (2 if cache_type == CacheType.NONE else 1)
    assert managed.expired() == set()
    managed.monotonic_used[key] -= 1801
    executor.execute(prompt(2),'fully-cached',execute_outputs=['2','3'])
    assert managed.expired() == set(), 'Fully cached output still counts as use'
    assert managed.db.execute('SELECT active FROM entries').fetchone()[0] == 0
    managed.monotonic_used[key] -= 1801
    executor.maintain_model_cache()
    assert not managed.path(key).exists(), 'Eviction must release actual safetensors mappings'
    executor.execute(prompt(3),'third',execute_outputs=['2','3'])
    assert executor.success
    assert LoadFixture.calls == (4 if cache_type == CacheType.NONE else 2)
    assert Independent.calls == (4 if cache_type == CacheType.NONE else 1)


def test_source_replacement_invalidates_cached_outputs(managed):
    executor = PromptExecutor(Server(),cache_type=CacheType.CLASSIC,cache_args={'ram':0,'ram_inactive':0})
    executor.execute(prompt(),'first',execute_outputs=['2'])
    assert executor.success
    source = managed.archive / 'diffusion_models/model.safetensors'
    original = source.read_bytes()
    source.write_bytes(original[:-1]+b'X')
    executor.execute(prompt(2),'changed',execute_outputs=['2'])
    assert executor.success
    assert LoadFixture.calls == 2
    assert len(set(LoadFixture.paths)) == 2


def test_async_loader_resources_survive_pending_execution(managed, monkeypatch):
    import asyncio
    class AsyncLoad(LoadFixture):
        async def load(self, name):
            await asyncio.sleep(0)
            return super().load(name)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS,'CacheLoadFixture',AsyncLoad)
    executor = PromptExecutor(Server(),cache_type=CacheType.LRU,cache_args={'ram':0,'ram_inactive':0,'lru':20})
    executor.execute(prompt(),'async',execute_outputs=['2'])
    assert executor.success
    key = managed.db.execute('SELECT key FROM entries').fetchone()[0]
    managed.monotonic_used[key] -= 1801
    executor.maintain_model_cache()
    assert not managed.path(key).exists()


def test_background_weight_loading_rejected(managed):
    with pytest.raises(store.CacheError,match='EXECUTION_REQUIRED'):
        load_torch_file(managed.archive / 'diffusion_models/model.safetensors')
