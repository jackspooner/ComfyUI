import json
import mmap
from pathlib import Path
import subprocess
import sys
import time

import pytest

from comfy.model_cache import store


@pytest.fixture
def cache(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'diffusion_models').mkdir(parents=True)
    root = tmp_path / 'cache'
    config = dict(archive_root=str(archive), archive_uuid='test-device', cache_root=str(root), max_bytes=1024 * 1024, reserve_bytes=0, idle_seconds=1)
    monkeypatch.setattr(store, 'check_archive', lambda config: Path(config['archive_root']))
    manager = store.ModelCache(config)
    yield manager
    manager.close()


def model(cache, name='sample.safetensors'):
    path = cache.archive / 'diffusion_models' / name
    header = json.dumps({'value': {'dtype': 'F32', 'shape': [1024], 'data_offsets': [0,4096]}}).encode()
    path.write_bytes(len(header).to_bytes(8,'little') + header + bytes(4096))
    return path


def test_copy_reuse_and_expiry(cache):
    original = model(cache)
    target, key = cache.stage(original)
    assert Path(target).read_bytes() == original.read_bytes()
    inode = Path(target).stat().st_ino
    assert cache.stage(original) == (target, key)
    assert Path(target).stat().st_ino == inode
    cache.touch({key}, finished=True)
    cache.monotonic_used[key] -= 2
    assert cache.expired() == {key}
    cache.evict({key})
    assert not Path(target).exists()
    assert original.exists()


def test_mapping_blocks_eviction_until_release(cache):
    target, key = cache.stage(model(cache))
    with open(target,'rb') as f:
        mapped = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    cache.evict({key})
    assert Path(target).exists()
    assert cache.db.execute('SELECT reason FROM entries').fetchone()[0]
    mapped.close()
    cache.evict({key})
    assert not Path(target).exists()


def test_restart_keeps_complete_and_discards_interrupted_copy(cache):
    target, key = cache.stage(model(cache))
    cache.touch({key}, finished=True)
    interrupted = 'a' * 64
    with cache.db:
        cache.db.execute("INSERT INTO entries(key,source,fingerprint,size,state,last_used) VALUES(?,?,?,?,'copying',?)",(interrupted,'unused','[]',50,time.time()))
    cache.path(interrupted).with_suffix('.partial').write_bytes(b'incomplete')
    cache.close()
    replacement = store.ModelCache(cache.config)
    try:
        assert replacement.stage(model(replacement))[0]  # source replacement creates a new generation
        assert Path(target).exists()
        assert not replacement.path(interrupted).with_suffix('.partial').exists()
        assert replacement.db.execute('SELECT key FROM entries WHERE key=?',(interrupted,)).fetchone() is None
    finally:
        replacement.close()


def test_source_change_creates_new_generation(cache):
    original = model(cache)
    target, key = cache.stage(original)
    data = original.read_bytes()
    original.write_bytes(data[:-1] + b'X')
    assert key in cache.changed()
    target2, key2 = cache.stage(original)
    assert key != key2
    assert Path(target).exists()
    assert Path(target2).read_bytes() == original.read_bytes()


def test_copy_failure_leaves_no_ready_file(cache):
    original = model(cache)
    def fail(done,total):
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):
        cache.stage(original, fail)
    assert list(cache.files.iterdir()) == []
    assert cache.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 0
    assert original.exists()


def test_capacity_and_pins(cache):
    original = model(cache)
    cache.config['max_bytes'] = 1
    with pytest.raises(store.CacheError,match='CACHE_FULL'):
        cache.stage(original)
    cache.config['max_bytes'] = 1024 * 1024
    target, key = cache.stage(original)
    with cache.db:
        cache.db.execute("INSERT INTO commands(operation,argument) VALUES('pin',?)",(key,))
    cache.commands()
    cache.monotonic_used[key] -= 2
    assert not cache.expired()
    cache.evict({key})
    assert Path(target).exists()


def test_archive_path_rejections(cache):
    original = model(cache)
    (original.parent / 'link.safetensors').symlink_to(original)
    for name in ('../sample.safetensors','/etc/passwd','link.safetensors'):
        with pytest.raises(store.CacheError):
            store.source_path(cache.config, 'diffusion_models', name)


def test_second_owner_refused(cache):
    with pytest.raises(store.CacheError,match='CACHE_BUSY'):
        store.ModelCache(cache.config)


def test_checksum_revalidated_after_restart(cache):
    original = model(cache)
    target,key = cache.stage(original)
    cache.close()
    data = Path(target).read_bytes()
    Path(target).chmod(0o600)
    Path(target).write_bytes(data[:-1] + b'X')
    replacement = store.ModelCache(cache.config)
    try:
        with pytest.raises(store.CacheError,match='CACHE_CORRUPT'):
            replacement.stage(original)
    finally:
        replacement.close()


def test_source_changed_mid_copy_is_rejected(cache):
    original = model(cache)
    def alter(done,total):
        original.touch()
    with pytest.raises(store.CacheError,match='SOURCE_CHANGED'):
        cache.stage(original, alter)
    assert not list(cache.files.iterdir())


def test_concurrent_requests_share_copy(cache):
    from concurrent.futures import ThreadPoolExecutor
    original = model(cache)
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda _: cache.stage(original), range(8)))
    assert len(set(results)) == 1
    assert cache.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 1


def test_invalid_weights_never_published(cache):
    original = model(cache)
    original.write_bytes(b'<html>not a model</html>')
    with pytest.raises(store.CacheError,match='MODEL_CORRUPT'):
        cache.stage(original)
    assert not list(cache.files.iterdir())


def test_wrong_mount_and_invalid_config(tmp_path):
    import yaml
    archive = tmp_path/'archive'
    archive.mkdir()
    config = dict(archive_root=str(archive), archive_uuid='not-this-filesystem', cache_root=str(tmp_path/'cache'))
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    parsed = store.load_config(path)
    assert parsed['idle_seconds'] == 1800
    with pytest.raises(store.CacheError,match='ARCHIVE_UNAVAILABLE'):
        store.check_archive(parsed)
    config['cache_root'] = str(archive/'nested')
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(store.CacheError,match='CONFIG_INVALID'):
        store.load_config(path)


def test_clock_rollback_and_crashed_activity(cache):
    original = model(cache)
    target,key = cache.stage(original)
    with cache.db:
        cache.db.execute('UPDATE entries SET last_used=?',(time.time()+99999,))
    cache.close()
    replacement = store.ModelCache(cache.config)
    try:
        row = replacement.db.execute('SELECT active,last_used FROM entries').fetchone()
        assert row['active'] == 0
        assert abs(row['last_used']-time.time()) < 5
        assert not replacement.expired()
    finally:
        replacement.close()


def test_cli_queues_commands_without_claiming_completion(cache, tmp_path):
    import yaml
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump(cache.config))
    model(cache)
    result = subprocess.run([sys.executable,'-m','comfy.model_cache','--config',str(config),'prefetch','diffusion_models/sample.safetensors'],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['state'] == 'pending'
    assert not list(cache.files.iterdir())
    cache.commands()
    assert cache.db.execute('SELECT state FROM commands').fetchone()[0] == 'done'
    assert cache.db.execute('SELECT active FROM entries').fetchone()[0] == 0


def test_reserve_exhaustion_during_copy(cache, monkeypatch):
    original = model(cache)
    real = store.shutil.disk_usage
    calls = 0
    def space(path):
        nonlocal calls
        calls += 1
        if calls > 1:
            return type('Space',(),{'free':0})()
        return real(path)
    monkeypatch.setattr(store.shutil,'disk_usage',space)
    with pytest.raises(store.CacheError,match='CACHE_FULL'):
        cache.stage(original)
    assert not list(cache.files.iterdir())


def test_process_death_mid_copy_recovers(cache):
    original = model(cache)
    config = cache.config.copy()
    cache.close()
    code = '''
import json, os, sys
from pathlib import Path
from comfy.model_cache import store
store.check_archive = lambda config: Path(config['archive_root'])
cache = store.ModelCache(json.loads(sys.argv[1]))
cache.stage(sys.argv[2], lambda done,total: os._exit(23))
'''
    result = subprocess.run([sys.executable,'-c',code,json.dumps(config),str(original)],check=False)
    assert result.returncode == 23
    replacement = store.ModelCache(config)
    try:
        assert not list(replacement.files.iterdir())
        assert replacement.db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] == 0
        target,_ = replacement.stage(original)
        assert Path(target).read_bytes() == original.read_bytes()
    finally:
        replacement.close()


def test_open_descriptor_blocks_eviction(cache):
    target,key = cache.stage(model(cache))
    with open(target,'rb'):
        cache.evict({key})
        assert Path(target).exists()
    cache.evict({key})
    assert not Path(target).exists()


def test_corruption_during_same_process_is_detected(cache):
    source = model(cache)
    target,key = cache.stage(source)
    path = Path(target)
    path.chmod(0o600)
    data = path.read_bytes()
    path.write_bytes(data[:-1] + b'X')
    with pytest.raises(store.CacheError,match='CACHE_CORRUPT'):
        cache.stage(source)
