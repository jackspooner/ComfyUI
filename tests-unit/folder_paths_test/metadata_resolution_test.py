from types import SimpleNamespace

import comfy.model_cache
import folder_paths


def test_metadata_resolution_does_not_stage_archive_during_node_execution(monkeypatch, tmp_path):
    original = tmp_path / 'archive' / 'model.safetensors'
    staged = []
    selected = []
    monkeypatch.setattr(comfy.model_cache, 'manager', SimpleNamespace(config={}))
    monkeypatch.setattr(folder_paths, 'source_path',
        lambda config, category, name: selected.append((category, name)) or original)
    monkeypatch.setattr(comfy.model_cache, 'resolve',
        lambda path: staged.append(path) or str(tmp_path / 'cached.safetensors'))
    with comfy.model_cache.node_scope('native-package-lookup'):
        assert folder_paths.get_full_path('clip', 'archive/model.safetensors', resolve_cache=False) == str(original)
        assert staged == []
        assert selected == [('text_encoders', 'model.safetensors')]
        assert folder_paths.get_full_path('clip', 'archive/model.safetensors') == str(tmp_path / 'cached.safetensors')
        assert staged == [original]


def test_metadata_resolution_retains_local_root_precedence(monkeypatch, tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / 'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'text_encoders',
        ([str(first), str(second)], {'.safetensors'}))
    assert folder_paths.get_full_path('clip', 'model.safetensors', resolve_cache=False) == str(first / 'model.safetensors')
