import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from comfy.cli_args import args
args.cpu = True
import comfy.audio_codec as ac
from comfy_extras.nodes_audio_codec import AudioCodecLoader, AudioCodecEncode, AudioCodecDecode, comfy_entrypoint


def payload(codes=None, lengths=(2, 1), **kwargs):
    if codes is None:
        codes = torch.zeros(2, 16, 3, dtype=torch.int64)
    return ac.AudioCodes(ac.MOSS_V1, (ac.AudioCodeSegment(codes, lengths, **kwargs),))


@pytest.mark.parametrize('change', [dict(version=2), dict(codec=replace(ac.MOSS_V1, revision='other')), dict(codec=replace(ac.MOSS_V1, codebook_sizes=(512,) * 32))])
def test_identity_is_required(change):
    with pytest.raises(ValueError, match='identity'):
        replace(payload(), **change).validate(ac.MOSS_V1)


@pytest.mark.parametrize('codes,lengths', [
    (torch.zeros(2, 16, 3), (2, 1)),
    (torch.zeros(2, 3, dtype=torch.int64), (2, 1)),
    (torch.zeros(2, 33, 3, dtype=torch.int64), (2, 1)),
    (torch.zeros(2, 0, 3, dtype=torch.int64), (2, 1)),
    (torch.zeros(2, 16, 3, dtype=torch.int64), (4, 1)),
    (torch.zeros(2, 16, 3, dtype=torch.int64), (1,)),
    (torch.zeros(2, 16, 3, dtype=torch.int64), (True, 1)),
])
def test_invalid_payload(codes, lengths):
    with pytest.raises(ValueError, match='AUDIO_CODES'):
        payload(codes, lengths).validate(ac.MOSS_V1)


def test_padding_is_outside_valid_frames_only():
    codes = torch.full((2, 16, 3), 1024, dtype=torch.int32)
    codes[0, :, :2] = 1023
    codes[1, :, :1] = 0
    data = payload(codes)
    data.validate(ac.MOSS_V1)
    codes[0, 0, 1] = -1
    with pytest.raises(ValueError, match='protocol padding'):
        data.validate(ac.MOSS_V1)


class CodecMath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def decode(self, codes):
        self.inputs.append(codes.clone())
        return torch.full((1, 1, codes.shape[-1] * 1920), 2.0)

    def encode(self, waveform, lengths, count):
        self.inputs.append(waveform.clone())
        frames = (waveform.shape[-1] + 1919) // 1920
        return torch.zeros(1, count, frames, dtype=torch.int64), [n // 1920 for n in lengths]


@pytest.fixture
def codec(monkeypatch):
    codec = ac.AudioCodec.__new__(ac.AudioCodec)
    codec.identity = ac.MOSS_V1
    codec.device = torch.device('cpu')
    codec.model = CodecMath()
    codec.patcher = object()
    loader = Mock()
    monkeypatch.setattr(ac.model_management, 'load_models_gpu', loader)
    return codec, loader


def test_decode_lengths_raw_pcm_and_memory_reservation(codec):
    codec, loader = codec
    data = payload()
    output = codec.decode(data)
    assert [a['waveform'].shape for a in output] == [(1, 1, 3840), (1, 1, 1920)]
    assert all(a['sample_rate'] == 24000 and a['waveform'].device.type == 'cpu' and torch.all(a['waveform'] == 2) for a in output)
    assert [x.shape[-1] for x in codec.model.inputs] == [2, 1]
    assert all(x.dtype == torch.int64 for x in codec.model.inputs)
    loader.assert_called_once_with([codec.patcher], memory_required=codec.memory_required(2))


def test_empty_decode_does_not_load_model(codec):
    codec, loader = codec
    output = codec.decode(payload(lengths=(0, 0)))
    assert [a['waveform'].numel() for a in output] == [0, 0]
    assert not codec.model.inputs
    loader.assert_not_called()


def test_encode_mono_resample_and_partial_frames(codec):
    codec, loader = codec
    audio = {'waveform': torch.ones(2, 2, 3874), 'sample_rate': 48000}
    data = codec.encode(audio, 16)
    data.validate(ac.MOSS_V1)
    assert len(data.segments) == 2
    for segment in data.segments:
        assert segment.codes.shape == (1, 16, 2)
        assert segment.lengths == (1,)
        assert segment.source_sample_lengths == (1937,)
        assert segment.codes.device.type == 'cpu'
    assert codec.model.inputs[0].shape == (1, 1, 1937)
    torch.testing.assert_close(codec.model.inputs[0][..., 10:-10], torch.ones(1, 1, 1917), atol=0.001, rtol=0)
    loader.assert_called_once_with([codec.patcher], memory_required=codec.memory_required(2, encode=True))


def test_empty_encode(codec):
    codec, loader = codec
    data = codec.encode({'waveform': torch.empty(1, 1, 0), 'sample_rate': 48000})
    assert data.segments[0].codes.shape == (1, 32, 0)
    assert data.segments[0].lengths == (0,)
    loader.assert_not_called()


def test_cancel_does_not_return_partial_batch(codec, monkeypatch):
    codec, _ = codec
    interrupted = Mock(side_effect=[None, ac.model_management.InterruptProcessingException()])
    monkeypatch.setattr(ac.model_management, 'throw_exception_if_processing_interrupted', interrupted)
    with pytest.raises(ac.model_management.InterruptProcessingException):
        codec.decode(payload())
    assert len(codec.model.inputs) == 1


@pytest.mark.parametrize('metadata', [None, {}, {'architecture': 'moss_audio_tokenizer', 'model_revision': 'wrong', 'conversion_version': '1'}])
def test_loader_rejects_unknown_artifact_before_allocation(metadata):
    with pytest.raises(ValueError, match='convert_moss_codec'):
        ac.AudioCodec({}, metadata)


def test_loader_rejects_incomplete_checkpoint():
    with pytest.raises(ValueError, match='complete folded FP32'):
        ac.AudioCodec({}, {'architecture': ac.MOSS_V1.family, 'model_revision': ac.MOSS_V1.revision, 'conversion_version': '1'})


def test_reconstruction_respects_dynamic_override(monkeypatch):
    state = {'weight': torch.ones(1)}
    metadata = {'architecture': ac.MOSS_V1.family}
    read = Mock(return_value=(state, metadata))
    owner = Mock(return_value=Mock(patcher=Mock()))
    monkeypatch.setattr(ac.comfy.utils, 'load_torch_file', read)
    monkeypatch.setattr(ac, 'AudioCodec', owner)
    codec = ac.load_audio_codec('/models/codec.safetensors', device=torch.device('cpu'))
    read.assert_called_once_with('/models/codec.safetensors', safe_load=True, return_metadata=True)
    factory, arguments = codec.patcher.cached_patcher_init
    patcher = factory(*arguments, disable_dynamic=True)
    assert patcher is codec.patcher
    assert owner.call_args.kwargs == {'device': torch.device('cpu'), 'disable_dynamic': True}


def test_node_contracts_and_contained_loading(tmp_path, monkeypatch, codec):
    import folder_paths
    path = tmp_path / 'codec.safetensors'
    path.touch()
    monkeypatch.setitem(folder_paths.folder_names_and_paths, 'vae', ([str(tmp_path)], {'.safetensors'}))
    load = Mock(return_value=codec[0])
    monkeypatch.setattr(ac, 'load_audio_codec', load)
    assert AudioCodecLoader.execute('codec.safetensors').result[0] is codec[0]
    load.assert_called_once_with(str(path))
    with pytest.raises(FileNotFoundError):
        AudioCodecLoader.execute('../outside.safetensors')
    assert AudioCodecDecode.define_schema().outputs[0].is_output_list
    assert AudioCodecEncode.define_schema().outputs[0].io_type == 'AUDIO_CODES'
    extension = asyncio.run(comfy_entrypoint())
    assert asyncio.run(extension.get_node_list()) == [AudioCodecLoader, AudioCodecEncode, AudioCodecDecode]
