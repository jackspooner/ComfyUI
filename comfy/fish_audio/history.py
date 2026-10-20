"""Explicit immutable CPU take histories; safetensors only, no session store."""
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


@dataclass(frozen=True)
class Reference:
    transcript: str
    speaker: int
    codes: torch.Tensor
    compatibility: str


@dataclass(frozen=True)
class Take:
    text: str
    chunks: tuple[tuple[str, torch.Tensor], ...]
    settings: dict


@dataclass(frozen=True)
class History:
    compatibility: str
    references: tuple[Reference, ...] = ()
    takes: tuple[Take, ...] = ()


def validate_codes(codes, reference=False):
    if codes.device.type != 'cpu' or codes.dtype not in (torch.int16, torch.int32, torch.int64) or codes.ndim != 2 or codes.shape[0] != 10 or codes.shape[1] < 1:
        raise ValueError('Drama 3 codes must be nonempty CPU integer [10, frames] tensors')
    if codes.min() < 0 or codes[0].max() >= 4096 or codes[1:].max() >= (1024 if reference else 4096):
        raise ValueError('Drama 3 codes exceed their codebook ranges')


def cpu_codes(codes):
    result = codes.detach().to(device='cpu', dtype=torch.int64, copy=True).contiguous()
    validate_codes(result)
    return result


def validate(history, compatibility=None):
    if not isinstance(history, History) or not isinstance(history.compatibility, str) or not history.compatibility or compatibility is not None and history.compatibility != compatibility:
        raise ValueError('Drama 3 history codec/tokenizer configuration does not match the loaded model')
    for ref in history.references:
        if ref.compatibility != history.compatibility or not isinstance(ref.transcript, str) or type(ref.speaker) is not int or not 0 <= ref.speaker <= 15:
            raise ValueError('Invalid Drama 3 voice reference')
        validate_codes(ref.codes, reference=True)
    for take in history.takes:
        if not isinstance(take.text, str) or not isinstance(take.settings, dict) or not take.chunks:
            raise ValueError('Invalid Drama 3 take')
        for text, codes in take.chunks:
            if not isinstance(text, str):
                raise ValueError('Drama 3 chunk text must be a string')
            validate_codes(codes)


def save(history, path):
    validate(history)
    tensors, references, takes = {}, [], []
    for i, ref in enumerate(history.references):
        key = f'references.{i}'
        tensors[key] = ref.codes.clone().contiguous()
        references.append(dict(transcript=ref.transcript, speaker=ref.speaker, tensor=key))
    for i, take in enumerate(history.takes):
        chunks = []
        for j, (text, codes) in enumerate(take.chunks):
            key = f'takes.{i}.{j}'
            tensors[key] = codes.clone().contiguous()
            chunks.append(dict(text=text, tensor=key))
        takes.append(dict(text=take.text, settings=take.settings, chunks=chunks))
    metadata = dict(version=1, compatibility=history.compatibility, references=references, takes=takes)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        save_file(tensors, str(temporary), metadata={'format': 'drama3_history', 'history': json.dumps(metadata, ensure_ascii=False)})
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path):
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        meta = saved.metadata() or {}
        if meta.get('format') != 'drama3_history':
            raise ValueError('Choose a history saved by Drama3SaveHistory')
        data = json.loads(meta['history'])
        if not isinstance(data, dict) or set(data) != {'version', 'compatibility', 'references', 'takes'} or type(data['version']) is not int or data['version'] != 1:
            raise ValueError('Unsupported Drama 3 history version or fields')
        if not isinstance(data['references'], list) or not isinstance(data['takes'], list):
            raise ValueError('History references and takes must be arrays')
        for ref in data['references']:
            if not isinstance(ref, dict) or set(ref) != {'transcript', 'speaker', 'tensor'}:
                raise ValueError('Invalid saved reference fields')
        for take in data['takes']:
            if not isinstance(take, dict) or set(take) != {'text', 'settings', 'chunks'} or not isinstance(take['chunks'], list):
                raise ValueError('Invalid saved take fields')
            if any(not isinstance(c, dict) or set(c) != {'text', 'tensor'} for c in take['chunks']):
                raise ValueError('Invalid saved chunk fields')
        used = set()
        def read(key):
            if not isinstance(key, str):
                raise ValueError('History tensor name must be a string')
            if key in used:
                raise ValueError('Duplicate history tensor')
            used.add(key)
            return saved.get_tensor(key).clone()
        references = tuple(Reference(r['transcript'], r['speaker'], read(r['tensor']), data['compatibility']) for r in data['references'])
        takes = tuple(Take(t['text'], tuple((c['text'], read(c['tensor'])) for c in t['chunks']), t['settings']) for t in data['takes'])
        if used != set(saved.keys()):
            raise ValueError('Unexpected history tensors')
    history = History(data['compatibility'], references, takes)
    validate(history)
    return history
