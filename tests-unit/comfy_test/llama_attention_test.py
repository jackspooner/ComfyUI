import pytest
import torch

from comfy.cli_args import args
args.cpu = True
from comfy.ldm.modules.attention import AttentionTensorContainer, attention_basic, attention_pytorch, wrap_attn
from comfy.ops import manual_cast
from comfy.text_encoders.llama import Attention, FixedKV, Qwen3_8BConfig, precompute_freqs_cis


@pytest.fixture(autouse=True)
def inference_execution():
    with torch.set_grad_enabled(False):
        yield


def make_attention(kv_heads=2, merged=False):
    config = Qwen3_8BConfig(hidden_size=32, num_attention_heads=4, num_key_value_heads=kv_heads, merged_qkv=merged)
    config.head_dim = 8
    module = Attention(config, dtype=torch.float32, ops=manual_cast)
    rng = torch.Generator().manual_seed(13)
    module.load_state_dict({name: torch.randn(value.shape, generator=rng) * 0.1 for name, value in module.state_dict().items()})
    return module


def rotary(start, length):
    return precompute_freqs_cis(8, torch.arange(start, start + length)[None], 1000000, device=torch.device('cpu'))


@pytest.mark.parametrize('kv_heads', [2, 4])
@pytest.mark.parametrize('merged', [False, True])
def test_preference_and_containers_preserve_masked_attention(kv_heads, merged):
    module = make_attention(kv_heads, merged)
    hidden = torch.randn(2, 5, 32)
    mask = torch.full((5, 5), float('-inf')).triu(1)
    expected, _ = module(hidden, mask, rotary(0, 5), attention_pytorch)
    calls = []

    @wrap_attn
    def selected(q, k, v, heads, **kwargs):
        calls.append((q.shape, k.shape, kwargs['mask']))
        return attention_pytorch(q, k, v, heads, **kwargs)

    def dispatch(q, k, v, *positional, **kwargs):
        assert all(isinstance(value, AttentionTensorContainer) for value in (q, k, v))
        assert kwargs['preferred_attention'] is module.comfy_attention
        result = attention_basic(q, k, v, *positional, **kwargs)
        assert q.tensor is k.tensor is v.tensor is None
        return result

    module.comfy_attention.function = selected
    actual, _ = module(hidden, mask, rotary(0, 5), dispatch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(calls) == 1
    assert calls[0][0] == (2, 4, 5, 8)
    assert calls[0][1] == (2, kv_heads, 5, 8)
    assert calls[0][2] is mask


@pytest.mark.parametrize('sliding_window', [None, 3])
def test_dynamic_cache_matches_full_prefix(sliding_window):
    module = make_attention()
    module.comfy_attention.function = attention_pytorch
    hidden = torch.randn(1, 7, 32)
    cache = (torch.empty(1, 2, 10, 8), torch.empty(1, 2, 10, 8), 0)
    for position in range(hidden.shape[1]):
        actual, cache = module(hidden[:, position:position+1], None, rotary(position, 1), attention_basic, cache, sliding_window)
        start = max(0, position + 1 - sliding_window) if sliding_window else 0
        mask = torch.full((position + 1, position + 1), float('-inf')).triu(1)
        mask[-1, :start] = float('-inf')
        expected, _ = module(hidden[:, :position+1], mask, rotary(0, position+1), attention_pytorch)
        torch.testing.assert_close(actual, expected[:, -1:], atol=1e-7, rtol=1e-5)
        assert cache[2] == position + 1


@pytest.mark.parametrize('kv_heads', [2, 4])
def test_fixed_cache_preference_masks_uninitialized_slots(kv_heads, monkeypatch):
    module = make_attention(kv_heads)
    hidden = torch.randn(2, 1, 32)
    key = torch.full((2, 9, kv_heads, 8), float('nan'))
    value = torch.full_like(key, float('nan'))
    lengths = [2, 4]
    for batch, length in enumerate(lengths):
        key[batch, :length] = torch.randn(length, kv_heads, 8)
        value[batch, :length] = torch.randn(length, kv_heads, 8)
    expected = []
    for batch, length in enumerate(lengths):
        cache = (key[batch:batch+1, :length].transpose(1, 2).clone(), value[batch:batch+1, :length].transpose(1, 2).clone(), length)
        output, _ = module(hidden[batch:batch+1], None, rotary(4, 1), attention_pytorch, cache)
        expected.append(output)

    def forbidden_flash(*args):
        pytest.fail('Fixed-cache decode bypassed the attention preference')

    monkeypatch.setattr('comfy_kitchen.flash_attention_decode', forbidden_flash)
    module.comfy_attention.function = attention_pytorch
    cache = FixedKV(key, value, 4, torch.empty(2, dtype=torch.int64), torch.tensor(lengths, dtype=torch.int32))
    cache.prepare(1)
    actual, returned = module(hidden, None, rotary(4, 1), attention_basic, cache)
    assert returned is cache
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.cat(expected), atol=1e-7, rtol=1e-5)
    assert torch.isnan(key[:, 5:]).all()


def test_default_fixed_cache_retains_flash_decode(monkeypatch):
    module = make_attention()
    hidden = torch.randn(1, 1, 32)
    cache = FixedKV(torch.zeros(1, 8, 2, 8), torch.zeros(1, 8, 2, 8), 3,
                    torch.tensor([3]), torch.tensor([4], dtype=torch.int32))
    calls = []

    def flash(q, key, value, seqlen):
        calls.append((q.shape, seqlen.tolist()))
        return torch.ones_like(q)

    monkeypatch.setattr('comfy_kitchen.flash_attention_decode', flash)
    output, returned = module(hidden, None, rotary(3, 1), attention_basic, cache)
    torch.testing.assert_close(output, module.o_proj(torch.ones(1, 1, 32)))
    assert returned is cache
    assert calls == [(torch.Size([1, 1, 4, 8]), [4])]


def test_preference_has_no_extra_checkpoint_weights():
    module = make_attention()
    state = module.state_dict()
    assert not any('comfy_attention' in name for name in state)
    make_attention().load_state_dict(state, strict=True)


@pytest.mark.parametrize('kv_heads', [2, 4])
def test_no_rotary_matches_identity_rotation(kv_heads):
    module = make_attention(kv_heads)
    hidden = torch.randn(2, 5, 32)
    mask = torch.full((5, 5), float('-inf')).triu(1)
    identity = (torch.ones(1, 1, 5, 8), torch.zeros(1, 1, 5, 4), torch.zeros(1, 1, 5, 4))
    expected, _ = module(hidden, mask, identity, attention_pytorch)
    actual, _ = module(hidden, mask, None, attention_pytorch)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
