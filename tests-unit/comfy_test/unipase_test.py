"""Native UniPASE boundary and numerical contracts; no checkpoints or services."""
import sys
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F
from comfy.cli_args import args
args.cpu = True
from comfy.unipase import runtime
from comfy.ldm.unipase.wavlm import Attention
from comfy.ldm.unipase.postnet import LayerNormalization, AllHeadPReLULayerNormalization4DC
import comfy.ops


@pytest.mark.parametrize('rate', [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize('seconds', [.001, 1, 8, 8.5, 10, 11, 12, 17, 25])
def test_chunk_stitch_identity(rate, seconds):
    length = int(rate * seconds)
    waveform = torch.linspace(-.8, .8, length).reshape(1, 1, -1).repeat(2, 2, 1)
    def identity(bundle, audio, sr, out, plc):
        return audio.clone()
    with patch.object(runtime, 'segment', identity):
        result = runtime.enhance({}, {'waveform': waveform, 'sample_rate': rate})
    assert result['sample_rate'] == rate
    torch.testing.assert_close(result['waveform'], waveform, atol=2e-7, rtol=0)
    assert max(end-start for start, end in runtime.segments(length, rate)) <= rate*10


@pytest.mark.parametrize('rate,out', [(44100,16000),(22050,48000),(16000,8000)])
def test_changed_rate_long_boundaries(rate, out):
    length = rate * 17 + 7
    waveform = torch.ones(1, 2, length) * .5
    def constant(bundle, audio, sr, target, plc):
        return torch.ones(audio.shape[0], audio.shape[-1] * target // sr) * .5
    with patch.object(runtime, 'segment', constant):
        result = runtime.enhance({}, {'waveform':waveform,'sample_rate':rate}, out, False)
    assert result['waveform'].shape == (1,2,length*out//rate)
    torch.testing.assert_close(result['waveform'], torch.full_like(result['waveform'], .5))


def test_cancellation_stops_before_model_work():
    with patch.object(runtime.mm,'throw_exception_if_processing_interrupted',side_effect=RuntimeError('cancelled')), patch.object(runtime,'segment') as model:
        with pytest.raises(RuntimeError,match='cancelled'):
            runtime.enhance({}, {'waveform':torch.ones(1,1,16000),'sample_rate':16000})
        model.assert_not_called()


def test_relative_attention_matches_explicit_math():
    torch.manual_seed(12)
    m = Attention(16, 2, 32, 128, True, torch.float32, 'cpu', torch.nn)
    m.grep_a.data.fill_(1)
    x = torch.randn(2,11,16)
    bias = m.position_bias(x)
    a,b=m.grep_linear(x.reshape(2,11,2,8).transpose(1,2)).reshape(2,2,11,2,4).sum(-1).sigmoid().chunk(2,-1)
    q,k,v=[p(x).reshape(2,11,2,8).transpose(1,2) for p in (m.q_proj,m.k_proj,m.v_proj)]
    expected=((q@k.transpose(-1,-2)/8**.5+(a*(b*m.grep_a-1)+2)*bias).softmax(-1)@v).transpose(1,2).reshape(2,11,16)
    torch.testing.assert_close(m(x,bias),m.out_proj(expected),atol=1e-6,rtol=1e-5)


@pytest.mark.parametrize('head', [False,True])
def test_postnet_normalization_preserves_math(head):
    torch.manual_seed(1)
    if head:
        m=AllHeadPReLULayerNormalization4DC((2,3))
        m.gamma.data.normal_();m.beta.data.normal_()
        x=torch.randn(2,6,7,8);y=m.act(x.reshape(2,2,3,7,8));dim=2
    else:
        m=LayerNormalization(6,dim=1)
        m.gamma.data.normal_();m.beta.data.normal_()
        x=torch.randn(2,6,7,8);y=x;dim=1
    expected=(y-y.mean(dim,keepdim=True))/torch.sqrt(y.var(dim,unbiased=False,keepdim=True)+m.eps)*m.gamma+m.beta
    torch.testing.assert_close(m(x),expected,atol=2e-6,rtol=2e-5)
