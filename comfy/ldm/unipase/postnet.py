"""UniPASE PostNet, adapted from ESPnet TFGridNet (Apache-2.0)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import comfy.ops
import comfy.model_management as mm
import comfy.model_prefetch as prefetch
from comfy.audio import resample
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention

class LSTM(nn.LSTM):

    def reset_parameters(self):
        pass

class PReLU(nn.PReLU):
    def reset_parameters(self):
        pass


class TFGridNet(nn.Module):

    def __init__(self, n_srcs=1, n_imics=1, n_bands=3, n_layers=5, lstm_hidden_units=100, attn_n_head=4, attn_qk_output_channel=2, emb_dim=48, emb_ks=4, emb_hs=1, activation='prelu', eps=1e-05, dtype=None, device=None, operations=None):
        super().__init__()
        self.bs = BandSplit(n_bands=n_bands)
        t_ksize = 3
        ks, padding = ((t_ksize, 3), (t_ksize // 2, 1))
        self.conv = nn.Sequential(operations.Conv2d(2 * n_imics * n_bands, emb_dim, ks, padding=padding, dtype=dtype, device=device), operations.GroupNorm(1, emb_dim, eps=eps, dtype=dtype, device=device))
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(GridNetV3Block(emb_dim, emb_ks, emb_hs, lstm_hidden_units, n_head=attn_n_head, qk_output_channel=attn_qk_output_channel, activation=activation, eps=eps, dtype=dtype, device=device, operations=operations))
        self.deconv = operations.ConvTranspose2d(emb_dim, n_srcs * 2 * n_bands, ks, padding=padding, dtype=dtype, device=device)

    def forward(self, x, sr_in=16000, sr_out=None, transformer_options={}):
        n_samples = x.shape[-1]
        if sr_out is None:
            sr_out = sr_in
        if x.ndim == 3:
            x = x.squeeze(1)
        std_ = torch.std(x, dim=1, keepdim=True) + 1e-12
        mean_ = torch.mean(x, dim=1, keepdim=True)
        x = (x - mean_) / std_
        if sr_in != 48000:
            x = resample(x, sr_in, 48000)
        nfft = int(48000 * 0.032)
        nhop = nfft // 2
        input = torch.stft(x, nfft, nhop, nfft, torch.hann_window(nfft, device=x.device, dtype=x.dtype), return_complex=True)
        input = torch.view_as_real(input)
        spec_ref = input.clone()
        batch = input.permute(0, 3, 2, 1)
        batch = self.bs(batch)
        batch = self.conv(batch)
        queue = prefetch.make_prefetch_queue(list(self.blocks), batch.device, transformer_options)
        with prefetch.malloc_graph_scope(batch.device):
            for block in self.blocks:
                mm.throw_exception_if_processing_interrupted()
                prefetch.prefetch_queue_pop(queue, batch.device, block, batch.dtype, malloc_scope='unipase_postnet')
                batch = block(batch)
            prefetch.prefetch_queue_pop(queue, batch.device, None, malloc_scope='unipase_postnet')
        batch = self.deconv(batch)
        batch = self.bs.inverse(batch)
        batch = batch.permute(0, 3, 2, 1)
        freqs = torch.arange(spec_ref.shape[1], device=batch.device) * 48000 / nfft
        cutoff_bw = 8000
        transition_bw = 800
        alpha = torch.zeros_like(freqs)
        alpha[freqs <= cutoff_bw - transition_bw] = 1.0
        mask = (freqs > cutoff_bw - transition_bw) & (freqs <= cutoff_bw)
        alpha[mask] = (cutoff_bw - freqs[mask]) / transition_bw
        alpha = alpha[None, :, None, None]
        batch_out = spec_ref + (1 - alpha) * batch
        output = torch.istft(batch_out[..., 0] + 1j * batch_out[..., 1], nfft, nhop, nfft, torch.hann_window(nfft, device=x.device, dtype=x.dtype))
        if sr_out != 48000:
            output = resample(output, orig_freq=48000, new_freq=sr_out)
        if sr_out == sr_in:
            out_samples = n_samples
        else:
            out_samples = int(n_samples * sr_out / sr_in)
        if output.shape[-1] < out_samples:
            output = torch.nn.functional.pad(output, (0, out_samples - output.shape[-1]))
        else:
            output = output[..., :out_samples]
        output = output * std_
        return output

class BandSplit(nn.Module):

    def __init__(self, n_bands=3):
        super().__init__()
        self.n_bands = n_bands

    def forward(self, x):
        n_freqs = x.shape[-1]
        assert n_freqs == 769
        sub_nfreqs = n_freqs // self.n_bands + 1
        x_bands = []
        for i in range(self.n_bands):
            x_bands.append(x[..., i * (sub_nfreqs - 1):i * (sub_nfreqs - 1) + sub_nfreqs])
        return torch.cat(x_bands, dim=1).contiguous()

    def inverse(self, x_bands):
        assert x_bands.shape[1] == self.n_bands * 2
        x = [x_bands[:, :2]]
        for i in range(1, self.n_bands):
            x.append(x_bands[:, 2 * i:2 * (i + 1), :, 1:])
        return torch.cat(x, dim=-1).contiguous()

class GridNetV3Block(nn.Module):

    def __init__(self, emb_dim, emb_ks, emb_hs, hidden_channels, n_head=4, qk_output_channel=4, activation='prelu', eps=1e-05, dtype=None, device=None, operations=None):
        super().__init__()
        assert activation == 'prelu'
        in_channels = emb_dim * emb_ks
        self.intra_norm = operations.LayerNorm(emb_dim, eps=eps, dtype=dtype, device=device)
        self.intra_rnn = LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True, dtype=dtype, device=device)
        if emb_ks == emb_hs:
            self.intra_linear = operations.Linear(hidden_channels * 2, in_channels, dtype=dtype, device=device)
        else:
            self.intra_linear = operations.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs, dtype=dtype, device=device)
        self.inter_norm = operations.LayerNorm(emb_dim, eps=eps, dtype=dtype, device=device)
        self.inter_rnn = LSTM(in_channels, hidden_channels, 1, batch_first=True, bidirectional=True, dtype=dtype, device=device)
        if emb_ks == emb_hs:
            self.inter_linear = operations.Linear(hidden_channels * 2, in_channels, dtype=dtype, device=device)
        else:
            self.inter_linear = operations.ConvTranspose1d(hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs, dtype=dtype, device=device)
        E = qk_output_channel
        assert emb_dim % n_head == 0
        self.add_module('attn_conv_Q', operations.Conv2d(emb_dim, n_head * E, 1, dtype=dtype, device=device))
        self.add_module('attn_norm_Q', AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps, dtype=dtype, device=device, operations=operations))
        self.add_module('attn_conv_K', operations.Conv2d(emb_dim, n_head * E, 1, dtype=dtype, device=device))
        self.add_module('attn_norm_K', AllHeadPReLULayerNormalization4DC((n_head, E), eps=eps, dtype=dtype, device=device, operations=operations))
        self.add_module('attn_conv_V', operations.Conv2d(emb_dim, n_head * emb_dim // n_head, 1, dtype=dtype, device=device))
        self.add_module('attn_norm_V', AllHeadPReLULayerNormalization4DC((n_head, emb_dim // n_head), eps=eps, dtype=dtype, device=device, operations=operations))
        self.add_module('attn_concat_proj', nn.Sequential(operations.Conv2d(emb_dim, emb_dim, 1, dtype=dtype, device=device), PReLU(), LayerNormalization(emb_dim, dim=-3, total_dim=4, eps=eps, dtype=dtype, device=device, operations=operations)))
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.comfy_attention = ComfyAttention()

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        olp = self.emb_ks - self.emb_hs
        T = math.ceil((old_T + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q + 2 * olp - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = x.permute(0, 2, 3, 1)
        x = F.pad(x, (0, 0, olp, Q - old_Q - olp, olp, T - old_T - olp))
        input_ = x
        intra_rnn = self.intra_norm(input_)
        if self.emb_ks == self.emb_hs:
            intra_rnn = intra_rnn.view([B * T, -1, self.emb_ks * C])
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([B, T, Q, C])
        else:
            intra_rnn = intra_rnn.view([B * T, Q, C])
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn, _ = self.intra_rnn(intra_rnn)
            intra_rnn = intra_rnn.transpose(1, 2)
            intra_rnn = self.intra_linear(intra_rnn)
            intra_rnn = intra_rnn.view([B, T, C, Q])
            intra_rnn = intra_rnn.transpose(-2, -1)
        intra_rnn = intra_rnn + input_
        intra_rnn = intra_rnn.transpose(1, 2)
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)
        if self.emb_ks == self.emb_hs:
            inter_rnn = inter_rnn.view([B * Q, -1, self.emb_ks * C])
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([B, Q, T, C])
        else:
            inter_rnn = inter_rnn.view(B * Q, T, C)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn, _ = self.inter_rnn(inter_rnn)
            inter_rnn = inter_rnn.transpose(1, 2)
            inter_rnn = self.inter_linear(inter_rnn)
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.transpose(-2, -1)
        inter_rnn = inter_rnn + input_
        inter_rnn = inter_rnn.permute(0, 3, 2, 1)
        inter_rnn = inter_rnn[..., olp:olp + old_T, olp:olp + old_Q]
        batch = inter_rnn
        Q = self.attn_norm_Q(self.attn_conv_Q(batch))
        K = self.attn_norm_K(self.attn_conv_K(batch))
        V = self.attn_norm_V(self.attn_conv_V(batch))
        Q = Q.view(-1, *Q.shape[2:])
        K = K.view(-1, *K.shape[2:])
        V = V.view(-1, *V.shape[2:])
        Q = Q.transpose(1, 2)
        Q = Q.flatten(start_dim=2)
        K = K.transpose(2, 3)
        K = K.contiguous().view([B * self.n_head, -1, old_T])
        V = V.transpose(1, 2)
        old_shape = V.shape
        V = V.flatten(start_dim=2)
        emb_dim = Q.shape[-1]
        width = V.shape[-1]
        Q = F.pad(Q, (0, width - Q.shape[-1]))
        K = F.pad(K.transpose(1, 2), (0, width - K.shape[1]))
        V = optimized_attention(AttentionTensorContainer(Q), AttentionTensorContainer(K), AttentionTensorContainer(V), 1, scale=emb_dim ** (-0.5), preferred_attention=self.comfy_attention)
        V = V.reshape(old_shape)
        V = V.transpose(1, 2)
        emb_dim = V.shape[1]
        batch = V.contiguous().view([B, self.n_head * emb_dim, old_T, old_Q])
        batch = self.attn_concat_proj(batch)
        out = batch + inter_rnn
        return out

class LayerNormalization(nn.Module):

    def __init__(self, input_dim, dim=1, total_dim=4, eps=1e-05, dtype=None, device=None, operations=None):
        super().__init__()
        self.dim = dim if dim >= 0 else total_dim + dim
        param_size = [1 if ii != self.dim else input_dim for ii in range(total_dim)]
        self.gamma = nn.Parameter(torch.empty(*param_size, dtype=dtype, device=device))
        self.beta = nn.Parameter(torch.empty(*param_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x):
        y = x.movedim(self.dim, -1)
        y = F.layer_norm(y, (y.shape[-1],), eps=self.eps).movedim(-1, self.dim)
        return y * comfy.ops.cast_to_input(self.gamma, y) + comfy.ops.cast_to_input(self.beta, y)

class AllHeadPReLULayerNormalization4DC(nn.Module):

    def __init__(self, input_dimension, eps=1e-05, dtype=None, device=None, operations=None):
        super().__init__()
        assert len(input_dimension) == 2, input_dimension
        H, E = input_dimension
        param_size = [1, H, E, 1, 1]
        self.gamma = Parameter(torch.empty(*param_size, dtype=dtype, device=device))
        self.beta = Parameter(torch.empty(*param_size, dtype=dtype, device=device))
        self.act = PReLU(num_parameters=H, init=0.25)
        self.eps = eps
        self.H = H
        self.E = E

    def forward(self, x):
        assert x.ndim == 4
        B, _, T, bins = x.shape
        x = x.view([B, self.H, self.E, T, bins])
        x = self.act(x)
        y = x.movedim(2, -1)
        y = F.layer_norm(y, (self.E,), eps=self.eps).movedim(-1, 2)
        return y * comfy.ops.cast_to_input(self.gamma, y) + comfy.ops.cast_to_input(self.beta, y)
