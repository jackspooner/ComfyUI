from types import SimpleNamespace

import torch
from torch import nn

from comfy.ldm.minimax.model import MiniMaxH3ImageModel
from comfy.ldm.minimax.sampling import MiniMaxH3ImageModelSampling


def test_image_sampler_lands_on_checkpoint_six_step_grid():
    grid = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
    sampling = MiniMaxH3ImageModelSampling(SimpleNamespace(unet_config={"pdd_sigmas": grid}, sampling_settings={}))
    torch.testing.assert_close(sampling.sigma(torch.linspace(1000, 0, 7)), torch.tensor(grid))


def test_image_only_forward_has_no_audio_branch_and_accepts_sixteen_refs():
    model = MiniMaxH3ImageModel(
        pdd_sigmas=[1.0, 0.0], hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=2, text_dim=5, timestep_input_dim=8,
        time_embed_hidden_size=8, time_embed_dim=8, dtype=torch.float32,
        device="cpu", operations=nn,
    )
    assert "audio_patch_proj" not in dict(model.named_modules())
    latent = torch.randn(1, 2, 1, 4, 6)
    refs = [{"kind": "image", "latent_h": 4, "latent_w": 6} for _ in range(16)]
    payload = {"refs": refs, "cond_video_latents": [torch.randn_like(latent) for _ in refs]}
    with torch.no_grad():
        output = model(latent, torch.tensor([500.0]), torch.randn(1, 4, 8), minimax_payload=payload)
    assert output.shape == latent.shape
    assert torch.isfinite(output).all()


def test_conditioning_retains_all_sixteen_reference_slots(monkeypatch):
    from comfy_extras import nodes_minimax_h3 as nodes

    seen = {}

    class Clip:
        def tokenize(self, prompt, **kwargs):
            seen.update(prompt=prompt, **kwargs)
            return prompt

        def encode_from_tokens_scheduled(self, tokens):
            return [[torch.zeros(1, 1, 1), {}]]

    class Vae:
        def encode(self, pixels):
            return torch.full((1, 24, 1, 2, 2), pixels[0, 0, 0, 0].item())

    monkeypatch.setattr(nodes, "_resize_to", lambda image, size: image)
    images = {f"image_{i}": torch.full((1, 32, 32, 3), float(i)) for i in reversed(range(1, 17))}
    positive, latent = nodes.MiniMaxH3ImageConditioning.execute(
        Clip(), "Combine <Picture 1> through <Picture 16>.", 64, 32, vae=Vae(), images=images
    )
    assert seen["picture_label"] == "Picture %d: "
    assert [image[0, 0, 0, 0].item() for image in seen["images"]] == list(range(1, 17))
    refs = positive[0][1]["minimax_refs"]
    assert [ref["latent"][0, 0, 0, 0, 0].item() for ref in refs] == list(range(1, 17))
    assert latent["samples"].shape == (1, 24, 1, 2, 4)
    assert not latent["samples"].is_nested
