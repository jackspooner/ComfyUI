import torch
import pytest

from comfy.cli_args import args
args.cpu = True

from comfy_extras.nodes_minimax_h3 import MiniMaxH3ContactSheet, MiniMaxH3ContactSheetDecode
from comfy.nested_tensor import NestedTensor


class Clip:
    def tokenize(self, prompt, **kwargs):
        self.references = kwargs["minimax_ref_items"]
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones(1), {"prompt": tokens}]]


class Vae:
    def encode(self, image):
        self.image = image
        return torch.ones(1, 24, 2, image.shape[1] // 16, image.shape[2] // 16)

    def decode(self, latent):
        assert latent.shape[2] == 2
        torch.testing.assert_close(latent[:, :, 0], latent[:, :, 1])
        return latent[0, 0, 0, 0, 0].expand(1, 5, 2, 3, 3)


def test_packed_slots_and_reference_geometry():
    clip, vae = Clip(), Vae()
    cond, latent = MiniMaxH3ContactSheet.execute(clip, vae, torch.zeros(1, 64, 128, 4), "<Picture 1>", 64).result
    video, audio = latent["samples"].unbind()
    assert video.shape == (1, 24, 5, 4, 4)
    assert audio.shape == (1, 32, 2, 28)
    assert vae.image.shape == (1, 32, 96, 3)
    assert cond[0][1]["minimax_refs"][0]["latent_w"] == 6
    assert clip.references[0]["data"] is vae.image


def test_each_slot_decoded_independently_in_order():
    video = torch.arange(5).reshape(1, 1, 5, 1, 1).expand(1, 24, 5, 2, 2).float()
    samples = {"samples": NestedTensor((video, torch.zeros(1, 32, 2, 28)))}
    views, sheet = MiniMaxH3ContactSheetDecode.execute(Vae(), samples).result
    assert views.shape == (5, 2, 3, 3)
    assert sheet.shape == (1, 2, 15, 3)
    for index in range(5):
        assert torch.all(views[index] == index)
        assert torch.all(sheet[:, :, index * 3:(index + 1) * 3] == index)


def test_rejects_video_latent_in_contact_sheet_decoder():
    with pytest.raises(ValueError, match="five-view"):
        MiniMaxH3ContactSheetDecode.execute(Vae(), {"samples": NestedTensor((torch.zeros(1, 24, 7, 2, 2), torch.zeros(1, 32, 2, 28)))})
