import pytest
import torch

from comfy.cli_args import args as cli_args
import comfy.utils

if not torch.cuda.is_available():
    cli_args.cpu = True

from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels  # noqa: E402


def test_output_schema_preserves_image_slot_and_names_dimensions():
    outputs = ImageScaleToTotalPixels.define_schema().outputs

    assert [output.get_io_type() for output in outputs] == ["IMAGE", "INT", "INT"]
    assert [output.id for output in outputs[1:]] == ["width", "height"]


@pytest.mark.parametrize("method", ImageScaleToTotalPixels.upscale_methods)
@pytest.mark.parametrize(
    "shape,megapixels,steps,expected_size",
    [
        ((1, 45, 90, 3), 0.015625, 1, (181, 91)),
        ((2, 90, 45, 4), 0.015625, 32, (96, 192)),
        ((1, 256, 256, 3), 0.015625, 64, (128, 128)),
    ],
)
def test_dimensions_match_scaled_image(method, shape, megapixels, steps, expected_size):
    image = torch.full(shape, 0.5)

    scaled, width, height = ImageScaleToTotalPixels.execute(
        image, method, megapixels, steps,
    ).result

    assert (width, height) == expected_size
    assert scaled.shape == (shape[0], height, width, shape[3])
    assert type(width) is int and type(height) is int
    assert scaled.dtype == image.dtype and scaled.device == image.device
    expected = comfy.utils.common_upscale(
        image.movedim(-1, 1), *expected_size, method, "disabled",
    ).movedim(1, -1)
    torch.testing.assert_close(scaled, expected)
