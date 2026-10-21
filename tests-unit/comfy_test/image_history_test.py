import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from comfy.cli_args import args
args.cpu = True
import folder_paths
from comfy.text_generation import read_history, write_history, select_context
from comfy.text_encoders.chat import count_chat_tokens
from comfy.ops import manual_cast
from comfy import model_management
from comfy.sd1_clip import SDClipModel
from comfy.text_encoders.gemma4 import Gemma4VisionEncoder, Gemma4UnifiedVisionEmbedder
from comfy.text_encoders.llama import Gemma3MultiModalProjector
from comfy.text_encoders.qwen3_omni import Qwen3Omni, Qwen2VLImageProcessor
from comfy.text_encoders.qwen_image import Qwen25_7BVLIModel
from comfy.text_encoders.qwen_vl import qwen2vl_mrope_position_ids
from comfy.text_encoders.qwen35 import Qwen35ImageTokenizer, Qwen35
from comfy.text_encoders.qwen3vl import Qwen3VLTokenizer, Qwen3VL
from comfy.text_encoders.qwen_image import QwenImageTokenizer
from comfy.text_encoders.llama import Qwen25_7BVLI, Gemma3_12B, BaseGenerate
from comfy.text_encoders.qwen_vl import process_qwen2vl_images
from comfy.text_encoders.gemma4 import Gemma4_Tokenizer, Gemma4Base, prepare_chat_image
from comfy.text_encoders.lt import Gemma3_Tokenizer
from comfy.text_encoders.glm_ocr import GlmOCR, GlmOCRTokenizer, process_image
from comfy.text_encoders.qwen3_omni import Qwen3OmniTokenizer
from comfy_extras.nodes_textgen import TextGenerate, TextGenerateLTX2Prompt
from comfy_extras import text_history_images as attachments


@pytest.fixture
def storage(tmp_path, monkeypatch):
    user, inputs = tmp_path / 'user', tmp_path / 'input'
    inputs.mkdir()
    monkeypatch.setattr(folder_paths, 'user_directory', str(user))
    monkeypatch.setattr(folder_paths, 'input_directory', str(inputs))
    return inputs


class Clip:
    def __init__(self, fail=False):
        self.tokenizer = Qwen35ImageTokenizer()
        self.fail = fail
        self.active = None

    def tokenize_chat(self, messages, thinking=False):
        self.active = copy.deepcopy(messages)
        return self.tokenizer.tokenize_chat(messages, thinking)

    def count_chat_tokens(self, tokens):
        return count_chat_tokens(tokens, lambda embed: Qwen35.image_token_length(None, embed))

    def get_chat_context_length(self):
        return 8192

    def generate(self, tokens, **kwargs):
        if self.fail:
            raise RuntimeError('cancelled generation')
        return [42]

    def decode(self, ids):
        return 'A red square.'


def run(clip, **kwargs):
    return TextGenerate.execute(clip, kwargs.pop('prompt', ''), 16, {'sampling_mode': 'off'}, **kwargs).result


def test_image_first_turn_replay_and_different_size_second_image(storage):
    pixels = torch.rand(2, 64, 96, 3)
    first = run(Clip(), image=pixels)
    history = json.loads(first[2])
    assert history['version'] == 2
    ids = [p['image_id'] for p in history['messages'][0]['content'] if p['type'] == 'image']
    assert len(ids) == 2
    for i, identifier in enumerate(ids):
        assert torch.equal(attachments.load_attachment(identifier), pixels[i:i + 1])
    clip = Clip()
    second = run(clip, prompt='Compare with the earlier images.', history=first[2], image=torch.rand(1, 96, 64, 3))
    assert len(json.loads(second[2])['messages']) == 4
    assert clip.active[0]['content'][0]['data'].shape == (1, 64, 96, 3)
    assert clip.active[2]['content'][0]['data'].shape == (1, 96, 64, 3)
    follow = Clip()
    run(follow, prompt='What was the first colour?', history=first[2])
    assert torch.equal(follow.active[0]['content'][0]['data'], pixels[:1])
    assert json.loads(first[2]) == history


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_exact_snapshot_deduplication_and_owning_slice(storage, dtype):
    batch = torch.rand(3, 32, 48, 4).to(dtype)
    messages = read_history('', 'hello')
    _, saved, pending = attachments.resolve_images(messages, batch[1:2])
    image = next(iter(pending.values()))
    assert image.untyped_storage().nbytes() == image.numel() * image.element_size()
    attachments.publish_images(pending)
    identifier = saved[0]['content'][0]['image_id']
    path = attachments.attachment_path(identifier)
    before = os.stat(path).st_mtime_ns
    attachments.publish_images(pending)
    assert os.stat(path).st_mtime_ns == before
    actual = attachments.load_attachment(identifier)
    assert actual.dtype == dtype and torch.equal(actual, batch[1:2])


def test_ordered_input_files_snapshot_and_deleted_upload(storage):
    Image.new('RGB', (48, 64), 'red').save(storage / 'red.png')
    parts = [{'type': 'text', 'text': 'Before'}, {'type': 'image', 'file': {'filename': 'red.png', 'subfolder': ''}}, {'type': 'text', 'text': 'After'}]
    clip = Clip()
    first = run(clip, prompt_content=json.dumps(parts))
    assert [p['type'] for p in clip.active[0]['content']] == ['text', 'image', 'text']
    (storage / 'red.png').unlink()
    run(Clip(), prompt='Again?', history=first[2])
    assert 'file' not in json.loads(first[2])['messages'][0]['content'][1]


def test_missing_corrupt_and_escaped_attachments(storage):
    first = run(Clip(), image=torch.ones(1, 32, 32, 3))
    identifier = json.loads(first[2])['messages'][0]['content'][0]['image_id']
    path = attachments.attachment_path(identifier)
    save_file({'image': torch.zeros(1, 32, 32, 3)}, path)
    with pytest.raises(ValueError, match='corrupt'):
        run(Clip(), history=first[2])
    os.unlink(path)
    with pytest.raises(ValueError, match='missing'):
        run(Clip(), history=first[2])
    with pytest.raises(ValueError, match='SHA-256'):
        attachments.attachment_path('../escape')
    (storage / 'escape.png').symlink_to(storage.parent / 'outside.png')
    for file in [{'filename': '../outside.png', 'subfolder': ''}, {'filename': 'escape.png', 'subfolder': ''}, {'filename': 'x.png', 'subfolder': '..'}]:
        with pytest.raises(ValueError):
            attachments.input_image_path(file)


def test_failures_publish_no_history_or_partial_files(storage, monkeypatch):
    with pytest.raises(RuntimeError, match='cancelled'):
        run(Clip(fail=True), image=torch.ones(1, 32, 32, 3))
    root = storage.parent / 'user'
    assert not list(root.rglob('*.safetensors'))
    monkeypatch.setattr(attachments, 'save_file', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        run(Clip(), image=torch.ones(1, 32, 32, 3))
    assert not list(root.rglob('*.tmp')) and not list(root.rglob('*.safetensors'))


def test_file_fingerprint_changes_and_linked_history_invalidates(storage):
    file = {'filename': 'image.png', 'subfolder': ''}
    Image.new('RGB', (32, 32), 'red').save(storage / file['filename'])
    content = json.dumps([{'type': 'image', 'file': file}])
    first = TextGenerate.fingerprint_inputs(prompt_content=content)
    Image.new('RGB', (32, 32), 'blue').save(storage / file['filename'])
    assert TextGenerate.fingerprint_inputs(prompt_content=content) != first
    assert str(TextGenerate.fingerprint_inputs(history=None)) == 'nan'


def test_context_counts_visual_tokens_and_drops_complete_turn(storage):
    clip = Clip()
    first = run(clip, prompt='old', image=torch.ones(1, 224, 224, 3))
    info = json.loads(first[3])
    tokens = clip.tokenizer.tokenize_chat(clip.active)
    assert info['input_tokens'] == sum(isinstance(t[0], int) for row in next(iter(tokens.values())) for t in row) + 49
    messages = read_history(first[2], 'now')
    resolved, _, _ = attachments.resolve_images(messages)
    _, exact = select_context(resolved, clip.tokenize_chat, 8192, 0, 16, 'error', clip.count_chat_tokens)
    limit = exact['input_tokens'] + 16
    select_context(resolved, clip.tokenize_chat, 8192, limit, 16, 'error', clip.count_chat_tokens)
    with pytest.raises(ValueError, match='Context overflow'):
        select_context(resolved, clip.tokenize_chat, 8192, limit - 1, 16, 'error', clip.count_chat_tokens)
    _, trimmed = select_context(resolved, clip.tokenize_chat, 8192, limit - 1, 16, 'drop_oldest_turns', clip.count_chat_tokens)
    assert trimmed['excluded_message_ranges'] == [[0, 2]]
    assert len(resolved) == 3


@pytest.mark.parametrize('tokenizer_class', [Qwen35ImageTokenizer, Qwen3VLTokenizer, QwenImageTokenizer])
def test_qwen_image_slots_cannot_be_consumed_by_literal_markers(tokenizer_class):
    tokenizer = tokenizer_class()
    image = torch.rand(1, 32, 64, 3)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': '<|image_pad|>literal'}, {'type': 'image', 'data': image}, {'type': 'text', 'text': 'tail'}]}]
    row = next(iter(tokenizer.tokenize_chat(messages).values()))[0]
    images = [(i, t[0]) for i, t in enumerate(row) if isinstance(t[0], dict)]
    assert len(images) == 1 and images[0][1]['data'] is image
    prefix = getattr(tokenizer, tokenizer.clip).tokenizer.decode([t[0] for t in row[:images[0][0]]])
    assert '<|image_pad|>literal' in prefix


@pytest.mark.parametrize('height,width', [(32, 64), (224, 224), (271, 383), (16, 16)])
@pytest.mark.parametrize('model,patch', [(Qwen25_7BVLI, 14), (Qwen35, 16), (Qwen3VL, 16)])
def test_qwen_counts_match_real_patch_processing(height, width, model, patch):
    image = torch.rand(1, height, width, 3)
    patches, _ = process_qwen2vl_images(image, patch_size=patch)
    assert model.image_token_length(None, {'data': image}) == patches.shape[0] // 4


def test_glm_count_matches_real_preprocessing():
    image = torch.rand(1, 71, 103, 3)
    patches, _ = process_image(image)
    assert GlmOCR.image_token_length(None, {'data': image}) == patches.shape[0] // 4


def test_text_only_model_rejects_images():
    with pytest.raises(ValueError, match='does not support'):
        BaseGenerate().count_chat_tokens({'model': [[({'type': 'image', 'data': torch.zeros(1, 32, 32, 3)}, 1)]]})


def test_version_two_text_only_downgrades_and_conflicting_inputs_fail(storage):
    history = write_history(read_history('{"version":2,"messages":[]}', '', prompt_content='[{"type":"text","text":"hello"}]'))
    assert json.loads(history) == {'version': 1, 'messages': [{'role': 'user', 'content': 'hello'}]}
    with pytest.raises(ValueError, match='empty prompt'):
        run(Clip(), prompt='ambiguous', prompt_content='[]')
    assert TextGenerateLTX2Prompt.history_output is False
    assert 'prompt_content' not in {i.id for i in TextGenerateLTX2Prompt.define_schema().inputs}


class CaptureTokenizer:
    def __call__(self, text):
        return {'input_ids': self.encode(text)}

    def encode(self, text, **kwargs):
        return list(text.encode())


def test_gemma_and_glm_roles_and_ordered_images():
    pixels = torch.rand(1, 64, 96, 3)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'first'}, {'type': 'image', 'data': pixels}, {'type': 'text', 'text': 'last'}]}, {'role': 'assistant', 'content': 'old'}, {'role': 'user', 'content': 'follow'}]
    gemma3 = Gemma3_Tokenizer()
    gemma3.tokenizer = SimpleNamespace(tokenizer=CaptureTokenizer())
    gemma4 = Gemma4_Tokenizer()
    gemma4.tokenizer = CaptureTokenizer()
    gemma4.prime_empty_thought = True
    for tokenizer, role_start, end in [(gemma3, '<start_of_turn>', '<end_of_turn>'), (gemma4, '<|turn>', '<turn|>')]:
        for thinking in (True, False):
            row = tokenizer.tokenize_chat(messages, thinking)[0]
            assert row[0][0] == 2
            text = bytes(t[0] for t in row[1:] if isinstance(t[0], int)).decode()
            assert f'{role_start}model\nold{end}' in text
            assert 'first\n\n<start_of_image><end_of_image>\n\nlast' in text if tokenizer is gemma3 else 'first<|image><image|>last' in text
            assert len([t for t in row if isinstance(t[0], dict)]) == 1
            if tokenizer is gemma4:
                assert ('<|think|>' in text) == thinking
                assert text.endswith('<|turn>model\n' + ('' if thinking else '<|channel>thought\n<channel|>'))
    glm = object.__new__(GlmOCRTokenizer)
    glm.glm_ocr = SimpleNamespace(tokenizer=CaptureTokenizer())
    row = next(iter(glm.tokenize_chat(messages).values()))[0]
    assert len([t for t in row if isinstance(t[0], dict)]) == 1
    ids = [t[0] for t in row if isinstance(t[0], int)]
    text = bytes(ids).decode()
    assert text.startswith('[gMASK]<sop><|user|>\nfirst<|begin_of_image|>')
    assert text.endswith('<|user|>\nfollow<|assistant|>\n')


@pytest.mark.parametrize('height,width', [(97, 281), (288, 192)])
def test_gemma4_counts_match_both_real_vision_expansion_paths(height, width):
    embed = prepare_chat_image(torch.rand(1, height, width, 3))
    cfg = {'patch_size': 16, 'pooling_kernel_size': 3, 'hidden_size': 8, 'head_dim': 4,
           'num_attention_heads': 2, 'num_hidden_layers': 0, 'rms_norm_eps': 1e-6,
           'position_embedding_size': 10240, 'intermediate_size': 16,
           'model_patch_size': 48, 'mm_embed_dim': 8, 'mm_posemb_size': 1120}
    owner = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vision_config=cfg)))
    count = Gemma4Base.image_token_length(owner, embed)
    pixels = embed['data'].movedim(-1, 1)
    encoder = Gemma4VisionEncoder(cfg, dtype=torch.float32, device='cpu', ops=manual_cast)
    assert encoder(pixels, max_soft_tokens=280).shape[1] == count
    unified = Gemma4UnifiedVisionEmbedder(cfg, dtype=torch.float32, device='cpu', ops=manual_cast)
    patches, positions = unified.patchify(pixels)
    assert unified(patches, positions).shape[1] == count


def test_gemma3_count_matches_real_projection():
    cfg = SimpleNamespace(vision_config={'image_size': 896, 'patch_size': 14, 'hidden_size': 8},
                          mm_tokens_per_image=256, hidden_size=16, rms_norm_eps=1e-6, rms_norm_add=True)
    projector = Gemma3MultiModalProjector(cfg, dtype=torch.float32, device='cpu', operations=manual_cast)
    output = projector(torch.rand(1, 4096, 8))
    assert output.shape[1] == Gemma3_12B.image_token_length(SimpleNamespace(model=SimpleNamespace(config=cfg)), {'data': torch.rand(1, 37, 71, 3)})


def test_omni_counts_match_real_processor_and_structured_slots():
    config = Qwen2VLImageProcessor().to_dict()
    model = SimpleNamespace(processor_config={'image_processor': config})
    pixels = torch.rand(1, 85, 173, 3)
    result = Qwen2VLImageProcessor.from_dict(config)(images=pixels[0].movedim(-1, 0), do_rescale=False, return_tensors='pt')
    assert Qwen3Omni.image_token_length(model, {'data': pixels}) == result.pixel_values.shape[0] // 4
    tokenizer = object.__new__(Qwen3OmniTokenizer)
    tokenizer.qwen3_omni = SimpleNamespace(tokenizer=Qwen35ImageTokenizer().qwen35_2b.tokenizer)
    tokens = tokenizer.tokenize_chat([{'role': 'user', 'content': [{'type': 'image', 'data': pixels}]}])
    assert sum(isinstance(t[0], dict) for t in tokens['qwen3_omni'][0]) == 1


def test_qwen25_generation_forwards_multimodal_positions():
    info = [{'type': 'image', 'index': 2, 'size': 4, 'extra': {'grid': (1, 4, 4)}}]
    # Use the actual preprocessing metadata layout rather than fabricating image positions.
    _, grid = process_qwen2vl_images(torch.rand(1, 56, 56, 3))
    info[0]['extra'] = grid
    captured = {}
    def generate(*args, **kwargs):
        captured.update(kwargs)
        return [42]
    owner = SimpleNamespace(execution_device='cpu', process_tokens=lambda *args: (torch.zeros(1, 9, 8), None, None, info), transformer=SimpleNamespace(generate=generate))
    Qwen25_7BVLIModel.generate(owner, [[(1, 1)]], False, 4, 1, 1, 1, 0, 1, 0)
    torch.testing.assert_close(captured['position_ids'], qwen2vl_mrope_position_ids(info, 9, torch.device('cpu')))


def test_cancellation_before_publication_and_ltx_no_writes(storage, monkeypatch):
    monkeypatch.setattr(model_management, 'throw_exception_if_processing_interrupted', lambda: (_ for _ in ()).throw(RuntimeError('cancelled')))
    with pytest.raises(RuntimeError, match='cancelled'):
        run(Clip(), image=torch.rand(1, 32, 32, 3))
    assert not list(storage.parent.rglob('*.safetensors'))
    clip = SimpleNamespace(tokenize=lambda *a, **k: {'text': [[(1, 1)]]}, generate=lambda *a, **k: [1], decode=lambda *a: 'prompt')
    TextGenerate.execute.__func__(TextGenerateLTX2Prompt, clip, 'Describe motion', 16, {'sampling_mode': 'off'}, image=torch.rand(1, 32, 32, 3))
    assert not list(storage.parent.rglob('*.safetensors'))


@pytest.mark.parametrize('fault', ['length', 'width', 'missing'])
def test_counted_image_expansion_must_match_actual_embeddings(fault):
    def embedding(ids, out_dtype):
        return torch.zeros(*ids.shape, 8)
    transformer = SimpleNamespace(get_input_embeddings=lambda: embedding,
        preprocess_embed=lambda *args, **kwargs: (None if fault == 'missing' else torch.zeros(1, 4, 9 if fault == 'width' else 8), None))
    owner = SimpleNamespace(special_tokens={}, transformer=transformer)
    image = {'type': 'image', 'data': torch.rand(1, 32, 32, 3), 'expected_length': 3 if fault == 'length' else 4}
    with pytest.raises(ValueError, match='Image'):
        SDClipModel.process_tokens(owner, [[1, image, 2]], 'cpu')


def test_shared_attachments_survive_partial_publication_and_symlink_is_rejected(storage, monkeypatch):
    _, saved, pending = attachments.resolve_images(read_history('', ''), torch.rand(2, 32, 32, 3))
    first, second = list(pending)
    original = attachments.save_file
    def fail_second(tensors, path):
        if attachments.image_id(tensors['image']) == second:
            raise OSError('disk full')
        original(tensors, path)
    monkeypatch.setattr(attachments, 'save_file', fail_second)
    with pytest.raises(OSError):
        attachments.publish_images(pending)
    assert torch.equal(attachments.load_attachment(first), pending[first])
    assert not list(storage.parent.rglob('*.tmp'))
    os.unlink(attachments.attachment_path(first))
    os.symlink(storage.parent / 'outside.safetensors', attachments.attachment_path(first))
    with pytest.raises(ValueError, match='escapes'):
        attachments.load_attachment(first)
