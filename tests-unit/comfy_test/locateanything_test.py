"""Small pinned CPU components and protocol fixtures; never full-model inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from comfy.cli_args import args
args.cpu = True
from comfy.locateanything.model import Language, Vision, inference_mask
from comfy.locateanything.processing import prepare_image, prepare_prompt
from comfy.locateanything.sampling import TOKENS, pattern, sample
from comfy.ops import manual_cast
from scripts.convert_locateanything import SOURCE_HASHES, expected_shapes
from scripts.convert_moss_audio import inspect_shards


REFERENCE = r'''
import importlib, json, sys, types
from pathlib import Path
import torch
from PIL import Image
from transformers import Qwen2TokenizerFast
source, output = Path(sys.argv[1]), sys.argv[2]
package = types.ModuleType('locate_pinned_reference')
package.__path__ = [str(source)]
sys.modules[package.__name__] = package
vit = importlib.import_module(package.__name__+'.modeling_vit')
qwen = importlib.import_module(package.__name__+'.modeling_qwen2')
images = importlib.import_module(package.__name__+'.image_processing_locateanything')
helpers = importlib.import_module(package.__name__+'.generate_utils')
processor = importlib.import_module(package.__name__+'.processing_locateanything')
config = json.loads((source/'config.json').read_text())
result = dict(config=config)
torch.manual_seed(7401)
with torch.inference_mode():
    image_processor = images.LocateAnythingImageProcessor(**json.loads((source/'preprocessor_config.json').read_text()))
    result['images'] = []
    for size in ((1, 1), (43, 71), (560, 373), (3000, 2000)):
        image = Image.new('RGB', size, (31, 111, 197))
        pixels, grid = image_processor._preprocess(image)
        result['images'].append((size, pixels, grid))
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(source), local_files_only=True)
    chat = processor.LocateAnythingProcessor(image_processor=image_processor, tokenizer=tokenizer)
    image = Image.new('RGB', (84, 56))
    message = [{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Point to: Save button.'}]}]
    prompt = chat.py_apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = chat(text=[prompt], images=[image], return_tensors='pt')
    result['tokenizer_json'] = tokenizer.backend_tokenizer.to_str()
    result['prompt_ids'] = inputs['input_ids']
    texts = ['  café\t你好\n🙂  ', '<box><0><1000></box>', 'Locate a single instance: "Save".', '\r\nthree   spaces']
    result['tokenization'] = [(text,tokenizer.encode(text,add_special_tokens=False)) for text in texts]
    result['models'] = []
    for dtype in (torch.float32, torch.bfloat16):
        vc = vit.MoonViTConfig(patch_size=2, init_pos_emb_height=2, init_pos_emb_width=2,
            num_attention_heads=4, num_hidden_layers=2, hidden_size=32, intermediate_size=64)
        vc._attn_implementation = 'sdpa'
        vision = vit.MoonVitPretrainedModel(vc).eval().to(dtype)
        projector = torch.nn.Sequential(torch.nn.LayerNorm(128), torch.nn.Linear(128, 32),
            torch.nn.GELU(), torch.nn.Linear(32, 32)).to(dtype)
        pixels = torch.randn(24, 3, 2, 2).to(dtype)
        visual = projector(vision(pixels, torch.tensor([[4,6]]))[0])
        tc = qwen.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128, rope_theta=1000000., rms_norm_eps=1e-6,
            block_size=6, text_mask_token_id=63)
        tc._attn_implementation = 'sdpa'
        language = qwen.Qwen2ForCausalLM(tc).eval().to(dtype)
        fixture = dict(dtype=str(dtype), vision_config=vc.to_dict(), text_config=tc.to_dict(),
            vision_state={'vision_model.'+k:v for k,v in vision.state_dict().items()},
            language_state=language.state_dict(), pixels=pixels, visual=visual, steps=[])
        fixture['vision_state'].update({'mlp1.'+k:v for k,v in projector.state_dict().items()})
        cache = None
        for qlen, past, mtp in ((14,0,True), (12,8,True), (1,14,False)):
            ids = torch.randint(0, 60, (1,qlen))
            positions = torch.arange(past,past+qlen)[None]
            if mtp:
                ids[0,-5:] = 63
                positions[0,-6:] -= 1
            expected = language(ids, position_ids=positions, past_key_values=cache, use_cache=True)
            fixture['steps'].append(dict(ids=ids, positions=positions, mtp=mtp,
                logits=expected.logits[:,-6:] if mtp else expected.logits[:,-1:]))
            keep = 8 if past == 0 else 14 if mtp else 15
            cache = tuple((k[:,:,:keep].clone(),v[:,:,:keep].clone()) for k,v in expected.past_key_values)
        result['models'].append(fixture)
    result['sampling'] = []
    # get_token_ids_from_config needs the nested config as an attribute owner.
    token_ids = helpers.get_token_ids_from_config(types.SimpleNamespace(**{**config,
        'text_config':types.SimpleNamespace(**config['text_config'])}))
    for mode in ('hybrid','fast','slow'):
        for length in (1,6):
            logits = torch.randn(1,length,152681)
            history = torch.tensor([[1,7,8,8,63]])
            torch.manual_seed(19)
            _,_,sampled,box = helpers.sample_tokens(logits, history, token_ids,
                temperature=.7,top_p=.9,repetition_penalty=1.1,generation_mode=mode)
            values = sampled[0] if box is None or (box[0] == 0).all() else box[0]
            result['sampling'].append(dict(mode=mode,logits=logits,history=history,values=values))
torch.save(result,output)
'''


@pytest.fixture(scope='module')
def reference(tmp_path_factory):
    source = os.environ.get('LOCATEANYTHING_REFERENCE')
    runtime = os.environ.get('LOCATEANYTHING_REFERENCE_RUNTIME')
    if not source or not runtime:
        pytest.skip('Requires pinned LOCATEANYTHING_REFERENCE and isolated 4.57.1 LOCATEANYTHING_REFERENCE_RUNTIME; no weights loaded')
    source = Path(source)
    for name, digest in SOURCE_HASHES.items():
        if not name.endswith('.safetensors'):
            assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    output = tmp_path_factory.mktemp('locate')/'reference.pt'
    env = dict(os.environ, PYTHONPATH=runtime, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run([sys.executable, '-c', REFERENCE, str(source), str(output)], env=env, check=True, capture_output=True)
    return torch.load(output, weights_only=True)


@pytest.fixture(autouse=True)
def deterministic():
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(7401)
        yield


def test_image_resizing_normalization_patches_and_prompt_tokenization(reference):
    from tokenizers import Tokenizer
    from transformers import Qwen2TokenizerFast
    processor = json.loads((Path(os.environ['LOCATEANYTHING_REFERENCE'])/'preprocessor_config.json').read_text())
    for size, expected, grid in reference['images']:
        actual, actual_grid = prepare_image(Image.new('RGB', size, (31,111,197)), processor)
        assert tuple(actual_grid) == tuple(grid)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    ids = prepare_prompt(Tokenizer.from_str(reference['tokenizer_json']), 'Point to: Save button.', (4,6))
    torch.testing.assert_close(ids, reference['prompt_ids'], atol=0, rtol=0)
    converted = Qwen2TokenizerFast.from_pretrained(os.environ['LOCATEANYTHING_REFERENCE'], local_files_only=True)
    candidate = Tokenizer.from_str(converted.backend_tokenizer.to_str())
    torch.testing.assert_close(prepare_prompt(candidate, 'Point to: Save button.', (4,6)), ids, atol=0, rtol=0)
    for text, expected in reference['tokenization']:
        assert candidate.encode(text).ids == expected


@pytest.mark.parametrize('index', [0,1])
def test_small_complete_vision_projector_and_cached_qwen_match(reference, index):
    case = reference['models'][index]
    dtype = torch.float32 if index == 0 else torch.bfloat16
    config = dict(vision_config=case['vision_config'], text_config=case['text_config'])
    vision = Vision(config, dtype, 'cpu', manual_cast)
    vision.load_state_dict(case['vision_state'], strict=True)
    visual = vision(case['pixels'], (4,6), {})
    torch.testing.assert_close(visual, case['visual'], atol=.008 if index else 2e-6, rtol=.02 if index else 1e-5)
    language = Language(config, dtype, 'cpu', manual_cast)
    language.load_state_dict(case['language_state'], strict=True)
    cache = language.model.init_kv_cache(1, 40, 'cpu', dtype)
    for step, keep in zip(case['steps'], (8,14,15)):
        logits = language(step['ids'], step['positions'], cache, step['mtp'], {})
        torch.testing.assert_close(logits, step['logits'], atol=.008 if index else 2e-6, rtol=.02 if index else 1e-5)
        cache = [(k,v,keep) for k,v,_ in cache]


def test_sampler_preserves_reference_and_host_rng(reference):
    for case in reference['sampling']:
        before = torch.random.get_rng_state().clone()
        actual = sample(case['logits'], case['history'], torch.Generator().manual_seed(19), case['mode'])
        assert actual == case['values'].tolist()
        assert torch.equal(before, torch.random.get_rng_state())


def test_masks_preserve_duplicate_exclusion_and_bidirectional_window():
    for qlen,past in ((14,0),(7,18)):
        mask = inference_mask(qlen,past,True,'cpu',torch.float32)
        assert torch.isneginf(mask[-6:,-7]).all()
        assert (mask[-6:,-6:] == 0).all()
        assert (mask[0,:past+1] == 0).all()
    assert inference_mask(1,18,False,'cpu',torch.float32) is None


def test_pattern_modes_empty_points_and_hybrid_fallback():
    t = TOKENS
    assert pattern([t['box_start'],t['none'],0,0,0,0],'hybrid') == ('box',[t['box_start'],t['none'],t['box_end']])
    point = [t['box_start'],t['coord_start']+10,t['coord_start']+20,t['box_end'],0,0]
    assert pattern(point,'hybrid') == ('box',point[:4])
    malformed = [t['box_start'],t['coord_start'],0,0,0,0]
    assert pattern(malformed,'hybrid') == ('error',malformed[:2])
    assert pattern(malformed,'fast') == ('box',malformed)


def test_complete_meta_inventory_matches_pinned_checkpoint_headers(reference):
    source = Path(os.environ['LOCATEANYTHING_REFERENCE'])
    expected = expected_shapes(reference['config'])
    assert len(expected) == len(inspect_shards(source, expected)) == 770


def test_generation_cache_rollback_modes_cancellation_and_retry():
    from comfy.locateanything.sampling import generate
    from comfy import model_management as mm
    class Scripted:
        def __init__(self):
            self.model = types.SimpleNamespace(init_kv_cache=lambda *args: [[]])
            self.calls = []

        def __call__(self, ids, positions, cache, mtp, options, visual=None):
            mm.throw_exception_if_processing_interrupted()
            self.calls.append((ids.clone(),positions.clone(),cache[0][2] if cache[0] else 0,mtp))
            cache[0] = (torch.empty(0),torch.empty(0),ids.shape[1])
            tokens = [TOKENS['box_start'],151687,151697,151757,151767,TOKENS['box_end']]
            selected = tokens if mtp else [tokens[0]]
            if len(self.calls) > 1:
                selected = [TOKENS['end']]*(6 if mtp else 1)
            scores = torch.full((1,len(selected),152681),-100.)
            for i,token in enumerate(selected):
                scores[0,i,token] = 100.
            return scores
    tokenizer = types.SimpleNamespace(decode=lambda ids,**kw: str(ids))
    for mode in ('hybrid','fast','slow'):
        model = Scripted()
        inputs = (model,torch.tensor([[1,2,3]]),torch.empty(1,4),tokenizer,20,7401,mode,{},lambda count: None)
        expected = generate(*inputs)
        assert len(model.calls) == 2
        assert model.calls[1][2] == 3
        if mode != 'slow':
            assert model.calls[0][0][0].tolist() == [1,2,3,3]+[TOKENS['mask']]*5
            assert model.calls[0][1][0].tolist() == [0,1,2,2,3,4,5,6,7]
        with patch.object(mm,'throw_exception_if_processing_interrupted',side_effect=mm.InterruptProcessingException):
            with pytest.raises(mm.InterruptProcessingException):
                generate(*inputs)
        retry = Scripted()
        assert generate(retry,*inputs[1:]) == expected


def test_native_failure_cleans_prefetch_without_publishing_output():
    from comfy.locateanything import runtime
    patcher = types.SimpleNamespace(load_device=torch.device('cpu'))
    component = types.SimpleNamespace(config={'vision_config':{'hidden_size':32}},processor={},patcher=patcher)
    with patch.object(runtime,'prepare_image',return_value=(torch.zeros(4,3,2,2),(2,2))), \
         patch.object(runtime,'prepare_prompt',return_value=torch.ones(1,3,dtype=torch.long)), \
         patch.object(runtime.mm,'load_models_gpu',side_effect=runtime.mm.InterruptProcessingException), \
         patch.object(runtime.comfy.model_prefetch,'cleanup_prefetch_queues') as cleanup:
        with pytest.raises(runtime.mm.InterruptProcessingException):
            runtime.locate(component,types.SimpleNamespace(**{**vars(component),'tokenizer':None}),Image.new('RGB',(4,4)), 'test','hybrid',10,0)
    cleanup.assert_called_once()
