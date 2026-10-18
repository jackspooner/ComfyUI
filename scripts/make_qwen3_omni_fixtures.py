"""Create deterministic local Omni acceptance fixtures from synthetic images/sounds and licensed audio."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import resample_poly
import soundfile as sf
import torch

import comfy.utils


def make_fixtures(destination, audio_source):
    destination.mkdir(parents=True, exist_ok=False)
    images = {}
    for name, color, shape in [('red_square', 'red', 'square'), ('blue_circle', 'blue', 'circle'),
                                ('green_triangle', 'green', 'triangle')]:
        image = Image.new('RGB', (256, 256), 'white')
        draw = ImageDraw.Draw(image)
        if shape == 'square':
            draw.rectangle((56, 56, 200, 200), fill=color)
        elif shape == 'circle':
            draw.ellipse((48, 48, 208, 208), fill=color)
        else:
            draw.polygon([(128, 32), (224, 224), (32, 224)], fill=color)
        path = destination / (name + '.png')
        image.save(path)
        images[name] = str(path)
    image = Image.new('RGB', (384, 128), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 108, 108), fill='red')
    draw.ellipse((264, 12, 360, 108), fill='blue')
    path = destination / 'non_square.png'
    image.save(path)
    images['non_square'] = str(path)
    image = Image.new('RGB', (384, 128), 'white')
    ImageDraw.Draw(image).text((24, 28), 'QWEN42', font=ImageFont.load_default(size=54), fill='black')
    path = destination / 'sign.png'
    image.save(path)
    images['sign'] = str(path)
    audio = {}
    for name, filename in [('speech', 'speech.wav'), ('music', 'music.wav'), ('birds', 'environment.wav')]:
        values, rate = sf.read(audio_source / filename, dtype='float32', always_2d=True)
        values = values[:rate * 8].mean(axis=1)
        divisor = np.gcd(rate, 16000)
        values = resample_poly(values, 16000 // divisor, rate // divisor).astype(np.float32)
        path = destination / (name + '.wav')
        sf.write(path, values, 16000, subtype='PCM_16')
        audio[name] = str(path)
    t = np.arange(32000, dtype=np.float32) / 16000
    for name, values in [('silence', np.zeros_like(t)), ('tone', .15 * np.sin(2 * np.pi * 440 * t)),
                         ('clicks', np.where((np.arange(len(t)) % 8000) < 64, .3, 0).astype(np.float32))]:
        path = destination / (name + '.wav')
        sf.write(path, values, 16000, subtype='PCM_16')
        audio[name] = str(path)
    cases = []
    for name, prompt, answer in [
        ('arithmetic', 'What is 7 plus 5? Reply only with the integer.', '12'),
        ('capital', 'What is the capital of France? Reply only with its name.', 'paris'),
        ('copy', 'Reply exactly with OMEGA.', 'omega'),
        ('unicode', 'Copy exactly this word: café', 'café'),
        ('extract', 'Return only the code in CODE: BLUE42.', 'blue42'),
        ('compare', 'Which integer is larger, 9 or 3? Reply only with that integer.', '9')]:
        cases.append({'id': 'text_' + name, 'category': 'text', 'prompt': prompt, 'assert_all': [[answer]]})
    for name, prompt, answer in [
        ('speech', 'Classify the main sound as speech, music, birds, silence, or a tone. Reply with the class.', ['speech']),
        ('music', 'Describe the main instrument you hear in one sentence.', ['trumpet']),
        ('birds', 'Describe the main sound you hear in one sentence.', ['bird']),
        ('silence', 'Describe whether this audio contains a clearly audible sound.', ['silence', 'silent', 'quiet', 'no audible', 'no sound']),
        ('tone', 'Describe this sound briefly.', ['tone', 'beep']),
        ('clicks', 'Describe the repeated sound briefly.', ['click', 'pop', 'tap', 'pulse'])]:
        cases.append({'id': 'audio_' + name, 'category': 'audio', 'prompt': prompt, 'audio': audio[name], 'assert_all': [answer]})
    for name, selected, prompt, answers in [
        ('red', ['red_square'], 'What color is the central shape? Reply with the color.', [['red']]),
        ('blue', ['blue_circle'], 'What color is the central shape? Reply with the color.', [['blue']]),
        ('triangle', ['green_triangle'], 'What shape is shown? Reply with the shape.', [['triangle']]),
        ('sign', ['sign'], 'Read the printed text exactly.', [['qwen42']]),
        ('wide', ['non_square'], 'Name the two colored objects from left to right.', [['red'], ['blue']]),
        ('batch', ['red_square', 'blue_circle'], 'Name the color in the first image and then the second.', [['red'], ['blue']])]:
        cases.append({'id': 'image_' + name, 'category': 'image', 'prompt': prompt,
                      'images': [images[item] for item in selected], 'assert_all': answers})
    prompt = 'Describe the visible image color or text and classify the main audio sound. Mention only modalities actually supplied.'
    for name, image_names, audio_name, answers in [
        ('baseline', ['red_square'], 'speech', [['red'], ['speech', 'voice', 'speaking', 'narrat']]),
        ('replace_image', ['blue_circle'], 'speech', [['blue'], ['speech', 'voice', 'speaking', 'narrat']]),
        ('replace_audio', ['red_square'], 'birds', [['red'], ['bird']]),
        ('remove_image', [], 'speech', [['speech', 'voice', 'speaking', 'narrat']]),
        ('remove_audio', ['red_square'], None, [['red']]),
        ('batch', ['sign', 'blue_circle'], 'tone', [['qwen42'], ['blue'], ['tone', 'beep']])]:
        case = {'id': 'combined_' + name, 'category': 'combined', 'prompt': prompt,
                'images': [images[item] for item in image_names], 'assert_all': answers}
        if audio_name:
            case['audio'] = audio[audio_name]
        cases.append(case)
    for case in cases:
        case['max_new_tokens'] = 96
        if case.get('images'):
            tensors = [torch.from_numpy(np.array(Image.open(path).convert('RGB'))).float().unsqueeze(0) / 255
                       for path in case['images']]
            first = tensors[0]
            for index in range(1, len(tensors)):
                if tensors[index].shape[1:] != first.shape[1:]:
                    tensors[index] = comfy.utils.common_upscale(tensors[index].movedim(-1, 1), first.shape[2], first.shape[1],
                                                               'bilinear', 'center').movedim(1, -1)
            path = destination / (case['id'] + '-images.npy')
            np.save(path, torch.cat(tensors).numpy(), allow_pickle=False)
            case['image_tensor'] = str(path)
    (destination / 'cases.json').write_text(json.dumps(cases, indent=2) + '\n')
    attribution = ['5703-47212-0000.txt', 'sorohanro_-_solo-trumpet-06.txt',
                   '456440__inspectorj__bird-whistling-robin-single-13.txt']
    (destination / 'credits.txt').write_text('\n\n'.join((audio_source / name).read_text() for name in attribution))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in destination.iterdir()}
    (destination / 'manifest.json').write_text(json.dumps({'fixture_version': 2, 'files': hashes,
        'image_input': 'Frozen FP32 BHWC IMAGE tensors; mixed source dimensions assembled with ComfyUI ImageBatch bilinear/center behavior.',
        'batch_utility_sha256': hashlib.sha256(Path(comfy.utils.__file__).read_bytes()).hexdigest(),
        'audio_source': str(audio_source), 'credits': 'Speech: LibriSpeech/SLR12 CC-BY 4.0; trumpet: sorohanro_; birds: InspectorJ. Preserve original fixture attribution sidecars.',
        'synthetic_assets': 'Images, silence, tone and clicks created locally by this script.'}, indent=2) + '\n')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audio-fixtures', type=Path, required=True)
    args = parser.parse_args()
    cases = make_fixtures(args.output.resolve(), args.audio_fixtures.resolve())
    print(f'Created {len(cases)} cases in {args.output}')


if __name__ == '__main__':
    main()
