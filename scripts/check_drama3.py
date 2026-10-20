"""Submit native Drama 3 acceptance cases to an isolated local ComfyUI queue."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    with urlopen(Request(base + path, data=body, headers={'Content-Type': 'application/json'}), timeout=60) as response:
        return json.load(response)


def run(base, name, prompt, output):
    folder = output / name
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'prompt.json').write_text(json.dumps(prompt, indent=2))
    submitted = request(base, '/prompt', {'prompt': prompt, 'client_id': 'drama3-native-acceptance'})
    (folder / 'submission.json').write_text(json.dumps(submitted, indent=2))
    prompt_id = submitted['prompt_id']
    started = time.monotonic()
    print(name, prompt_id, flush=True)
    while True:
        result = request(base, '/history/' + prompt_id).get(prompt_id)
        if result and result.get('status', {}).get('completed'):
            (folder / 'history.json').write_text(json.dumps(result, indent=2))
            if result['status']['status_str'] != 'success':
                raise RuntimeError(f'{name} failed; inspect {folder}/history.json')
            print(name, 'completed', round(time.monotonic() - started, 2), flush=True)
            return result
        if result and result.get('status', {}).get('status_str') == 'error':
            (folder / 'history.json').write_text(json.dumps(result, indent=2))
            raise RuntimeError(f'{name} failed; inspect {folder}/history.json')
        time.sleep(2)


def graph(model, codec, text, seed, prefix):
    return {'load': {'class_type': 'Drama3Loader', 'inputs': {'model_name': model, 'codec_name': codec}},
            'generate': {'class_type': 'Drama3Generate', 'inputs': {'model': ['load', 0], 'text': text, 'seed': seed,
                'temperature': .7, 'top_p': .7, 'top_k': 30, 'max_frames': 1024, 'single_chunk': False}},
            'audio': {'class_type': 'SaveAudio', 'inputs': {'audio': ['generate', 0], 'filename_prefix': prefix}},
            'history': {'class_type': 'Drama3SaveHistory', 'inputs': {'history': ['generate', 1], 'filename_prefix': prefix}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8199')
    parser.add_argument('--model', default='drama3_nvfp4.safetensors')
    parser.add_argument('--codec', default='fish_dac_drama3.safetensors')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['plain', 'warm', 'revision', 'delivery', 'dialogue', 'long', 'multilingual', 'clone', 'reload'], default='plain')
    parser.add_argument('--history-name')
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Use the isolated local ComfyUI endpoint')
    args.output.mkdir(parents=True, exist_ok=True)
    text = 'The quarterly report is due on Wednesday. Please finish it by noon.'
    if args.case == 'dialogue':
        text = '<|speaker:0|> Are we ready to begin?\n<|speaker:1|> [laughs] Yes, let us get started.'
    elif args.case == 'long':
        text = ('The station was quiet when we arrived. A small light glowed above the doorway, and someone inside was playing a familiar tune. We stopped to listen before knocking. ' * 4).strip()
    elif args.case == 'multilingual':
        text = 'Hello, welcome to the studio. 你好，欢迎来到录音室。'
    prompt = graph(args.model, args.codec, text, 81, 'drama3-acceptance/' + args.case)
    if args.case == 'warm':
        prompt['generate']['inputs']['seed'] = 82
    if args.case == 'clone':
        prompt['original'] = graph(args.model, args.codec, 'Welcome to the studio. This is a reference voice.', 81, 'unused')['generate']
        prompt['reference_audio'] = {'class_type': 'SaveAudio', 'inputs': {'audio': ['original', 0], 'filename_prefix': 'drama3-acceptance/clone-source'}}
        prompt['reference'] = {'class_type': 'Drama3EncodeReference', 'inputs': {'model': ['load', 0], 'audio': ['original', 0], 'transcript': 'Welcome to the studio. This is a reference voice.', 'speaker': 0}}
        prompt['generate']['inputs']['references.reference_0'] = ['reference', 0]
    if args.case == 'reload':
        if not args.history_name:
            raise ValueError('--history-name is required for reload')
        prompt['previous'] = {'class_type': 'Drama3LoadHistory', 'inputs': {'history_name': args.history_name}}
        prompt['generate']['inputs']['history'] = ['previous', 0]
        prompt['generate']['inputs']['text'] = text.replace('Wednesday', 'Thursday')
    if args.case in ('revision', 'delivery'):
        original = graph(args.model, args.codec, text, 81, 'drama3-acceptance/original')
        prompt['original'] = original['generate']
        prompt['generate']['inputs']['history'] = ['original', 1]
        prompt['generate']['inputs']['seed'] = 82
        prompt['generate']['inputs']['text'] = text.replace('Wednesday', 'Thursday') if args.case == 'revision' else '[whisper] ' + text
    run(args.base_url.rstrip('/'), args.case, prompt, args.output)


if __name__ == '__main__':
    main()
