"""Local instruction templates and duration estimates adapted from Tencent AuK PE (MIT)."""
import json
import math
from pathlib import Path
import re


with Path(__file__).with_name('auk_prompt_templates.json').open() as source:
    TASKS = json.load(source)['tasks']

SYSTEM_PROMPT = '''Prepare one AuK audio instruction from the user's request. Return only JSON:
{"task_type":"...", "operation_subtype":null, "language":"en", "text_language":"en",
 "params_extracted":{...}, "timing_ratio":1.0}
Choose the closest task from the catalogue below and extract its required parameters.
Use instruct_tts without reference audio; use zero_shot_tts for cloning a reference voice to say new text.
Preserve supplied speech, replacement words, anchors and punctuation exactly, as substrings of the request.
The supplied reference transcript is context, not new speech to synthesize. Do not normalize or translate it.
For style_desc and timbre_desc, describe only the requested voice/style. Do not invent speech.
Use language en or zh. For synthesis, text_language describes the speech content.
For synthesis only, timing_ratio adjusts ordinary speaking duration for explicitly requested speed/pauses;
use 1.0 unless timing changes are requested. Do not adjust timing for pitch, age or accent alone.
Select one supported subtype where required. Return no Markdown, commentary or reasoning.
'''


def parse_preparation(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError('AuK Prompt Helper returned invalid JSON. Retry with a compatible local text-generation CLIP, or supply the instruction and duration directly.') from exc
    if not isinstance(result, dict) or not isinstance(result.get('task_type'), str) or result['task_type'] not in TASKS or not isinstance(result.get('params_extracted'), dict):
        raise ValueError('AuK Prompt Helper must return a supported task_type and a params_extracted object. Supply instruction and duration directly to bypass it.')
    return result


def text_weight(text, language='en'):
    scripts = ['zh' if '\u3400' <= c <= '\u9fff' else 'en' if c.isascii() and c.isalpha() else None for c in text]
    next_script, following = None, [None] * len(text)
    for i in range(len(text) - 1, -1, -1):
        next_script = scripts[i] or next_script
        following[i] = next_script
    previous, total = None, 0.0
    for i, char in enumerate(text):
        current = scripts[i] or previous or following[i] or language
        previous = scripts[i] or previous
        total += len(char.encode('utf-8')) * (0.0803 if current == 'zh' else 0.0656)
    return total


def spoken_duration(text):
    return len(re.findall(r'[\u3400-\u9fff]', text)) * 0.21 + len(re.findall(r'[A-Za-z]+(?:[\'-][A-Za-z]+)*', text)) * 0.30


def number(value, name):
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'AuK Prompt Helper returned an invalid {name}. Supply an explicit duration or bypass the helper.') from exc


def prepare(result, request, reference_seconds=None, transcript='', explicit_duration=None):
    task = result['task_type']
    spec = TASKS[task]
    params = dict(result['params_extracted'])
    subtype = result.get('operation_subtype')
    if subtype is not None and not isinstance(subtype, str):
        raise ValueError('AuK Prompt Helper returned an invalid operation subtype. Retry or supply a direct instruction.')
    language = result.get('text_language') if task in ('instruct_tts', 'zero_shot_tts') else result.get('language')
    language = language or 'en'
    if language not in ('en', 'zh'):
        raise ValueError('AuK Prompt Helper language must be en or zh.')
    if spec['needs_audio'] and reference_seconds is None:
        raise ValueError(f'AuK task {task} requires reference audio. Supply audio or use a synthesis instruction.')
    for name in ('text', 'new', 'orig', 'anchor', 'target'):
        if name in params and (not isinstance(params[name], str) or params[name] not in request):
            raise ValueError(f'AuK Prompt Helper changed the supplied {name}. Retry or bypass the helper; speech content must be copied exactly.')
    for param in spec.get('params', []):
        name = param['name']
        if (param.get('required') or subtype in param.get('required_for', [])) and name not in params:
            raise ValueError(f'AuK Prompt Helper omitted {name} for {task}. Retry or supply a direct instruction.')
        if name in params:
            if param['type'] == 'string' and not isinstance(params[name], str):
                raise ValueError(f'AuK Prompt Helper returned invalid {name}; expected text. Retry or supply a direct instruction.')
            if param['type'] in ('int', 'float') and not math.isfinite(number(params[name], name)):
                raise ValueError(f'AuK Prompt Helper returned invalid {name}; expected a finite number. Retry or supply a direct instruction.')
    templates = spec['templates']
    selected = templates.get(language) or templates.get('zh') or templates.get('en')
    if isinstance(selected, dict):
        selected = selected.get(subtype)
        if selected is None:
            # Some upstream tasks have English wording for only one subtype.
            selected = templates.get('zh', {}).get(subtype)
    if selected is None:
        raise ValueError(f'AuK Prompt Helper returned an unsupported subtype for {task}.')
    if 'emotion' in params:
        labels = next(p['labels'] for p in spec['params'] if p['name'] == 'emotion')
        if params['emotion'] in labels:
            params['emotion'] = labels[params['emotion']][language]
    if 'n' in params:
        params['n_zh'] = str(params['n'])
    try:
        instruction = selected.format(**params)
    except (KeyError, ValueError) as exc:
        raise ValueError(f'AuK Prompt Helper returned incomplete parameters for {task}. Retry or bypass the helper.') from exc
    if explicit_duration is not None:
        duration = number(explicit_duration, 'explicit duration')
    elif task in ('instruct_tts', 'zero_shot_tts'):
        text = params['text']
        speed = 0.3 if len(text.encode('utf-8')) < 10 else 1.0
        weight = text_weight(text, language)
        if task == 'zero_shot_tts' and transcript and text_weight(transcript) > 0:
            frames = int(reference_seconds * 24000 / 256)
            duration = int(frames * weight / text_weight(transcript) / speed) * 256 / 24000
        else:
            duration = int(weight * 24000 / 256 / speed) * 256 / 24000
        if task == 'instruct_tts':
            duration *= number(result.get('timing_ratio', 1.0), 'timing ratio')
    else:
        rule = spec['duration']
        duration = reference_seconds
        if rule['strategy'] == 'speed_scaled':
            speed = number(params['speed_multiplier'], 'speed multiplier')
            if speed <= 0:
                raise ValueError('AuK Prompt Helper returned a non-positive speed multiplier.')
            duration /= speed
        elif rule['strategy'] == 'content_scaled':
            add, delete = ('new', 'orig') if task == 'vocal_edit' else {
                'insert_before': ('text', None), 'insert_after': ('text', None), 'delete': (None, 'target'),
                'delete_before': (None, 'target'), 'delete_after': (None, 'target'), 'replace': ('new', 'orig')}[subtype]
            original = spoken_duration(transcript)
            if original > 0:
                duration *= max(.05, original + spoken_duration(params.get(add, '')) - spoken_duration(params.get(delete, ''))) / original
            elif add and delete and spoken_duration(params[delete]) > 0:
                duration *= spoken_duration(params[add]) / spoken_duration(params[delete])
        elif 'emotion_multipliers' in rule:
            duration *= rule['emotion_multipliers'].get(result['params_extracted']['emotion'], rule['default_emotion_multiplier'])
        elif 'nonverbal_adjustments' in rule:
            operation = 'delete' if subtype == 'delete' else 'add'
            adjustment = rule['default_nonverbal_adjustment'][operation]
            for entry in rule['nonverbal_adjustments']:
                if any(word in params['sound'].lower() for word in entry['keywords']):
                    adjustment = entry[operation]
                    break
            duration = max(.1, duration + adjustment)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('AuK Prompt Helper produced an invalid duration. Supply a positive explicit duration or bypass the helper.')
    return instruction, duration
