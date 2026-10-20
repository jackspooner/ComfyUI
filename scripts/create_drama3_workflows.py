"""Write editable Drama 3 examples and matching API templates to an explicit library."""
import argparse
import json
from pathlib import Path


def make(kind, model, codec):
    nodes, links, groups, api = [], [], [], {}
    def node(identifier, name, title, x, y, size, widgets, outputs):
        n = dict(id=identifier, type=name, title=title, pos=[x, y], size=size, flags={}, order=len(nodes), mode=0,
                 inputs=[], outputs=[dict(name=t, type=t, links=[], slot_index=i) for i, t in enumerate(outputs)],
                 properties={'Node name for S&R': name}, widgets_values=widgets)
        nodes.append(n)
        api[str(identifier)] = dict(class_type=name, inputs={})
        return n
    def connect(source, slot, target, name, value_type, widget=False):
        identifier = len(links) + 1
        entry = dict(name=name, type=value_type, link=identifier)
        if widget:
            entry['widget'] = {'name': name}
        target['inputs'].append(entry)
        source['outputs'][slot]['links'].append(identifier)
        links.append([identifier, source['id'], slot, target['id'], len(target['inputs']) - 1, value_type])
        api[str(target['id'])]['inputs'][name] = [str(source['id']), slot]
    def group(title, color, contents):
        left = min(n['pos'][0] for n in contents) - 30
        top = min(n['pos'][1] for n in contents) - 80
        right = max(n['pos'][0] + n['size'][0] for n in contents) + 30
        bottom = max(n['pos'][1] + n['size'][1] for n in contents) + 30
        groups.append(dict(id=len(groups)+1, title=f'{len(groups)+1}. {title}', bounding=[left, top, right-left, bottom-top], color=color, font_size=24, flags={}))
    load = node(1, 'Drama3Loader', 'Drama 3 + matching codec', 0, 80, [330, 150], [model, codec], ['DRAMA3_MODEL'])
    api['1']['inputs'] = dict(model_name=model, codec_name=codec)
    text = 'The quarterly report is due on Wednesday. Please finish it by noon.'
    if kind == 'Dialogue':
        text = '<|speaker:0|> Are we ready to begin?\n<|speaker:1|> Yes, let us get started.'
    if kind == 'Reload':
        text = text.replace('Wednesday', 'Thursday')
    prompt = node(2, 'PrimitiveStringMultiline', 'Script / dialogue', 500, 80, [330, 220], [text], ['STRING'])
    api['2']['inputs'] = dict(value=text)
    seed = node(3, 'PrimitiveInt', 'Seed', 500, 470, [330, 100], [81, 'fixed'], ['INT'])
    api['3']['inputs'] = dict(value=81)
    frames = node(4, 'PrimitiveInt', 'Maximum frames per chunk', 500, 610, [330, 100], [1024, 'fixed'], ['INT'])
    api['4']['inputs'] = dict(value=1024)
    generate = node(5, 'Drama3Generate', 'Generate original take', 1000, 80, [380, 490], ['', 81, 'fixed', .7, .7, 30, 1024, False], ['AUDIO', 'DRAMA3_HISTORY'])
    api['5']['inputs'] = dict(temperature=.7, top_p=.7, top_k=30, single_chunk=False)
    connect(load, 0, generate, 'model', 'DRAMA3_MODEL')
    connect(prompt, 0, generate, 'text', 'STRING', True)
    connect(seed, 0, generate, 'seed', 'INT', True)
    connect(frames, 0, generate, 'max_frames', 'INT', True)
    group('Model Loading', '#665191', [load])
    group('Inputs', '#336b87', [prompt])
    group('Configuration', '#a67c38', [seed, frames])
    group('Speech Generation', '#347c65', [generate])
    result = generate
    output_x = 1600
    if kind == 'Revision':
        revised_text = text.replace('Wednesday', 'Thursday')
        revised = node(6, 'PrimitiveStringMultiline', 'Revised script / delivery cues', 500, 900, [330, 220], [revised_text], ['STRING'])
        api['6']['inputs'] = dict(value=revised_text)
        result = node(7, 'Drama3Generate', 'Revise using original take', 1500, 80, [380, 490], ['', 82, 'fixed', .7, .7, 30, 1024, False], ['AUDIO', 'DRAMA3_HISTORY'])
        api['7']['inputs'] = dict(seed=82, temperature=.7, top_p=.7, top_k=30, single_chunk=False)
        connect(load, 0, result, 'model', 'DRAMA3_MODEL')
        connect(revised, 0, result, 'text', 'STRING', True)
        connect(frames, 0, result, 'max_frames', 'INT', True)
        connect(generate, 1, result, 'history', 'DRAMA3_HISTORY')
        group('Revision Input', '#336b87', [revised])
        group('Revised Speech', '#347c65', [result])
        output_x = 2100
    elif kind == 'Reload':
        saved = node(6, 'Drama3LoadHistory', 'Choose a saved take history', 0, 440, [330, 100], [''], ['DRAMA3_HISTORY'])
        api['6']['inputs'] = dict(history_name='')
        connect(saved, 0, generate, 'history', 'DRAMA3_HISTORY')
        generate['title'] = 'Revise saved take'
        group('Saved Take Input', '#336b87', [saved])
    audio = node(8, 'SaveAudio', 'Save generated speech', output_x, 80, [330, 120], ['drama3/' + kind.lower()], [])
    history = node(9, 'Drama3SaveHistory', 'Save tokens and revision history', output_x, 320, [330, 100], ['drama3/' + kind.lower()], [])
    api['8']['inputs'] = dict(filename_prefix='drama3/' + kind.lower())
    api['9']['inputs'] = dict(filename_prefix='drama3/' + kind.lower())
    connect(result, 0, audio, 'audio', 'AUDIO')
    connect(result, 1, history, 'history', 'DRAMA3_HISTORY')
    group('Outputs', '#8c4c4c', [audio, history])
    return dict(last_node_id=max(n['id'] for n in nodes), last_link_id=len(links), nodes=nodes, links=links,
                groups=groups, config={}, extra={'ds': {'scale': .65, 'offset': [80, 80]}}, version=.4), api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--model', default='drama3_nvfp4.safetensors')
    p.add_argument('--codec', default='fish_dac_drama3.safetensors')
    args = p.parse_args()
    args.library.mkdir(parents=True, exist_ok=True)
    for kind in ('Speech', 'Dialogue', 'Revision', 'Reload'):
        workflow, api = make(kind, args.model, args.codec)
        name = f'Drama 3 - {kind}'
        for suffix, data in [('.json', workflow), ('.api.json', api)]:
            path = args.library / (name + suffix)
            with path.open('x') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(path)


if __name__ == '__main__':
    main()
