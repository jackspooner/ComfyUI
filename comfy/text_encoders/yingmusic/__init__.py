"""Singer Plus Chinese/English phonemes. Adapted from YingMusic and Amphion (MIT)."""

import json

import numpy as np
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

from .english import english_to_ipa
from .mandarin import chinese_to_ipa
from .text_tokenizers import TextTokenizer


class ChineseResources:
    def __init__(self, path):
        self.word_pinyin_dict = dict(line.split("\t") for line in (path / "chinese_lexicon.txt").read_text(encoding="utf-8").splitlines())
        self.pinyin_2_bopomofo_dict = dict(line.split("\t") for line in (path / "pinyin_2_bpmf.txt").read_text(encoding="utf-8").splitlines())
        bert = path / "g2p_chinese_model"
        self.poly_dict = set((bert / "polychar.txt").read_text(encoding="utf-8").splitlines())
        self.g2pw_poly_predict = PolyphonePredictor(bert)


class PolyphonePredictor:
    def __init__(self, path):
        self.vocab = {token: index for index, token in enumerate((path / "vocab.txt").read_text(encoding="utf-8").splitlines())}
        self.pronunciations = json.loads((path / "polydict.json").read_text(encoding="utf-8"))
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = InferenceSession(str(path / "poly_bert_model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        self.session.disable_fallback()

    def predict_process(self, text_and_index):
        text, index = text_and_index
        tokens = [self.vocab["[CLS]"]] + [self.vocab.get(char, self.vocab["[UNK]"]) for char in text]
        prediction = self.session.run(["outputs"], {"input_ids": np.array([tokens], dtype=np.int32)})[0]
        return [self.pronunciations[str(int(prediction[0, index].argmax()) + 1)]]


def segments(text):
    current, language = "", "other"
    result = []
    for char in text:
        kind = "zh" if "\u4e00" <= char <= "\u9fa5" else "en" if char.isascii() and char.isalpha() else "other"
        if language == "other":
            current += char
            language = kind
        elif kind in (language, "other"):
            current += char
        else:
            result.append((current, language))
            current, language = char, kind
    if current:
        result.append((current, language))
    return result


class YingMusicTokenizer:
    def __init__(self, path):
        self.path = path
        self.vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))["vocab"]
        self.separator = len(self.vocab) + 2
        self.english = TextTokenizer(language="en-us")
        self.chinese = None

    def encode(self, text):
        parts = segments(text)
        tokens = []
        for index, (part, language) in enumerate(parts):
            if language == "zh":
                if self.chinese is None:
                    self.chinese = ChineseResources(self.path)
                phones = chinese_to_ipa(part, text, self.chinese)
            elif language == "en":
                phones = english_to_ipa(part, self.english)
            else:
                phones = part
            ids = [int(self.vocab[phone]) + 1 for phone in phones.split("\t")[0].split("|") if phone in self.vocab]
            if language == "en" and index == len(parts) - 1 and phones.endswith("_"):
                ids = ids[:-1]
            tokens.extend(ids)
        return tokens

    def align(self, reference, target, reference_frames, total_frames):
        tokens = [0] * total_frames
        cursor = 0
        for start, text in ((0, reference), (reference_frames, target)):
            text = text.replace("|", " ")
            for punctuation in "，。！？、；：,.!?;:":
                text = text.replace(punctuation, ",").strip("，。！？、；：,.!?;: ")
            ids = self.encode(text) + [self.separator]
            start = max(start, cursor)
            if start + len(ids) > total_frames:
                raise ValueError("YingMusic lyrics exceed the reference audio duration; shorten the lyrics or use longer audio.")
            tokens[start:start + len(ids)] = ids
            cursor = start + len(ids)
        return tokens
