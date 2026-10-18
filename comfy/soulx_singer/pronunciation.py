"""Offline word pronunciation; G2p-en predictor adapted from Kyubyong Park/Jongseok Kim (Apache-2.0)."""

import json
import re

import numpy as np
from safetensors import safe_open

try:
    from pypinyin import lazy_pinyin, Style
except ImportError:
    lazy_pinyin = None


class Pronunciation:
    def __init__(self, root):
        self.lexicon = {}
        for line in (root / "cmudict.dict").read_text().splitlines():
            fields = line.split("#", 1)[0].split()
            if len(fields) > 1:
                self.lexicon.setdefault(re.sub(r"\(\d+\)$", "", fields[0]), fields[1:])
        with safe_open(root / "pronunciation.safetensors", framework="numpy") as saved:
            self.weights = {key: saved.get_tensor(key) for key in saved.keys()}
            self.phonemes = json.loads(saved.metadata()["phonemes"])
        self.graphemes = {c: i for i, c in enumerate(["<pad>", "<unk>", "</s>", *"abcdefghijklmnopqrstuvwxyz"])}

    def cell(self, x, h, prefix):
        w = self.weights
        ih = x @ w[prefix + "_w_ih"].T + w[prefix + "_b_ih"]
        hh = h @ w[prefix + "_w_hh"].T + w[prefix + "_b_hh"]
        size = h.shape[-1]
        rz = 1 / (1 + np.exp(-(ih[:, :size * 2] + hh[:, :size * 2])))
        r, z = np.split(rz, 2, axis=-1)
        n = np.tanh(ih[:, size * 2:] + r * hh[:, size * 2:])
        return (1 - z) * n + z * h

    def __call__(self, word):
        if not word.isascii():
            if lazy_pinyin is None:
                raise ValueError("Mandarin needs optional pypinyin or explicit zh_/yue_ pronunciation overrides.")
            return ["zh_" + ph for ph in lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)]
        word = word.lower()
        if word in self.lexicon:
            return self.lexicon[word]
        w = self.weights
        h = np.zeros((1, w["enc_w_hh"].shape[-1]), np.float32)
        for char in [*word, "</s>"]:
            h = self.cell(w["enc_emb"][[self.graphemes.get(char, 1)]], h, "enc")
        token, result = 2, []
        for _ in range(20):
            h = self.cell(w["dec_emb"][[token]], h, "dec")
            token = int((h @ w["fc_w"].T + w["fc_b"]).argmax())
            if token == 3:
                break
            result.append(self.phonemes[token])
        return result
