# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List

import cn2an
import jieba
from pypinyin import BOPOMOFO, lazy_pinyin

BLANK_LEVEL = 0

# List of (Latin alphabet, bopomofo) pairs:
_latin_to_bopomofo = [
    (re.compile("%s" % x[0], re.IGNORECASE), x[1])
    for x in [
        ("a", "ㄟˉ"),
        ("b", "ㄅㄧˋ"),
        ("c", "ㄙㄧˉ"),
        ("d", "ㄉㄧˋ"),
        ("e", "ㄧˋ"),
        ("f", "ㄝˊㄈㄨˋ"),
        ("g", "ㄐㄧˋ"),
        ("h", "ㄝˇㄑㄩˋ"),
        ("i", "ㄞˋ"),
        ("j", "ㄐㄟˋ"),
        ("k", "ㄎㄟˋ"),
        ("l", "ㄝˊㄛˋ"),
        ("m", "ㄝˊㄇㄨˋ"),
        ("n", "ㄣˉ"),
        ("o", "ㄡˉ"),
        ("p", "ㄆㄧˉ"),
        ("q", "ㄎㄧㄡˉ"),
        ("r", "ㄚˋ"),
        ("s", "ㄝˊㄙˋ"),
        ("t", "ㄊㄧˋ"),
        ("u", "ㄧㄡˉ"),
        ("v", "ㄨㄧˉ"),
        ("w", "ㄉㄚˋㄅㄨˋㄌㄧㄡˋ"),
        ("x", "ㄝˉㄎㄨˋㄙˋ"),
        ("y", "ㄨㄞˋ"),
        ("z", "ㄗㄟˋ"),
    ]
]

# List of (bopomofo, ipa) pairs:
_bopomofo_to_ipa = [
    (re.compile("%s" % x[0]), x[1])
    for x in [
        ("ㄅㄛ", "p⁼wo"),
        ("ㄆㄛ", "pʰwo"),
        ("ㄇㄛ", "mwo"),
        ("ㄈㄛ", "fwo"),
        ("ㄧㄢ", "|jɛn"),
        ("ㄩㄢ", "|ɥæn"),
        ("ㄧㄣ", "|in"),
        ("ㄩㄣ", "|ɥn"),
        ("ㄧㄥ", "|iŋ"),
        ("ㄨㄥ", "|ʊŋ"),
        ("ㄩㄥ", "|jʊŋ"),
        # Add
        ("ㄧㄚ", "|ia"),
        ("ㄧㄝ", "|iɛ"),
        ("ㄧㄠ", "|iɑʊ"),
        ("ㄧㄡ", "|ioʊ"),
        ("ㄧㄤ", "|iɑŋ"),
        ("ㄨㄚ", "|ua"),
        ("ㄨㄛ", "|uo"),
        ("ㄨㄞ", "|uaɪ"),
        ("ㄨㄟ", "|ueɪ"),
        ("ㄨㄢ", "|uan"),
        ("ㄨㄣ", "|uən"),
        ("ㄨㄤ", "|uɑŋ"),
        ("ㄩㄝ", "|ɥɛ"),
        # End
        ("ㄅ", "p⁼"),
        ("ㄆ", "pʰ"),
        ("ㄇ", "m"),
        ("ㄈ", "f"),
        ("ㄉ", "t⁼"),
        ("ㄊ", "tʰ"),
        ("ㄋ", "n"),
        ("ㄌ", "l"),
        ("ㄍ", "k⁼"),
        ("ㄎ", "kʰ"),
        ("ㄏ", "x"),
        ("ㄐ", "tʃ⁼"),
        ("ㄑ", "tʃʰ"),
        ("ㄒ", "ʃ"),
        ("ㄓ", "ts`⁼"),
        ("ㄔ", "ts`ʰ"),
        ("ㄕ", "s`"),
        ("ㄖ", "ɹ`"),
        ("ㄗ", "ts⁼"),
        ("ㄘ", "tsʰ"),
        ("ㄙ", "|s"),
        ("ㄚ", "|a"),
        ("ㄛ", "|o"),
        ("ㄜ", "|ə"),
        ("ㄝ", "|ɛ"),
        ("ㄞ", "|aɪ"),
        ("ㄟ", "|eɪ"),
        ("ㄠ", "|ɑʊ"),
        ("ㄡ", "|oʊ"),
        ("ㄢ", "|an"),
        ("ㄣ", "|ən"),
        ("ㄤ", "|ɑŋ"),
        ("ㄥ", "|əŋ"),
        ("ㄦ", "əɹ"),
        ("ㄧ", "|i"),
        ("ㄨ", "|u"),
        ("ㄩ", "|ɥ"),
        ("ˉ", "→|"),
        ("ˊ", "↑|"),
        ("ˇ", "↓↑|"),
        ("ˋ", "↓|"),
        ("˙", "|"),
    ]
]
must_not_er_words = {"女儿", "老儿", "男儿", "少儿", "小儿"}

tone_dict = {
    "0": "˙",
    "5": "˙",
    "1": "",
    "2": "ˊ",
    "3": "ˇ",
    "4": "ˋ",
}

# Convert numbers to Chinese pronunciation
def number_to_chinese(text):
    # numbers = re.findall(r'\d+(?:\.?\d+)?', text)
    # for number in numbers:
    #     text = text.replace(number, cn2an.an2cn(number), 1)
    text = cn2an.transform(text, "an2cn")
    return text


def normalization(text):
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    text = text.replace("！", "!")
    text = text.replace("？", "?")
    text = text.replace("；", ";")
    text = text.replace("：", ":")
    text = text.replace("、", ",")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")
    text = text.replace("⋯", "…")
    text = text.replace("···", "…")
    text = text.replace("・・・", "…")
    text = text.replace("...", "…")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^\u4e00-\u9fff\s_,\.\?!;:\'…]", "", text)
    text = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", text)
    return text


def change_tone(bopomofo: str, tone: str) -> str:
    if bopomofo[-1] not in "˙ˊˇˋ":
        bopomofo = bopomofo + tone
    else:
        bopomofo = bopomofo[:-1] + tone
    return bopomofo


def er_sandhi(word: str, bopomofos: List[str]) -> List[str]:
    if len(word) > 1 and word[-1] == "儿" and word not in must_not_er_words:
        bopomofos[-1] = change_tone(bopomofos[-1], "˙")
    return bopomofos


def bu_sandhi(word: str, bopomofos: List[str]) -> List[str]:
    valid_char = set(word)
    if len(valid_char) == 1 and "不" in valid_char:
        pass
    elif word in ["不字"]:
        pass
    elif len(word) == 3 and word[1] == "不" and bopomofos[1][:-1] == "ㄅㄨ":
        bopomofos[1] = bopomofos[1][:-1] + "˙"
    else:
        for i, char in enumerate(word):
            if (
                i + 1 < len(bopomofos)
                and char == "不"
                and i + 1 < len(word)
                and 0 < len(bopomofos[i + 1])
                and bopomofos[i + 1][-1] == "ˋ"
            ):
                bopomofos[i] = bopomofos[i][:-1] + "ˊ"
    return bopomofos


def yi_sandhi(word: str, bopomofos: List[str]) -> List[str]:
    punc = "：，；。？！“”‘’':,;.?!()（）{}【】[]-~`、 "
    if word.find("一") != -1 and any(
        [item.isnumeric() for item in word if item != "一"]
    ):
        for i in range(len(word)):
            if (
                i == 0
                and word[0] == "一"
                and len(word) > 1
                and word[1]
                not in [
                    "零",
                    "一",
                    "二",
                    "三",
                    "四",
                    "五",
                    "六",
                    "七",
                    "八",
                    "九",
                    "十",
                ]
            ):
                if len(bopomofos[0]) > 0 and bopomofos[1][-1] in ["ˋ", "˙"]:
                    bopomofos[0] = change_tone(bopomofos[0], "ˊ")
                else:
                    bopomofos[0] = change_tone(bopomofos[0], "ˋ")
            elif word[i] == "一":
                bopomofos[i] = change_tone(bopomofos[i], "")
        return bopomofos
    elif len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
        bopomofos[1] = change_tone(bopomofos[1], "˙")
    elif word.startswith("第一"):
        bopomofos[1] = change_tone(bopomofos[1], "")
    elif word.startswith("一月") or word.startswith("一日") or word.startswith("一号"):
        bopomofos[0] = change_tone(bopomofos[0], "")
    else:
        for i, char in enumerate(word):
            if char == "一" and i + 1 < len(word):
                if (
                    len(bopomofos) > i + 1
                    and len(bopomofos[i + 1]) > 0
                    and bopomofos[i + 1][-1] in {"ˋ"}
                ):
                    bopomofos[i] = change_tone(bopomofos[i], "ˊ")
                else:
                    if word[i + 1] not in punc:
                        bopomofos[i] = change_tone(bopomofos[i], "ˋ")
                    else:
                        pass
    return bopomofos


def merge_bu(seg: List) -> List:
    new_seg = []
    last_word = ""
    for word in seg:
        if word != "不":
            if last_word == "不":
                word = last_word + word
            new_seg.append(word)
        last_word = word
    return new_seg


def merge_er(seg: List) -> List:
    new_seg = []
    for i, word in enumerate(seg):
        if i - 1 >= 0 and word == "儿":
            new_seg[-1] = new_seg[-1] + seg[i]
        else:
            new_seg.append(word)
    return new_seg


def merge_yi(seg: List) -> List:
    new_seg = []
    # function 1
    for i, word in enumerate(seg):
        if (
            i - 1 >= 0
            and word == "一"
            and i + 1 < len(seg)
            and seg[i - 1] == seg[i + 1]
        ):
            if i - 1 < len(new_seg):
                new_seg[i - 1] = new_seg[i - 1] + "一" + new_seg[i - 1]
            else:
                new_seg.append(word)
                new_seg.append(seg[i + 1])
        else:
            if i - 2 >= 0 and seg[i - 1] == "一" and seg[i - 2] == word:
                continue
            else:
                new_seg.append(word)
    seg = new_seg
    new_seg = []
    isnumeric_flag = False
    for i, word in enumerate(seg):
        if all([item.isnumeric() for item in word]) and not isnumeric_flag:
            isnumeric_flag = True
            new_seg.append(word)
        else:
            new_seg.append(word)
    seg = new_seg
    new_seg = []
    # function 2
    for i, word in enumerate(seg):
        if new_seg and new_seg[-1] == "一":
            new_seg[-1] = new_seg[-1] + word
        else:
            new_seg.append(word)
    return new_seg


# Word Segmentation, and convert Chinese pronunciation to pinyin (bopomofo)
def chinese_to_bopomofo(text_short, sentence, resources):
    # bopomofos = conv(text_short)
    words = jieba.lcut(text_short, cut_all=False)
    words = merge_yi(words)
    words = merge_bu(words)
    words = merge_er(words)
    text = ""

    char_index = 0
    for word in words:
        bopomofos = []
        if word in resources.word_pinyin_dict and word not in resources.poly_dict:
            pinyin = resources.word_pinyin_dict[word]
            for py in pinyin.split(" "):
                if py[:-1] in resources.pinyin_2_bopomofo_dict and py[-1] in tone_dict:
                    bopomofos.append(
                        resources.pinyin_2_bopomofo_dict[py[:-1]] + tone_dict[py[-1]]
                    )
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
                else:
                    bopomofos_lazy = lazy_pinyin(word, BOPOMOFO)
                    bopomofos += bopomofos_lazy
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
        else:
            for i in range(len(word)):
                c = word[i]
                if c in resources.poly_dict:
                    poly_pinyin = resources.g2pw_poly_predict.predict_process(
                        [text_short, char_index + i]
                    )[0]
                    py = poly_pinyin[2:-1]
                    bopomofos.append(
                        resources.pinyin_2_bopomofo_dict[py[:-1]] + tone_dict[py[-1]]
                    )
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
                elif c in resources.word_pinyin_dict:
                    py = resources.word_pinyin_dict[c]
                    bopomofos.append(
                        resources.pinyin_2_bopomofo_dict[py[:-1]] + tone_dict[py[-1]]
                    )
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
                else:
                    bopomofos.append(c)
                    if BLANK_LEVEL == 1:
                        bopomofos.append("_")
        if BLANK_LEVEL == 2:
            bopomofos.append("_")
        char_index += len(word)

        if (
            len(word) == 3
            and bopomofos[0][-1] == "ˇ"
            and bopomofos[1][-1] == "ˇ"
            and bopomofos[-1][-1] == "ˇ"
        ):
            bopomofos[0] = bopomofos[0] + "ˊ"
            bopomofos[1] = bopomofos[1] + "ˊ"
        if len(word) == 2 and bopomofos[0][-1] == "ˇ" and bopomofos[-1][-1] == "ˇ":
            bopomofos[0] = bopomofos[0][:-1] + "ˊ"
        bopomofos = bu_sandhi(word, bopomofos)
        bopomofos = yi_sandhi(word, bopomofos)
        bopomofos = er_sandhi(word, bopomofos)
        if not re.search("[\u4e00-\u9fff]", word):
            text += "|" + word
            continue
        for i in range(len(bopomofos)):
            bopomofos[i] = re.sub(r"([\u3105-\u3129])$", r"\1ˉ", bopomofos[i])
        if text != "":
            text += "|"
        text += "|".join(bopomofos)
    return text


# Convert latin pronunciation to pinyin (bopomofo)
def latin_to_bopomofo(text):
    for regex, replacement in _latin_to_bopomofo:
        text = re.sub(regex, replacement, text)
    return text


# Convert pinyin (bopomofo) to IPA
def bopomofo_to_ipa(text):
    for regex, replacement in _bopomofo_to_ipa:
        text = re.sub(regex, replacement, text)
    return text


def _chinese_to_ipa(text, sentence, resources):
    text = number_to_chinese(text.strip())
    text = normalization(text)
    text = chinese_to_bopomofo(text, sentence, resources)
    # pinyin = bpmf_to_pinyin(text)
    text = latin_to_bopomofo(text)
    text = bopomofo_to_ipa(text)
    text = re.sub("([sɹ]`[⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text)
    text = re.sub("([s][⁼ʰ]?)([→↓↑ ]+|$)", r"\1ɹ\2", text)
    text = re.sub(r"^\||[^\w\s_,\.\?!;:\'…\|→↓↑⁼ʰ`]", "", text)
    text = re.sub(r"([,\.\?!;:\'…])", r"|\1|", text)
    text = re.sub(r"\|+", "|", text)
    text = text.rstrip("|")
    return text


# Convert Chinese to IPA
def chinese_to_ipa(text, sentence, resources):
    # phonemes = text_tokenizer(text.strip())
    if type(text) == str:
        return _chinese_to_ipa(text, sentence, resources)
    else:
        result_ph = []
        for t in text:
            result_ph.append(_chinese_to_ipa(t, sentence, resources))
        return result_ph
