# Adapted from viitor-ai/viitor-voice-nar 40359993534de0980eefff2d5a25bddc30f6bf58.
# Apache-2.0; see LICENSE. Only the pure sentence-editing helpers are retained.
from __future__ import annotations

import re
import random
from typing import Optional
import torch


PROMPT_TEXT_START_TOKEN = "<|prompt_text_start|>"


PROMPT_TEXT_END_TOKEN = "<|prompt_text_end|>"


TARGET_TEXT_START_TOKEN = "<|target_text_start|>"


TARGET_TEXT_END_TOKEN = "<|target_text_end|>"


TEXT_MASK_TOKEN = "<|text_mask|>"


PAUSE_ANCHOR_TOKEN = "<|pause_anchor|>"


PAUSE_ANCHOR_ALIASES = ("<|pause-anchor|>",)


AUTO_PAUSE_PUNCTUATION = frozenset(".!?。！？;；:：")


_JA_KANJI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


_JA_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff]")


_NONVERBAL_TAG_PATTERN = (
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)


_VIITORVOICE_TAG_PATTERN = r"<\|[^|]+?\|>"


_SPECIAL_TOKEN_PATTERN = re.compile(f"{_NONVERBAL_TAG_PATTERN}|{_VIITORVOICE_TAG_PATTERN}")


NVV_TAG_MAP = {
    "<|nv-Cough|>": "(coughs)",
    "<|nv-Sigh|>": "(sighs)",
    "<|nv-Breathing|>": "(inhale)",
    "<|nv-Surprise-oh|>": "(gasps)",
    "<|nv-Surprise-ah|>": "(gasps)",
    "<|nv-Dissatisfaction-hnn|>": "(groans)",
    "<|nv-Laughter|>": "(laughs)",
    "<|nv-Uhm|>": "(emm)",
    "<|nv-Question-en|>": "(humming)",
    "<|nv-Question-oh|>": "(humming)",
    "<|nv-Question-ah|>": "(humming)",
    "<|nv-Question-ei|>": "(humming)",
    "<|nv-Question-huh|>": "(emm)",
    "<|nv-Confirmation-en|>": "(humming)",
    "<|nv-Throat clearing|>": "(clear-throat)",
    "<|nv-Gasp|>": "(gasps)",
    "<|nv-Groaning|>": "(groans)",
    "<|nv-Surprise-wa|>": "(gasps)",
    "<|nv-Surprise-yo|>": "(gasps)",
    "<|nv-Question-yi|>": "(humming)",
}


_NVV_TAG_RE = re.compile("|".join(re.escape(tag) for tag in sorted(NVV_TAG_MAP, key=len, reverse=True)))


_PAUSE_ANCHOR_ALIAS_RE = re.compile("|".join(re.escape(tag) for tag in PAUSE_ANCHOR_ALIASES))


_LANG_NAME_TO_ID = {
    "english": "en",
    "chinese": "zh",
    "mandarin": "zh",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
    "portuguese": "pt",
    "italian": "it",
}


_LANG_IDS = frozenset(
    {
        "ar",
        "de",
        "en",
        "es",
        "fr",
        "hi",
        "it",
        "ja",
        "ko",
        "pt",
        "ru",
        "zh",
        *_LANG_NAME_TO_ID.values(),
    }
)


def resolve_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip().lower().replace("_", "-")
    if not normalized or normalized == "none":
        return None
    aliases = {
        "en-us": "en",
        "en-gb": "en",
        "eng": "en",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "zho": "zh",
        "cmn": "zh",
        "cn": "zh",
        "ja-jp": "ja",
        "jpn": "ja",
        "jp": "ja",
        "ko-kr": "ko",
        "kor": "ko",
        "kr": "ko",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in _LANG_IDS:
        return normalized
    return _LANG_NAME_TO_ID.get(normalized)


def normalize_nvv_tags(text: str) -> str:
    if not text:
        return text
    text = _NVV_TAG_RE.sub(lambda match: NVV_TAG_MAP[match.group(0)], text)
    return _PAUSE_ANCHOR_ALIAS_RE.sub(PAUSE_ANCHOR_TOKEN, text)


def prepare_text_for_tokenizer(text: str, tokenizer, language: Optional[str] = None) -> str:
    if not text:
        return text
    text = normalize_nvv_tags(text)
    is_japanese = language == "ja" or bool(_JA_KANA_RE.search(text))
    if not is_japanese:
        return text
    vocab_set = set(tokenizer.get_vocab())
    formatted = []
    for char in text:
        if _JA_KANJI_RE.fullmatch(char):
            ja_tag = f"<|ja-{char}|>"
            formatted.append(ja_tag if ja_tag in vocab_set else char)
        else:
            formatted.append(char)
    return "".join(formatted)


def insert_text_pause_anchors(text: str) -> str:
    if not text or PAUSE_ANCHOR_TOKEN in text:
        return text
    chars = list(text)
    anchored: list[str] = []
    for i, char in enumerate(chars):
        anchored.append(char)
        if char not in AUTO_PAUSE_PUNCTUATION:
            continue
        if any(not c.isspace() for c in chars[i + 1 :]):
            anchored.append(PAUSE_ANCHOR_TOKEN)
    return "".join(anchored)


def tokenize_with_special_tokens(text: str, tokenizer) -> torch.Tensor:
    parts: list[list[int]] = []
    last_end = 0
    for match in _SPECIAL_TOKEN_PATTERN.finditer(text):
        if match.start() > last_end:
            segment = text[last_end : match.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(match.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = match.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)
    if not parts:
        return tokenizer(text, return_tensors="pt").input_ids
    combined = []
    for token_ids in parts:
        combined.extend(token_ids)
    return torch.tensor([combined], dtype=torch.long)


def make_mask_text(mask_len: int) -> str:
    return TEXT_MASK_TOKEN * max(1, int(mask_len))


def wrap_prompt_target_text(prompt_text: str, target_text: str) -> str:
    return (
        f"{PROMPT_TEXT_START_TOKEN}{prompt_text}{PROMPT_TEXT_END_TOKEN}"
        f"{TARGET_TEXT_START_TOKEN}{target_text}{TARGET_TEXT_END_TOKEN}"
    )


def estimate_ref_text_mask_len(
    language_id: str | None,
    prompt_audio_frames: int,
    audio_frame_rate: int,
    tokens_per_second: dict[str, float],
    min_tokens: int,
    max_tokens: int,
    default_tokens_per_second: float,
    jitter_ratio: float = 0.0,
    jitter: bool = False,
    rng: random.Random | None = None,
) -> int:
    if audio_frame_rate <= 0:
        raise ValueError(f"audio_frame_rate should be positive, got {audio_frame_rate}.")
    if min_tokens <= 0:
        raise ValueError(f"min_tokens should be positive, got {min_tokens}.")
    if max_tokens < min_tokens:
        raise ValueError(
            f"max_tokens should be >= min_tokens, got {max_tokens} < {min_tokens}."
        )
    seconds = max(0.0, prompt_audio_frames / audio_frame_rate)
    rate = tokens_per_second.get(language_id or "", default_tokens_per_second)
    if rate <= 0:
        rate = default_tokens_per_second
    estimated = seconds * rate
    if jitter and jitter_ratio > 0.0:
        chooser = rng if rng is not None else random
        low = max(0.0, 1.0 - jitter_ratio)
        high = 1.0 + jitter_ratio
        estimated *= chooser.uniform(low, high)
    return max(min_tokens, min(max_tokens, int(round(estimated))))
