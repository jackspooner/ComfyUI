# Adapted from the Apache-2.0 OpenMOSS/LAION MOSS-TTS v1.5 processor,
# revision aabb7b609c2b56e5cfa8e8641eb1d8ce9e3d1997.
# See ../ldm/audio/moss_audio_tokenizer.LICENSE.
"""Pinned v1.5 text normalization, used only when explicitly selected by its node."""
from __future__ import annotations

import re

import unicodedata

_CJK_CHARS = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"

_CJK = f"[{_CJK_CHARS}]"

_PROT = r"___PROT\d+___"

_URL_RE = re.compile(r"https?://[^\s\u3000，。！？；、）】》〉」』]+")

_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")

_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")

_REDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")

_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+")

_DOT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+")

_FILELIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9._/+:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/+:-]*[._/+:-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?"
    r"(?![A-Za-z0-9_])"
)

_LATINISH = rf"(?:{_PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")

def normalize_tts_text(text: str) -> str:
    """对 TTS 输入做鲁棒性正则化。"""
    text = _base_cleanup(text)
    text = _normalize_markdown_and_lines(text)
    text, protected = _protect_spans(text)

    text = _normalize_spaces(text)
    text = _normalize_structural_punctuation(text)
    text = _normalize_repeated_punctuation(text)
    text = _normalize_spaces(text)

    text = _restore_spans(text, protected)
    return text.strip()

def _base_cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = _ZERO_WIDTH_RE.sub("", text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t " or not cat.startswith("C"):
            cleaned.append(ch)
    return "".join(cleaned)

def _normalize_markdown_and_lines(text: str) -> str:
    text = re.sub(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", r"\1 \2", text)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"^#{1,6}\s+", "", line)   # 标题
        line = re.sub(r"^>\s+", "", line)        # 引用
        line = re.sub(r"^[-*+]\s+", "", line)    # 无序列表
        line = re.sub(r"^\d+[.)]\s+", "", line)  # 有序列表
        lines.append(line)

    return "。".join(lines) if lines else ""

def _protect_spans(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(match.group(0))
        return f"___PROT{idx}___"

    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _MENTION_RE,
        _REDDIT_RE,
        _HASHTAG_RE,
        _DOT_TOKEN_RE,
        _FILELIKE_RE,
    ):
        text = pattern.sub(repl, text)

    return text, protected

def _restore_spans(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(f"___PROT{idx}___", original)
    return text

def _normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    text = re.sub(rf"({_CJK})\s+(?={_CJK})", r"\1", text)

    text = re.sub(rf"({_CJK})\s+(?=\d)", r"\1", text)
    text = re.sub(rf"(\d)\s+(?={_CJK})", r"\1", text)

    text = re.sub(rf"({_CJK})(?=({_LATINISH}))", r"\1 ", text)
    text = re.sub(rf"(({_LATINISH}))(?={_CJK})", r"\1 ", text)

    text = re.sub(r" {2,}", " ", text)

    text = re.sub(r"\s+([，。！？；：、”’」』】）》])", r"\1", text)
    text = re.sub(r"([（【「『《“‘])\s+", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s*", r"\1", text)

    text = re.sub(r"\s+([,.;!?])", r"\1", text)

    return re.sub(r" {2,}", " ", text).strip()

def _normalize_structural_punctuation(text: str) -> str:
    for _ in range(2):
        text = re.sub(
            r"(^|[。！？!?；;]\s*)[【〖『「]([^】〗』」]+)[】〗』」]\s*",
            r"\1\2。",
            text,
        )

    text = re.sub(
        r"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
        r"\1\2",
        text,
    )

    text = re.sub(
        r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*",
        "，",
        text,
    )

    text = re.sub(r"\s*(?:—|–|―|-){2,}\s*", "。", text)

    return text

def _normalize_repeated_punctuation(text: str) -> str:
    text = re.sub(r"(?:\.{3,}|…{2,}|……+)", "。", text)

    text = re.sub(r"[。．]{2,}", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)

    def _mixed_qe(match: re.Match[str]) -> str:
        s = match.group(0)
        has_q = any(ch in s for ch in "?？")
        has_e = any(ch in s for ch in "!！")
        if has_q and has_e:
            return "？！"
        return "？" if has_q else "！"

    text = re.sub(r"[!?！？]{2,}", _mixed_qe, text)
    return text
