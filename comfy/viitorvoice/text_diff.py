# Adapted from viitor-ai/viitor-voice-nar 40359993534de0980eefff2d5a25bddc30f6bf58.
# Apache-2.0; see LICENSE. Only the pure sentence-editing helpers are retained.
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from .alignment import AlignmentItem


_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


_JA_KANA_RE = re.compile(r"[\u3040-\u30ff]")


_KO_HANGUL_RE = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]")


_ASCII_WORD_RE = re.compile(r"[A-Za-z]")


_WORD_DIFF_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class _DiffToken:
    text: str
    start: int
    end: int

    @property
    def key(self) -> str:
        return self.text.casefold()


@dataclass(frozen=True)
class _TextDiff:
    granularity: str
    opcodes: list[tuple[str, int, int, int, int]]
    changed_spans: list[tuple[int, int]]


def _normalize_edit_language(language: str | None) -> str | None:
    if language is None:
        return None
    text = str(language).strip().lower().replace("_", "-")
    aliases = {
        "english": "en",
        "eng": "en",
        "en-us": "en",
        "en-gb": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "zho": "zh",
        "cmn": "zh",
        "cn": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
        "japanese": "ja",
        "jpn": "ja",
        "jp": "ja",
        "ja-jp": "ja",
        "korean": "ko",
        "kor": "ko",
        "kr": "ko",
        "ko-kr": "ko",
    }
    return aliases.get(text, text or None)


def _build_text_diff(*, original_text: str, edited_text: str, language: str | None) -> _TextDiff:
    if _should_use_word_diff(language=language, text=f"{original_text}\n{edited_text}"):
        return _build_word_text_diff(original_text, edited_text)
    opcodes = difflib.SequenceMatcher(a=original_text, b=edited_text, autojunk=False).get_opcodes()
    return _TextDiff(
        granularity="char",
        opcodes=opcodes,
        changed_spans=_changed_text_spans(original_text, opcodes),
    )


def _should_use_word_diff(*, language: str | None, text: str) -> bool:
    lang = _normalize_edit_language(language)
    if lang == "en":
        return True
    if lang in {"zh", "yue", "ja", "ko"}:
        return False
    if _CJK_CHAR_RE.search(text) or _JA_KANA_RE.search(text) or _KO_HANGUL_RE.search(text):
        return False
    return bool(_ASCII_WORD_RE.search(text))


def _build_word_text_diff(original_text: str, edited_text: str) -> _TextDiff:
    original_tokens = _word_diff_tokens(original_text)
    edited_tokens = _word_diff_tokens(edited_text)
    if not original_tokens or not edited_tokens:
        opcodes = difflib.SequenceMatcher(a=original_text, b=edited_text, autojunk=False).get_opcodes()
        return _TextDiff(
            granularity="char",
            opcodes=opcodes,
            changed_spans=_changed_text_spans(original_text, opcodes),
        )

    token_opcodes = difflib.SequenceMatcher(
        a=[token.key for token in original_tokens],
        b=[token.key for token in edited_tokens],
        autojunk=False,
    ).get_opcodes()
    opcodes = [
        (
            tag,
            *_token_opcode_span(original_text, original_tokens, start, end),
            *_token_opcode_span(edited_text, edited_tokens, edited_start, edited_end),
        )
        for tag, start, end, edited_start, edited_end in token_opcodes
    ]
    spans = [
        _expand_word_changed_text_span(
            original_text=original_text,
            edited_text=edited_text,
            original_tokens=original_tokens,
            edited_tokens=edited_tokens,
            start=start,
            end=end,
            edited_start=edited_start,
            edited_end=edited_end,
        )
        for tag, start, end, edited_start, edited_end in token_opcodes
        if tag != "equal"
    ]
    if not spans:
        raise ValueError("edited_text is identical to original_text.")
    return _TextDiff(
        granularity="word",
        opcodes=opcodes,
        changed_spans=[
            (start, end)
            for start, end, _, _ in _merge_overlapping_text_spans(original_text, spans)
        ],
    )


def _word_diff_tokens(text: str) -> list[_DiffToken]:
    return [_DiffToken(match.group(0), match.start(), match.end()) for match in _WORD_DIFF_TOKEN_RE.finditer(text)]


def _token_opcode_span(text: str, tokens: list[_DiffToken], start: int, end: int) -> tuple[int, int]:
    if start < end:
        return tokens[start].start, tokens[end - 1].end
    if start <= 0:
        return 0, 0
    if start >= len(tokens):
        return len(text), len(text)
    return tokens[start].start, tokens[start].start


def _expand_word_changed_text_span(
    *,
    original_text: str,
    edited_text: str,
    original_tokens: list[_DiffToken],
    edited_tokens: list[_DiffToken],
    start: int,
    end: int,
    edited_start: int,
    edited_end: int,
) -> tuple[int, int, int, int]:
    if start < end:
        original_start = original_tokens[start].start
        original_end = original_tokens[end - 1].end
    elif start > 0:
        original_start = original_tokens[start - 1].start
        original_end = original_tokens[start - 1].end
    elif start < len(original_tokens):
        original_start = original_tokens[start].start
        original_end = original_tokens[start].end
    else:
        raise ValueError("original_text cannot be empty for text local edit.")

    if edited_start < edited_end:
        edited_char_start = edited_tokens[edited_start].start
        edited_char_end = edited_tokens[edited_end - 1].end
    elif edited_start > 0:
        edited_char_start = edited_tokens[edited_start - 1].start
        edited_char_end = edited_tokens[edited_start - 1].end
    elif edited_start < len(edited_tokens):
        edited_char_start = edited_tokens[edited_start].start
        edited_char_end = edited_tokens[edited_start].end
    else:
        edited_char_start = edited_char_end = min(len(edited_text), original_start)

    if start == end and edited_start < edited_end:
        if start > 0 and edited_start > 0:
            edited_char_start = edited_tokens[edited_start - 1].start
        elif start < len(original_tokens) and edited_end < len(edited_tokens):
            edited_char_end = edited_tokens[edited_end].end

    if edited_start == edited_end and start < end:
        if start > 0 and edited_start > 0:
            original_start = original_tokens[start - 1].start
            edited_char_start = edited_tokens[edited_start - 1].start
        elif end < len(original_tokens) and edited_start < len(edited_tokens):
            original_end = original_tokens[end].end
            edited_char_end = edited_tokens[edited_start].end

    return original_start, original_end, edited_char_start, edited_char_end


def _changed_text_spans(
    original_text: str,
    opcodes: list[tuple[str, int, int, int, int]],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int, int, int]] = []
    for tag, start, end, edited_start, edited_end in opcodes:
        if tag == "equal":
            continue
        spans.append(
            _expand_changed_text_span(
                original_text=original_text,
                start=start,
                end=end,
                edited_start=edited_start,
                edited_end=edited_end,
            )
        )
    if not spans:
        raise ValueError("edited_text is identical to original_text.")
    return [(start, end) for start, end, _, _ in _merge_overlapping_text_spans(original_text, spans)]


def _expand_changed_text_span(
    *,
    original_text: str,
    start: int,
    end: int,
    edited_start: int,
    edited_end: int,
) -> tuple[int, int, int, int]:
    if start == end:
        if start > 0:
            return start - 1, start, edited_start - 1, edited_end
        if original_text:
            return 0, 1, edited_start, edited_end + 1
        raise ValueError("original_text cannot be empty for text local edit.")

    if edited_start == edited_end:
        if start > 0:
            return start - 1, end, edited_start - 1, edited_end
        if end < len(original_text):
            return start, end + 1, edited_start, edited_end + 1
        raise ValueError("Deleting the entire text is not supported by local edit.")

    return start, end, edited_start, edited_end


def _merge_overlapping_text_spans(
    original_text: str,
    spans: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    ordered = sorted(spans, key=lambda span: (span[0], span[1]))
    merged: list[tuple[int, int, int, int]] = []
    for start, end, edited_start, edited_end in ordered:
        if not merged:
            merged.append((start, end, edited_start, edited_end))
            continue
        separator = original_text[merged[-1][1] : start]
        if start > merged[-1][1] and any(char.isalnum() for char in separator):
            merged.append((start, end, edited_start, edited_end))
            continue
        prev_start, prev_end, prev_edited_start, prev_edited_end = merged[-1]
        merged[-1] = (
            prev_start,
            max(prev_end, end),
            min(prev_edited_start, edited_start),
            max(prev_edited_end, edited_end),
        )
    return merged


def _merge_overlapping_index_groups(groups: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for group in groups:
        values = sorted(set(int(item) for item in group))
        if not values:
            continue
        if not merged:
            merged.append(values)
            continue
        previous = merged[-1]
        if not (set(previous) & set(values)) and values[0] > previous[-1] + 1:
            merged.append(values)
            continue
        merged[-1] = sorted(set(previous) | set(values))
    return merged


def _alignment_indices_for_char_span(
    alignments: list[AlignmentItem],
    start: int,
    end: int,
) -> list[int]:
    if not alignments:
        raise ValueError("Alignment returned no items.")
    indexed: list[tuple[int, int, int]] = []
    for item in alignments:
        if item.start_char is not None and item.end_char is not None and item.end_char > item.start_char:
            indexed.append((int(item.index), int(item.start_char), int(item.end_char)))
    if not indexed:
        raise ValueError("Alignment items do not include character spans.")

    selected = [idx for idx, item_start, item_end in indexed if item_end > start and item_start < end]
    if selected:
        return sorted(set(selected))

    after = [(item_start, idx) for idx, item_start, _ in indexed if item_start >= end]
    if after:
        return [min(after)[1]]
    before = [(item_end, idx) for idx, _, item_end in indexed if item_end <= start]
    if before:
        return [max(before)[1]]
    raise ValueError(f"Could not map text span [{start}, {end}) to alignment indices.")


def _char_span_for_alignment_indices(
    alignments: list[AlignmentItem],
    indices: list[int],
) -> tuple[int, int]:
    wanted = set(indices)
    selected = [
        item
        for item in alignments
        if int(item.index) in wanted and item.start_char is not None and item.end_char is not None and item.end_char > item.start_char
    ]
    if not selected:
        raise ValueError(f"Alignment indices do not include character spans: {indices}")
    return min(int(item.start_char) for item in selected), max(int(item.end_char) for item in selected)


def _edited_span_for_original_span(
    opcodes: list[tuple[str, int, int, int, int]],
    start: int,
    end: int,
    *,
    original_text: str = "",
) -> tuple[int, int]:
    edited_ranges: list[tuple[int, int]] = []
    for tag, orig_start, orig_end, edited_start, edited_end in opcodes:
        if tag == "insert":
            if start <= orig_start <= end or (
                orig_start >= end and original_text[end:orig_start].strip() == ""
            ):
                edited_ranges.append((edited_start, edited_end))
            continue
        overlap_start = max(start, orig_start)
        overlap_end = min(end, orig_end)
        if overlap_start >= overlap_end:
            continue
        if tag == "equal":
            edited_ranges.append(
                (
                    edited_start + (overlap_start - orig_start),
                    edited_start + (overlap_end - orig_start),
                )
            )
        else:
            edited_ranges.append((edited_start, edited_end))
    if not edited_ranges:
        raise ValueError(f"Could not map original text span [{start}, {end}) to edited text.")
    return min(item[0] for item in edited_ranges), max(item[1] for item in edited_ranges)
