"""Exact Fish prompt rows and UTF-8 chunking. Adapted from Fish Speech 5415669."""
import re
from collections.abc import Sequence
import torch
from tokenizers import Tokenizer
from .config import FishOmniConfig
InvalidInputError = ValueError
class Drama3PromptTooLongError(ValueError):
    pass
DRAMA3_TTS_INSTRUCTION = "convert the provided text to speech"
DRAMA3_REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
DRAMA3_MESSAGE_END = "<|im_end|>\n"
DRAMA3_GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
DRAMA3_STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
# The tokenizer has one token per speaker, <|speaker:0|> to <|speaker:15|>.
MAX_SPEAKER_ID = 15
# The Unicode White_Space property; str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
DRAMA3_MAX_PROMPT_ROWS = 8192
DRAMA3_DEFAULT_MAX_NEW_TOKENS = 2048
DRAMA3_PREFILL_CHUNK_SIZE = 128


def validate_drama3_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise InvalidInputError(f"{location}: text must not be empty")
    for token in DRAMA3_STRUCTURAL_TOKENS:
        if token in text:
            raise InvalidInputError(
                f"{location}: {token} is written by the prompt builder"
            )
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise InvalidInputError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )

def with_speaker_tag(text: str) -> str:
    """Prefix <|speaker:0|> unless a tag already leads the text after white space."""
    if SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is None:
        return f"<|speaker:0|> {text}"
    return text

def build_drama3_prompt(
    text: str,
    tokenizer: Tokenizer,
    config: FishOmniConfig,
    prompt_texts: Sequence[str] = (),
    prompt_tokens: Sequence[torch.Tensor] = (),
    history: Sequence[tuple[str, torch.Tensor]] = (),
) -> torch.Tensor:
    """CPU int64 rows [rows, 1 + num_codebooks] ending in the open assistant voice turn.

    Text rows are [token, 0, ..., 0]; audio rows are [c0 + semantic_start, c0..cN-1].
    Every literal is tokenized on its own: tokenizing the joined string changes ids
    at the seams. history holds earlier (text, codes) chunks, written as completed
    user and assistant voice turns before the current one.
    """
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    if len(prompt_texts) != len(prompt_tokens):
        raise InvalidInputError("every reference needs both a transcript and codes")
    validate_drama3_text(text, "text")

    def check_codes(codes: torch.Tensor, location: str) -> None:
        if (
            codes.dtype.is_floating_point
            or codes.dtype == torch.bool
            or codes.ndim != 2
            or codes.shape[0] != num_codebooks
            or codes.shape[1] == 0
        ):
            raise InvalidInputError(
                f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
                f"got {codes.dtype} {tuple(codes.shape)}"
            )
        if codes.min() < 0 or codes.max() >= codebook_size:
            raise InvalidInputError(
                f"{location}: codes must be in [0, {codebook_size})"
            )

    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if prompt_tokens else DRAMA3_TTS_INSTRUCTION,
    ]

    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        location = f"reference {index}"
        validate_drama3_text(transcript, location)
        check_codes(codes, location)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(DRAMA3_MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_drama3_text(turn_text, location)
        check_codes(codes, location)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            DRAMA3_MESSAGE_END,
            DRAMA3_GENERATION_START,
            codes,
            DRAMA3_MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        DRAMA3_MESSAGE_END,
        DRAMA3_GENERATION_START,
    ]

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment).ids, dtype=torch.long)
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(torch.cat((main, codes)).T)
    rows = torch.cat(rows)
    if len(rows) > DRAMA3_MAX_PROMPT_ROWS:
        raise Drama3PromptTooLongError(
            f"prompt has {len(rows)} rows; at most {DRAMA3_MAX_PROMPT_ROWS}"
        )
    return rows

# A threshold, not a hard cap: a protected span longer than this stays whole.
DRAMA3_CHUNK_BYTES = 300
DRAMA3_CHUNK_MAX_SPEAKERS = 3
# Frames per chunk for max_new_tokens <= 0, about 27.9 s at 2048 samples per
# frame and 44.1 kHz.
DRAMA3_CHUNK_MAX_NEW_TOKENS = 600

SENTENCE_ENDS = frozenset(".!?;。！？；…\n")
CLAUSE_MARKS = frozenset(",:，：、")
SPACES = frozenset(" ")
# Never split inside: special tokens, paired tags such as <whisper>…</whisper>, and
# [bracketed] cues.
PROTECTED_SPAN = re.compile(
    r"<\|[^|>]*\|>|<([A-Za-z][\w-]*)\b[^>]*>.*?</\1>|\[[^\[\]]*\]", re.S
)

Piece = tuple[str, bool]  # (text, protected)


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _protected_pieces(text: str) -> list[Piece]:
    pieces, last = [], 0
    for match in PROTECTED_SPAN.finditer(text):
        if match.start() > last:
            pieces.append((text[last : match.start()], False))
        pieces.append((match.group(0), True))
        last = match.end()
    if last < len(text):
        pieces.append((text[last:], False))
    return pieces


def _breaks_at(text: str, index: int, marks: frozenset[str]) -> bool:
    if text[index] not in marks:
        return False
    # Keep decimals such as 3.14 in one piece.
    return not (
        text[index] == "."
        and 0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _break_at_marks(pieces: list[Piece], length: int, marks: frozenset[str]):
    """Split free pieces longer than length bytes after each mark, keeping it."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        start = 0
        for index in range(len(text)):
            if _breaks_at(text, index, marks):
                out.append((text[start : index + 1], False))
                start = index + 1
        if start < len(text):
            out.append((text[start:], False))
    return out


def _break_by_length(pieces: list[Piece], length: int) -> list[Piece]:
    """Split the free pieces still longer than length on character boundaries."""
    out = []
    for text, protected in pieces:
        if protected or utf8_len(text) <= length:
            out.append((text, protected))
            continue
        current, size = "", 0
        for char in text:
            if current and size + utf8_len(char) > length:
                out.append((current, False))
                current, size = "", 0
            current += char
            size += utf8_len(char)
        out.append((current, False))
    return out


def split_by_length(text: str, length: int = DRAMA3_CHUNK_BYTES) -> list[str]:
    """Chunks of about length bytes, packed sentence by sentence.

    Text is cut at every sentence end; a sentence still longer than length is cut
    at clause marks, then spaces, then on raw length. Adjacent pieces are then
    merged back up to length.
    """
    if not text.strip(UNICODE_WHITE_SPACE):
        return []
    if utf8_len(text) <= length:
        return [text]
    pieces = _break_at_marks(_protected_pieces(text), 0, SENTENCE_ENDS)
    for marks in (CLAUSE_MARKS, SPACES):
        pieces = _break_at_marks(pieces, length, marks)
    pieces = _break_by_length(pieces, length)

    chunks, current = [], ""
    for piece, _ in pieces:
        if current and utf8_len(current + piece) > length:
            chunks.append(current)
            current = ""
        current += piece
    chunks.append(current)
    return [
        chunk.strip(UNICODE_WHITE_SPACE)
        for chunk in chunks
        if chunk.strip(UNICODE_WHITE_SPACE)
    ]


def parse_speaker_segments(text: str) -> list[tuple[str, str]]:
    """(speaker id, text) per tagged segment; text before the first tag is speaker 0."""
    parts = SPEAKER_TAG_PATTERN.split(text)
    segments = [("0", parts[0])] if parts[0].strip(UNICODE_WHITE_SPACE) else []
    for speaker, body in zip(parts[1::2], parts[2::2]):
        if body.strip(UNICODE_WHITE_SPACE):
            segments.append((speaker, body))
    return segments


def split_drama3_text(
    text: str,
    max_bytes: int = DRAMA3_CHUNK_BYTES,
    max_speakers: int = DRAMA3_CHUNK_MAX_SPEAKERS,
) -> list[str]:
    """Generation chunks for text, in order.

    Untagged text is split by length. Tagged text is split into speaker segments
    (long ones cut further, each piece keeping its tag) and regrouped: a new chunk
    starts when a segment would bring in one speaker too many or exceed max_bytes,
    so A, B, A can share a chunk.
    """
    if SPEAKER_TAG_PATTERN.search(text) is None:
        return split_by_length(text, max_bytes)

    segments = []
    for speaker, body in parse_speaker_segments(text):
        for piece in split_by_length(body, max_bytes):
            segments.append((speaker, f"<|speaker:{speaker}|> {piece}"))

    chunks, batch, speakers, size = [], [], [], 0
    for speaker, segment in segments:
        new_speaker = speaker not in speakers
        too_many_speakers = new_speaker and len(speakers) >= max_speakers
        too_long = bool(batch) and size + utf8_len(segment) > max_bytes
        if too_many_speakers or too_long:
            chunks.append("\n".join(batch))
            batch, speakers, size = [], [], 0
            new_speaker = True
        batch.append(segment)
        if new_speaker:
            speakers.append(speaker)
        size += utf8_len(segment)
    if batch:
        chunks.append("\n".join(batch))
    return chunks
