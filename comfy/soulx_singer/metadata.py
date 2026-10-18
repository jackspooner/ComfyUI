"""Explicit, editable SoulX phoneme/note timing. Times are seconds, F0 is 50 Hz."""

import json
import math
import re
import statistics


def values(value, convert):
    return [convert(x) for x in (value.split() if isinstance(value, str) else value)]


def preprocess(meta, phone_set):
    durations = values(meta["duration"], float)
    phonemes = values(meta["phoneme"], str)
    pitches = values(meta["note_pitch"], int)
    types = values(meta["note_type"], int)
    if len({len(durations), len(phonemes), len(pitches), len(types)}) != 1 or not durations:
        raise ValueError("SoulX metadata needs equally sized duration, phoneme, note_pitch and note_type arrays.")
    if any(not math.isfinite(d) or d < 0 for d in durations):
        raise ValueError("Note durations must be finite and nonnegative.")
    merged = []
    for ph, duration, pitch, kind in zip(phonemes, durations, pitches, types):
        ph = ph.replace("<AP>", "<SP>")
        kind = 1 if ph == "<SP>" else kind
        if merged and ph == merged[-1][0] == "<SP>" and (pitch, kind) == tuple(merged[-1][2:]):
            merged[-1][1] += duration
        else:
            merged.append([ph, duration, pitch, kind])
    frame_count = int(sum(row[1] for row in merged) * 50)
    if frame_count == 0:
        raise ValueError("SoulX metadata contains no complete 20 ms frame.")
    alignment = [0] * frame_count
    tokens, note_pitches, note_types, locations = ["<PAD>"], [0], [1], []
    elapsed = 0
    for ph, duration, pitch, kind in merged:
        start = min(round(elapsed * 50), frame_count - 1)
        phones = ["en_" + p for p in ph[3:].split("-")] + ["<SEP>"] if ph.startswith("en_") else [ph]
        locations.append((start, len(phones)))
        tokens.extend(["<BOW>", *phones, "<EOW>"])
        note_pitches.extend([pitch] * (len(phones) + 2))
        note_types.extend([kind] * (len(phones) + 2))
        elapsed += duration
    token = 1
    for idx, (start, count) in enumerate(locations):
        end = locations[idx + 1][0] if idx + 1 < len(locations) else frame_count
        if start >= frame_count or start + count > frame_count:
            break
        while start < frame_count and alignment[start] > 0:
            start += 1
        if start >= frame_count:
            break
        alignment[start] = token
        k = start + 1
        while k + count < end:
            alignment[k:k + count] = range(token + 1, token + count + 1)
            k += count
        alignment[end - 1] = token + count + 1
        token += count + 2
    f0 = values(meta.get("f0", []), float)
    if f0:
        alignment = alignment[:len(f0)]
        f0 = f0[:len(alignment)]
    lookup = {ph: i for i, ph in enumerate(phone_set)}
    missing = sorted(set(tokens) - lookup.keys())
    if missing:
        raise ValueError(f"Unknown SoulX phonemes: {', '.join(missing[:12])}. Correct the phoneme metadata.")
    return {"phoneme": [lookup[ph] for ph in tokens], "note_pitch": note_pitches,
            "note_type": note_types, "mel2note": alignment, "f0": f0}


def pitch_notes(f0, min_frames=3):
    """A transparent F0-to-note estimate; not ROSVOT or a polyphonic transcription."""
    midi = [round(69 + 12 * math.log2(hz / 440)) if hz > 0 else 0 for hz in f0]
    smoothed = [statistics.median(midi[max(0, i - 1):i + 2]) for i in range(len(midi))]
    notes = []
    for i, pitch in enumerate(smoothed):
        if notes and (pitch == notes[-1][2] or i - notes[-1][0] < min_frames):
            notes[-1][1] = i + 1
        else:
            notes.append([i, i + 1, pitch])
    return [{"start": a / 50, "end": b / 50, "pitch": int(p)} for a, b, p in notes]


def syllables(phones):
    result, pending = [], []
    for phone in phones:
        if phone[-1:].isdigit() and any(p[-1:].isdigit() for p in pending):
            # A vowel nucleus starts a new timing slot; consonants remain explicit/editable.
            result.append(pending)
            pending = []
        pending.append(phone)
    if pending:
        result.append(pending)
    return result


def align_lyrics(pitch, timing, lyrics, pronounce, overrides=None):
    f0 = values(pitch["f0"], float)
    overrides = overrides or {}
    if isinstance(timing, dict):
        timing = timing.get("segments", timing.get("phrases", []))
    phrases = [p.strip() for p in re.split(r"\n|\|", lyrics) if p.strip()]
    if not phrases:
        timing = [word for region in timing for word in (region.get("words") or [region])]
    # Supplying explicit replacement phrase times avoids assuming the old words have the same syllable count.
    if phrases and len(phrases) != len(timing):
        raise ValueError(f"Replacement has {len(phrases)} phrases; timing has {len(timing)}. Separate phrases with | or newline, or edit the timing JSON.")
    rows = []
    for index, region in enumerate(timing):
        start, end = float(region["start"]), float(region["end"])
        text = phrases[index] if phrases else region.get("text", region.get("word", ""))
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)*|[\u3400-\u9fff]", text)
        items = []
        for word_index, word in enumerate(words):
            phone = overrides.get(word, overrides.get(word.lower()))
            phone = phone.split() if isinstance(phone, str) else phone
            phone = phone or pronounce(word)
            parts = syllables(phone) if word.isascii() else [phone]
            for syllable_index, part in enumerate(parts):
                items.append((word, word_index, syllable_index, part))
        if not items:
            continue
        a, b = max(0, round(start * 50)), min(len(f0), round(end * 50))
        if a >= b:
            continue
        voiced = [i for i in range(a, b) if f0[i] > 0]
        for k, (word, wi, si, phones) in enumerate(items):
            # Allocate each new syllable on the retained musical timeline; no waveform stretching.
            left = voiced[min(len(voiced) - 1, k * len(voiced) // len(items))] if voiced else a + (b - a) * k // len(items)
            right = (voiced[min(len(voiced) - 1, (k + 1) * len(voiced) // len(items))] if k + 1 < len(items) else b) if voiced else a + (b - a) * (k + 1) // len(items)
            prefix = "en_" if word.isascii() else ""
            voiced_pitch = [hz for hz in f0[left:right] if hz > 0]
            note = round(69 + 12 * math.log2(statistics.median(voiced_pitch) / 440)) if voiced_pitch else 0
            ph = prefix + "-".join(phones) if prefix else phones[0]
            # Vibrato/glides stay in F0: splitting each fluctuation into a phoneme
            # event can leave too few frames to emit the actual phones.
            rows.append({"start": left / 50, "end": right / 50, "text": word, "phoneme": ph,
                         "note_pitch": note, "note_type": 2, "phrase": index, "word": wi, "syllable": si})
    return metadata_from_rows(rows, f0, pitch.get("duration", len(f0) / 50))


def metadata_from_rows(rows, f0, duration):
    rows = sorted(rows, key=lambda r: r["start"])
    all_rows, cursor = [], 0.0
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if start < cursor - 1e-6 or end < start:
            raise ValueError("Lyric/note rows overlap or have reversed times; correct the editable JSON.")
        if start > cursor:
            all_rows.append({"start": cursor, "end": start, "phoneme": "<SP>", "note_pitch": 0, "note_type": 1})
        all_rows.append(row)
        cursor = end
    if cursor < duration:
        all_rows.append({"start": cursor, "end": duration, "phoneme": "<SP>", "note_pitch": 0, "note_type": 1})
    return {"time": [0, round(duration * 1000)], "duration": [r["end"] - r["start"] for r in all_rows],
            "phoneme": [r["phoneme"] for r in all_rows], "note_pitch": [r["note_pitch"] for r in all_rows],
            "note_type": [r.get("note_type", 2) for r in all_rows], "f0": f0, "rows": all_rows,
            "alignment_status": "estimated; inspect and correct word/syllable boundaries"}


def read_metadata(text):
    data = json.loads(text)
    def refresh(item):
        if "rows" not in item:
            return item
        time = item["time"]
        result = metadata_from_rows(item["rows"], values(item.get("f0", []), float), (time[1] - time[0]) / 1000)
        result["time"] = time
        return result
    return [refresh(item) for item in data] if isinstance(data, list) else refresh(data)


def segment_metadata(data, seconds=25):
    total = data["time"][1] / 1000
    segments = []
    start = 0
    while start < total:
        # Cut on a note boundary when one is available near the capacity window.
        end = min(total, start + seconds)
        candidates = [r["end"] for r in data["rows"] if start + seconds * .7 < r["end"] <= end]
        end = candidates[-1] if candidates else end
        rows = [{**r, "start": max(start, r["start"]) - start, "end": min(end, r["end"]) - start}
                for r in data["rows"] if r["start"] < end and r["end"] > start]
        part = metadata_from_rows(rows, data["f0"][round(start * 50):round(end * 50)], end - start)
        part["time"] = [round(start * 1000), round(end * 1000)]
        segments.append(part)
        start = end
    return segments
