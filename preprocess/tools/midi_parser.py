"""
SoulX-Singer MIDI <-> metadata converter.

Converts between SoulX-Singer-style metadata JSON (with note_text, note_dur,
note_pitch, note_type per segment) and standard MIDI files. Uses an internal
Note dataclass (start_s, note_dur, note_text, note_pitch, note_type) as the
intermediate representation.
"""
import os
import json
import shutil
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

import librosa
import mido
from soundfile import write

from .f0_extraction import F0Extractor
from .g2p import g2p_transform


# Audio, MIDI and segmentation constants
SAMPLE_RATE = 44100     # Hz, fixed for all audio processing in this script to ensure consistent timing with MIDI ticks.
MIDI_TICKS_PER_BEAT = 500
MIDI_TEMPO = 500000     # microseconds per beat (120 BPM)
MIDI_TIME_SIGNATURE = (4, 4)
MIDI_VELOCITY = 64
END_EXTENSION_SEC = 0.4  # extend each segment end by this much silence (sec) to give the model more context
MAX_GAP_SEC = 2.0  # gap (sec) above which we start a new segment
MAX_SEGMENT_DUR_SUM_SEC = 60.0  # max cumulative note duration per segment (sec)
SILENCE_THRESHOLD_SEC = 0.2  # treat as separate <SP> if gap larger


@dataclass
class Note:
    """Single note: text, duration (seconds), pitch (MIDI), type. start_s is absolute start time in seconds (for ordering / MIDI)."""
    start_s: float
    note_dur: float
    note_text: str
    note_pitch: int
    note_type: int

    @property
    def end_s(self) -> float:
        return self.start_s + self.note_dur

def meta2notes(meta_path: str) -> List[Note]:
    """Parse SoulX-Singer metadata JSON into a flat list of Note (absolute start_s)."""
    with open(meta_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    if not isinstance(segments, list):
        raise ValueError(f"Metadata must be a list of segments, got {type(segments).__name__}")
    if not segments:
        raise ValueError("Metadata has no segments.")

    notes: List[Note] = []
    for seg in segments:
        offset_s = seg["time"][0] / 1000
        words = [str(x).replace("<AP>", "<SP>") for x in seg["text"].split()]
        word_durs = [float(x) for x in seg["duration"].split()]
        pitches = [int(x) for x in seg["note_pitch"].split()]
        types = [int(x) if words[i] != "<SP>" else 1 for i, x in enumerate(seg["note_type"].split())]
        if len(words) != len(word_durs) or len(word_durs) != len(pitches) or len(pitches) != len(types):
            raise ValueError(
                f"Length mismatch in segment {seg.get('item_name', '?')}: "
                "note_text, note_dur, note_pitch, note_type must have same length"
            )
        current_s = offset_s
        for text, dur, pitch, type_ in zip(words, word_durs, pitches, types):
            notes.append(
                Note(
                    start_s=current_s,
                    note_dur=float(dur),
                    note_text=str(text),
                    note_pitch=int(pitch),
                    note_type=int(type_),
                )
            )
            current_s += float(dur)
    return notes

def _append_segment_to_meta(
    meta_path_str: str,
    cut_wavs_output_dir: str | None,
    vocal_file: str | None,
    language: str,
    audio_data: Any | None,
    meta_data: List[dict],
    note_start: List[float],
    note_end: List[float],
    note_text: List[Any],
    note_pitch: List[Any],
    note_type: List[Any],
    note_dur: List[float],
) -> None:
    """Write one segment wav and append one segment dict to meta_data. Caller clears note_* lists after."""
    if not all((note_start, note_end, note_text, note_pitch, note_type, note_dur)):
        return

    base_name = os.path.splitext(os.path.basename(meta_path_str))[0]
    item_name = f"{base_name}_{len(meta_data)}"
    wav_fn = None
    if cut_wavs_output_dir and vocal_file and audio_data is not None:
        wav_fn = os.path.join(cut_wavs_output_dir, f"{item_name}.wav")
        end_pad = int(END_EXTENSION_SEC * SAMPLE_RATE)
        start_sample = max(0, int(note_start[0] * SAMPLE_RATE))
        end_sample = min(len(audio_data), int(note_end[-1] * SAMPLE_RATE) + end_pad)

        end_pad_dur = (end_sample / SAMPLE_RATE - note_end[-1]) if end_sample > int(note_end[-1] * SAMPLE_RATE) else 0.0
        if end_pad_dur > 0:
            note_dur = note_dur + [end_pad_dur]
            note_text = note_text + ["<SP>"]
            note_pitch = note_pitch + [0]
            note_type = note_type + [1]
        start_ms = int(start_sample / SAMPLE_RATE * 1000)
        end_ms = int(end_sample / SAMPLE_RATE * 1000)
        write(wav_fn, audio_data[start_sample:end_sample], SAMPLE_RATE)
    else:
        start_ms = int(note_start[0] * 1000)
        end_ms = int(note_end[-1] * 1000)

    meta_data.append({
        "item_name": item_name,
        "wav_fn": wav_fn,
        "origin_wav_fn": vocal_file,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "language": language,
        "note_text": list(note_text),
        "note_pitch": list(note_pitch),
        "note_type": list(note_type),
        "note_dur": list(note_dur),
    })


def convert_meta(meta_data: List[dict], pitch_extractor: F0Extractor | None) -> List[dict]:
    converted_data = []

    for item in meta_data:
        language = item.get("language", "Mandarin")
        wav_fn = item.get("wav_fn")
        if pitch_extractor is not None:
            if not wav_fn or not os.path.isfile(wav_fn):
                raise FileNotFoundError(f"Segment wav file not found: {wav_fn}")
            f0 = pitch_extractor.process(wav_fn)
        else:
            f0 = []
        converted_item = {
            "index": item.get("item_name"),
            "language": language,
            "time": [item.get("start_time_ms", 0), item.get("end_time_ms", sum(item["note_dur"]) * 1000)],
            "duration": " ".join(str(round(x, 2)) for x in item.get("note_dur", [])),
            "text": " ".join(item.get("note_text", [])),
            "phoneme": " ".join(g2p_transform(item.get("note_text", []), language)),
            "note_pitch": " ".join(str(x) for x in item.get("note_pitch", [])),
            "note_type": " ".join(str(x) for x in item.get("note_type", [])),
            "f0": " ".join(str(round(float(x), 1)) for x in f0),
        }
        converted_data.append(converted_item)

    return converted_data


def _edit_data_to_meta(
    meta_path_str: str,
    edit_data: List[dict],
    vocal_file: str | None,
    language: str,
    pitch_extractor: F0Extractor | None,
) -> None:
    """Write SoulX-Singer metadata JSON from edit_data (list of {start, end, note_text, note_pitch, note_type})."""
    # Store temporary cut wavs beside the source vocal (same folder, fixed subdir name).
    cut_wavs_output_dir = None
    if vocal_file:
        cut_wavs_output_dir = os.path.join(os.path.dirname(vocal_file), "cut_wavs_tmp")
        os.makedirs(cut_wavs_output_dir, exist_ok=True)

    note_text: List[Any] = []
    note_pitch: List[Any] = []
    note_type: List[Any] = []
    note_dur: List[float] = []
    note_start: List[float] = []
    note_end: List[float] = []
    meta_data: List[dict] = []
    audio_data = None
    if vocal_file:
        audio_data, _ = librosa.load(vocal_file, sr=SAMPLE_RATE, mono=True)
    dur_sum = 0.0

    def flush_current_segment() -> None:
        nonlocal dur_sum
        _append_segment_to_meta(
            meta_path_str,
            cut_wavs_output_dir,
            vocal_file,
            language,
            audio_data,
            meta_data,
            note_start,
            note_end,
            note_text,
            note_pitch,
            note_type,
            note_dur,
        )
        note_text.clear()
        note_pitch.clear()
        note_type.clear()
        note_dur.clear()
        note_start.clear()
        note_end.clear()
        dur_sum = 0.0

    for entry in edit_data:
        start = float(entry["start"])
        end = float(entry["end"])
        text = entry["note_text"]
        pitch = entry["note_pitch"]
        type_ = entry["note_type"]

        if text == "" or pitch == "" or type_ == "":
            note_text.append("<SP>")
            note_pitch.append(0)
            note_type.append(1)
            note_dur.append(end - start)
            note_start.append(start)
            note_end.append(end)
            dur_sum += end - start
            continue

        if (
            len(note_text) > 0
            and note_text[-1] == "<SP>"
            and note_dur[-1] > MAX_GAP_SEC
        ):
            note_text.pop()
            note_pitch.pop()
            note_type.pop()
            note_dur.pop()
            note_start.pop()
            note_end.pop()

            dur_sum = sum(note_dur)
            flush_current_segment()

        if dur_sum + (end - start) > MAX_SEGMENT_DUR_SUM_SEC and len(note_text) > 0:
            flush_current_segment()

        note_text.append(text)
        note_pitch.append(int(pitch))
        note_type.append(int(type_))
        note_dur.append(end - start)
        note_start.append(start)
        note_end.append(end)
        dur_sum += end - start

    if note_text:
        flush_current_segment()

    # Merge only consecutive <SP> tokens to reduce fragmentation in silence regions.
    for segment in meta_data:
        phoneme = segment['note_text']
        duration = segment['note_dur']
        note_pitch = segment['note_pitch']
        note_type = segment['note_type']

        merged_items: List[Tuple[str, float, int, int]] = []
        prev_item = None
        for text, dur, pitch, note_type in zip(phoneme, duration, note_pitch, note_type):
            if prev_item and text == "<SP>" and prev_item[0] == "<SP>":
                merged_items[-1] = (prev_item[0], prev_item[1] + dur, prev_item[2], prev_item[3])
            else:
                merged_items.append((text, dur, pitch, note_type))
            prev_item = merged_items[-1]

        segment['note_text'] = [item[0] for item in merged_items]
        segment['note_dur'] = [item[1] for item in merged_items]
        segment['note_pitch'] = [item[2] for item in merged_items]
        segment['note_type'] = [item[3] for item in merged_items]

    converted_data = convert_meta(meta_data, pitch_extractor)

    with open(meta_path_str, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    # Clean up temporary cut wavs directory
    if cut_wavs_output_dir:
        try:
            shutil.rmtree(cut_wavs_output_dir, ignore_errors=True)
        except Exception:
            pass


def notes2meta(
    notes: List[Note],
    meta_path: str,
    vocal_file: str | None,
    language: str,
    pitch_extractor: F0Extractor | None,
) -> None:
    """Write SoulX-Singer metadata JSON from a list of Note (segmenting + wav cuts)."""
    edit_data = [
        {
            "start": n.start_s,
            "end": n.end_s,
            "note_text": n.note_text,
            "note_pitch": str(n.note_pitch),
            "note_type": str(n.note_type),
        }
        for n in notes
    ]
    _edit_data_to_meta(
        str(meta_path),
        edit_data,
        vocal_file,
        language,
        pitch_extractor=pitch_extractor,
    )


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo: int) -> int:
    # ticks = seconds * (ticks_per_beat beats) / (tempo microseconds per beat)
    return int(round(seconds * ticks_per_beat * 1_000_000 / tempo))


def notes2midi(
    notes: List[Note],
    midi_path: str,
) -> None:
    """Write MIDI file from a list of Note."""
    if not notes:
        raise ValueError("Empty note list.")

    events: List[Tuple[int, int, Union[mido.Message, mido.MetaMessage]]] = []
    for n in notes:
        start_s = n.start_s
        end_s = n.end_s
        if end_s <= start_s:
            continue

        start_ticks = _seconds_to_ticks(
            start_s, MIDI_TICKS_PER_BEAT, MIDI_TEMPO
        )
        end_ticks = _seconds_to_ticks(
            end_s, MIDI_TICKS_PER_BEAT, MIDI_TEMPO
        )
        if end_ticks <= start_ticks:
            end_ticks = start_ticks + 1

        lyric = n.note_text
        # Some DAWs store lyric text as latin1-compatible bytes; keep best-effort round-trip.
        try:
            lyric = lyric.encode("utf-8").decode("latin1")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        if n.note_type == 3:
            lyric = "-"

        events.append(
            (start_ticks, 1, mido.MetaMessage("lyrics", text=lyric, time=0))
        )
        events.append(
            (
                start_ticks,
                2,
                mido.Message(
                    "note_on",
                    note=n.note_pitch,
                    velocity=MIDI_VELOCITY,
                    time=0,
                ),
            )
        )
        events.append(
            (
                end_ticks,
                0,
                mido.Message("note_off", note=n.note_pitch, velocity=0, time=0),
            )
        )

    # Keep deterministic ordering at same tick: note_off -> lyric -> note_on.
    events.sort(key=lambda x: (x[0], x[1]))

    mid = mido.MidiFile(ticks_per_beat=MIDI_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=MIDI_TEMPO, time=0))
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=MIDI_TIME_SIGNATURE[0],
            denominator=MIDI_TIME_SIGNATURE[1],
            time=0,
        )
    )

    last_tick = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - last_tick)
        track.append(msg)
        last_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.save(midi_path)


def midi2notes(midi_path: str) -> List[Note]:
    """Parse MIDI file into a list of Note.

    Merges all tracks and uses the latest encountered set_tempo as global tempo.
    """
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000

    raw_notes: List[dict] = []
    lyrics: List[Tuple[int, str]] = []

    for track in mid.tracks:
        abs_ticks = 0
        active = {}
        for msg in track:
            abs_ticks += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "lyrics":
                text = msg.text
                try:
                    text = text.encode("latin1").decode("utf-8")
                except Exception:
                    pass
                lyrics.append((abs_ticks, text))
            elif msg.type == "note_on":
                key = (msg.channel, msg.note)
                if msg.velocity > 0:
                    active[key] = (abs_ticks, msg.velocity)
                else:
                    if key in active:
                        start_ticks, vel = active.pop(key)
                        raw_notes.append(
                            {
                                "midi": msg.note,
                                "start_ticks": start_ticks,
                                "duration_ticks": abs_ticks - start_ticks,
                                "velocity": vel,
                                "lyric": "",
                            }
                        )
            elif msg.type == "note_off":
                key = (msg.channel, msg.note)
                if key in active:
                    start_ticks, vel = active.pop(key)
                    raw_notes.append(
                        {
                            "midi": msg.note,
                            "start_ticks": start_ticks,
                            "duration_ticks": abs_ticks - start_ticks,
                            "velocity": vel,
                            "lyric": "",
                        }
                    )

    if not raw_notes:
        raise ValueError("No notes found in MIDI file")

    for n in raw_notes:
        n["end_ticks"] = n["start_ticks"] + n["duration_ticks"]

    raw_notes.sort(key=lambda n: n["start_ticks"])
    lyrics.sort(key=lambda x: x[0])

    trimmed = []
    # Remove/trim overlaps so generated notes are strictly non-overlapping in tick domain.
    for note in raw_notes:
        while trimmed:
            prev = trimmed[-1]
            if note["start_ticks"] < prev["end_ticks"]:
                prev["end_ticks"] = note["start_ticks"]
                prev["duration_ticks"] = prev["end_ticks"] - prev["start_ticks"]
                if prev["duration_ticks"] <= 0:
                    trimmed.pop()
                    continue
            break
        trimmed.append(note)
    raw_notes = trimmed

    tolerance = ticks_per_beat // 100
    # Attach lyrics near note_on positions with a small tick tolerance.
    lyric_idx = 0
    for note in raw_notes:
        while lyric_idx < len(lyrics) and lyrics[lyric_idx][0] < note["start_ticks"] - tolerance:
            lyric_idx += 1
        if lyric_idx < len(lyrics):
            lyric_ticks, lyric_text = lyrics[lyric_idx]
            if abs(lyric_ticks - note["start_ticks"]) <= tolerance:
                note["lyric"] = lyric_text
                lyric_idx += 1

    def ticks_to_seconds(ticks: int) -> float:
        return (ticks / ticks_per_beat) * (tempo / 1_000_000)

    result: List[Note] = []
    prev_end_s = 0.0
    for idx, n in enumerate(raw_notes):
        start_s = ticks_to_seconds(n["start_ticks"])
        end_s = ticks_to_seconds(n["end_ticks"])
        if prev_end_s > start_s:
            start_s = prev_end_s
        dur_s = end_s - start_s
        if dur_s <= 0:
            continue

        lyric = n.get("lyric", "")
        # SoulX-Singer convention mapping from lyric token to note_type/text.
        if not lyric:
            tp = 2
            text = "啦"
        elif lyric == "<SP>":
            tp = 1
            text = "<SP>"
        elif lyric == "-":
            tp = 3
            text = raw_notes[idx - 1].get("lyric", "-") if idx > 0 else "-"
        else:
            tp = 2
            text = lyric

        if start_s - prev_end_s > SILENCE_THRESHOLD_SEC:
            # Explicitly represent long gaps as <SP> notes.
            result.append(
                Note(
                    start_s=prev_end_s,
                    note_dur=start_s - prev_end_s,
                    note_text="<SP>",
                    note_pitch=0,
                    note_type=1,
                )
            )
        else:
            if len(result) > 0:
                result[-1].note_dur = start_s - result[-1].start_s

        result.append(
            Note(
                start_s=start_s,
                note_dur=dur_s,
                note_text=text,
                note_pitch=n["midi"],
                note_type=tp,
            )
        )
        prev_end_s = end_s

    return result


class MidiParser:
    def __init__(
        self,
        rmvpe_model_path: str,
        device: str = "cuda",
    ) -> None:
        self.rmvpe_model_path = rmvpe_model_path
        self.device = device
        self.pitch_extractor: F0Extractor | None = None

    def _get_pitch_extractor(self) -> F0Extractor:
        if self.pitch_extractor is None:
            self.pitch_extractor = F0Extractor(
                self.rmvpe_model_path,
                device=self.device,
                verbose=False,
            )
        return self.pitch_extractor

    def midi2meta(
        self,
        midi_path: str,
        meta_path: str,
        vocal_file: str | None = None,
        language: str = "Mandarin",
    ) -> None:
        meta_dir = os.path.dirname(meta_path)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

        notes = midi2notes(midi_path)
        pitch_extractor = self._get_pitch_extractor() if vocal_file else None
        notes2meta(
            notes,
            meta_path,
            vocal_file,
            language,
            pitch_extractor=pitch_extractor,
        )
        print(f"Saved Meta to {meta_path}")

    def meta2midi(self, meta_path: str, midi_path: str) -> None:
        notes = meta2notes(meta_path)
        notes2midi(notes, midi_path)
        print(f"Saved MIDI to {midi_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SoulX-Singer metadata JSON <-> MIDI."
    )
    parser.add_argument("--meta", type=str, help="Path to metadata JSON")
    parser.add_argument("--midi", type=str, help="Path to MIDI file")
    parser.add_argument("--vocal", type=str, default=None, help="Path to vocal wav (optional for midi2meta)")
    parser.add_argument("--language", type=str, default="Mandarin", help="Lyric language for metadata phoneme conversion (default: Mandarin)")
    parser.add_argument(
        "--meta2midi",
        action="store_true",
        help="Convert meta -> midi (requires --meta and --midi)",
    )
    parser.add_argument(
        "--midi2meta",
        action="store_true",
        help="Convert midi -> meta (requires --midi and --meta; --vocal is optional)",
    )
    parser.add_argument(
        "--rmvpe_model_path",
        type=str,
        help="Path to RMVPE model",
        default="pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for RMVPE",
        default="cuda",
    )
    args = parser.parse_args()
    midi_parser = MidiParser(
        rmvpe_model_path=args.rmvpe_model_path,
        device=args.device,
    )

    if args.meta2midi:
        if not args.meta or not args.midi:
            parser.error("--meta2midi requires --meta and --midi")
        midi_parser.meta2midi(args.meta, args.midi)
    elif args.midi2meta:
        if not args.midi or not args.meta:
            parser.error(
                "--midi2meta requires --midi and --meta"
            )
        midi_parser.midi2meta(args.midi, args.meta, args.vocal, args.language)
    else:
        parser.print_help()