import os
import random
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr

from preprocess.pipeline import PreprocessPipeline
from preprocess.tools.midi_parser import MidiParser
from soulxsinger.utils.file_utils import load_config
from cli.inference import build_model as build_svs_model, process as svs_process


ROOT = Path(__file__).parent
SAMPLE_RATE = 44100
PROMPT_MAX_SEC_DEFAULT = 30
TARGET_MAX_SEC_DEFAULT = 60
PROMPT_MAX_MERGE_DURATION_MS = 30000
TARGET_MAX_MERGE_DURATION_MS = 60000

ENGLISH_EXAMPLE_PROMPT_AUDIO = "example/audio/en_prompt.mp3"
ENGLISH_EXAMPLE_PROMPT_META = "example/audio/en_prompt.json"
ENGLISH_EXAMPLE_TARGET_AUDIO = "example/audio/en_target.mp3"
ENGLISH_EXAMPLE_TARGET_META = "example/audio/en_target.json"

MANDARIN_EXAMPLE_PROMPT_AUDIO = "example/audio/zh_prompt.mp3"
MANDARIN_EXAMPLE_PROMPT_META = "example/audio/zh_prompt.json"
MANDARIN_EXAMPLE_TARGET_AUDIO = "example/audio/zh_target.mp3"
MANDARIN_EXAMPLE_TARGET_META = "example/audio/zh_target.json"

CANTONESE_EXAMPLE_PROMPT_AUDIO = "example/audio/yue_prompt.mp3"
CANTONESE_EXAMPLE_PROMPT_META = "example/audio/yue_prompt.json"
CANTONESE_EXAMPLE_TARGET_AUDIO = "example/audio/yue_target.mp3"
CANTONESE_EXAMPLE_TARGET_META = "example/audio/yue_target.json"

MUSIC_EXAMPLE_TARGET_AUDIO = "example/audio/music.mp3"
MUSIC_EXAMPLE_TARGET_META = "example/audio/music.json"

# Use absolute paths so Examples load correctly (including File components for metadata)
EXAMPLES_LIST = [
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / MANDARIN_EXAMPLE_TARGET_META),
        "Mandarin",
        "Mandarin",
        "melody-controlled",
        "no",
        "yes",
        "yes",
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / CANTONESE_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / CANTONESE_EXAMPLE_TARGET_META),
        "Mandarin",
        "Cantonese",
        "melody-controlled",
        "no",
        "yes",
        "yes",
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / ENGLISH_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / ENGLISH_EXAMPLE_TARGET_META),
        "Mandarin",
        "English",
        "melody-controlled",
        "no",
        "yes",
        "yes",
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / MUSIC_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / MUSIC_EXAMPLE_TARGET_META),
        "Mandarin",
        "Mandarin",
        "score-controlled",
        "no",
        "yes",
        "yes",
        0,
    ],
]


# i18n
_I18N_KEY2LANG = dict(
    display_lang_label=dict(en="Display Language", zh="显示语言"),
    section_input_audio=dict(en="Input Audio", zh="输入音频"),
    section_transcriptions=dict(en="Transcriptions & Metadata", zh="转录与元数据"),
    section_synthesis=dict(en="Singing Synthesis", zh="歌声合成"),
    seed_label=dict(en="Seed", zh="种子"),
    prompt_audio_label=dict(en=f"Prompt audio (reference voice), limit to {PROMPT_MAX_SEC_DEFAULT} seconds", zh=f"Prompt 音频（参考音色），限制在 {PROMPT_MAX_SEC_DEFAULT} 秒以内"),
    target_audio_label=dict(en=f"Target audio (melody / lyrics source), limit to {TARGET_MAX_SEC_DEFAULT} seconds", zh=f"Target 音频（旋律/歌词来源），限制在 {TARGET_MAX_SEC_DEFAULT} 秒以内"),
    transcription_btn_label=dict(en="Run singing transcription", zh="开始歌声转录"),
    synthesis_btn_label=dict(en="🎤Generate singing voice", zh="🎤歌声合成"),
    prompt_meta_label=dict(en="Prompt metadata", zh="Prompt 元数据"),
    prompt_midi_label=dict(en="Prompt MIDI", zh="Prompt MIDI"),
    target_meta_label=dict(en="Target metadata", zh="Target 元数据"),
    target_midi_label=dict(en="Target MIDI", zh="Target MIDI"),
    prompt_wav_label=dict(en="Prompt WAV (reference)", zh="Prompt WAV（参考音色）"),
    generated_audio_label=dict(en="Generated merged audio", zh="合成结果音频"),
    prompt_lyric_lang_label=dict(en="Prompt lyric language", zh="Prompt 歌词语种"),
    target_lyric_lang_label=dict(en="Target lyric language", zh="Target 歌词语种"),
    lyric_lang_mandarin=dict(en="Mandarin", zh="普通话"),
    lyric_lang_cantonese=dict(en="Cantonese", zh="粤语"),
    lyric_lang_english=dict(en="English", zh="英语"),
    warn_missing_synthesis=dict(
        en="Please provide prompt WAV, prompt metadata, and target metadata. Check the content in Transcriptions & Metadata above.",
        zh="请提供 Prompt WAV、Prompt metadata 与 Target metadata，并检查上方 Transcriptions & Metadata 里的内容。",
    ),
    prompt_vocal_sep_label=dict(en="Prompt vocal separation", zh="Prompt人声分离"),
    target_vocal_sep_label=dict(en="Target vocal separation", zh="Target人声分离"),
    option_yes=dict(en="yes", zh="是"),
    option_no=dict(en="no", zh="否"),
    auto_shift_label=dict(en="Auto pitch shift", zh="自动变调"),
    pitch_shift_label=dict(en="Pitch shift (semitones)", zh="指定变调（半音）"),
    control_type_label=dict(en="Control type", zh="控制类型"),
    control_melody=dict(en="melody-controlled", zh="旋律控制"),
    control_score=dict(en="score-controlled", zh="乐谱控制"),
    examples_label=dict(en="Reference examples (click to load)", zh="参考样例（点击加载）"),
    example_choice_0=dict(en="—", zh="—"),
    example_choice_1=dict(en="Example 1: Mandarin → Mandarin (melody), Start singing synthesis!", zh="样例 1: 普通话 → 普通话 (melody), 开始歌声合成吧!"),
    example_choice_2=dict(en="Example 2: Mandarin → Cantonese (melody), Start singing synthesis!", zh="样例 2: 普通话 → 粤语 (melody), 开始歌声合成吧!"),
    example_choice_3=dict(en="Example 3: Mandarin → English (melody), Start singing synthesis!", zh="样例 3: 普通话 → 英语 (melody), 开始歌声合成吧!"),
    example_choice_4=dict(en="Example 4: Mandarin → Music (score), Start singing synthesis!", zh="样例 4: 普通话 → 音乐 (score), 开始歌声合成吧!"),
    instruction_title=dict(en="Usage", zh="使用说明"),
    instruction_p1=dict(
        en="Upload prompt and target audio, and the corresponding metadata and MIDI files will be automatically transcribed.",
        zh="上传 Prompt 与 Target 音频，将自动转录生成 Prompt 与 Target 两份 metadata 文件以及对应的 MIDI 文件。",
    ),
    instruction_p2=dict(
        en="Auto-transcribed lyrics and notes are often misaligned, which may lead to suboptimal synthesis results. For best results, import the generated MIDI into the [SoulX-Singer-Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor) for manual adjustment. After adjustment, re-upload the MIDI file and the metadata will be automatically updated.",
        zh="自动转录的歌词与音高对齐效果通常不理想，可能导致合成效果不佳，建议将生成的 MIDI 文件导入 [SoulX-Singer-Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor) 进行手动调整，调整后的 MIDI 文件重新上传后，metadata 将会自动更新。",
    ),
    instruction_p3=dict(
        en="Once prompt audio, prompt metadata, and target metadata are all set, click **🎤Generate singing voice** to run the singing synthesis and generate the final merged audio.",
        zh="Prompt audio, Prompt metadata 和 Target metadata 都准备好后，点击「🎤歌声合成」开始最终生成。",
    ),
)

_GLOBAL_LANG: Literal["zh", "en"] = "zh"


def _i18n(key: str) -> str:
    return _I18N_KEY2LANG[key][_GLOBAL_LANG]


def _load_example(choice_value):
    """Return 11 example values + skip_clear_count.

    When loading an example, the next two audio.change events should not clear metadata.
    """
    output_count = 11
    if choice_value is None:
        return [gr.update()] * output_count + [0]

    choice_to_index = {
        _i18n("example_choice_1"): 1,
        _i18n("example_choice_2"): 2,
        _i18n("example_choice_3"): 3,
        _i18n("example_choice_4"): 4,
    }

    idx = 0
    if isinstance(choice_value, int):
        idx = 0 if choice_value <= 0 else min(choice_value - 1, len(EXAMPLES_LIST) - 1)
    else:
        idx = choice_to_index.get(choice_value, 0)

    if idx <= 0:
        return [gr.update()] * output_count + [0]

    list_idx = idx - 1
    if list_idx >= len(EXAMPLES_LIST):
        return [gr.update()] * output_count + [0]

    row = EXAMPLES_LIST[list_idx]
    return [
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10],
        2,  # skip_clear_metadata_count: next 2 audio.change events (prompt + target) will not clear metadata
    ]


def _clear_prompt_meta_unless_example(_audio, skip_count):
    if skip_count and skip_count > 0:
        return gr.skip(), max(0, skip_count - 1)
    return None, 0


def _clear_target_meta_unless_example(_audio, skip_count):
    if skip_count and skip_count > 0:
        return gr.skip(), max(0, skip_count - 1)
    return None, 0


def _get_device() -> str:
    """Use CUDA if available, else CPU (e.g. for CI or CPU-only environments)."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _session_dir() -> Path:
    # Use per-call timestamped session dir to avoid cross-request collisions.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return ROOT / "outputs" / "gradio" / timestamp


def _print_exception(context: str) -> None:
    print(f"[{context}]\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def _get_lyric_lang_choices():
    """Lyric language dropdown (display, value) for current UI language."""
    return [
        (_i18n("lyric_lang_mandarin"), "Mandarin"),
        (_i18n("lyric_lang_cantonese"), "Cantonese"),
        (_i18n("lyric_lang_english"), "English"),
    ]


def _resolve_file_path(x):
    """Gradio file input can be path string or (path, None) tuple."""
    if x is None:
        return None
    if isinstance(x, tuple):
        x = x[0]
    return x if (x and os.path.isfile(x)) else None


def _normalize_audio_input(audio):
    """Normalize Gradio audio input to a filepath string."""
    return audio[0] if isinstance(audio, tuple) else audio


def _trim_and_save_audio(src_audio_path: str, dst_wav_path: Path, max_sec: int, sr: int = SAMPLE_RATE) -> None:
    """Load audio as mono, trim to max_sec, and save as wav for preprocess."""
    audio_data, _ = librosa.load(src_audio_path, sr=sr, mono=True)
    audio_data = audio_data[: max_sec * sr]
    sf.write(dst_wav_path, audio_data, sr)


def _yes_no_to_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() == "yes"


def _control_to_internal(control: str) -> str:
    if control in ("melody", "melody-controlled"):
        return "melody"
    return "score"


class AppState:
    def __init__(self) -> None:
        self.device = _get_device()
        self.preprocess_pipeline = PreprocessPipeline(
            device=self.device,
            language="Mandarin",
            save_dir=str(ROOT / "outputs" / "gradio" / "_placeholder" / "transcriptions"),
            vocal_sep=True,
            max_merge_duration=60000,
        )
        config = load_config("soulxsinger/config/soulxsinger.yaml")
        self.svs_config = config
        self.svs_model = build_svs_model(
            model_path="pretrained_models/SoulX-Singer/model.pt",
            config=config,
            device=self.device,
        )
        self.phoneset_path = "soulxsinger/utils/phoneme/phone_set.json"
        self.midi_parser = MidiParser(
            rmvpe_model_path="pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt",
            device=self.device
        )

    def run_preprocess(
        self,
        audio_path: Path,
        save_path: Path,
        vocal_sep: bool,
        lyric_lang: str,
        max_merge_duration: int
    ) -> Tuple[bool, str]:
        try:
            self.preprocess_pipeline.save_dir = str(save_path)
            self.preprocess_pipeline.run(
                audio_path=str(audio_path),
                vocal_sep=vocal_sep,
                max_merge_duration=max_merge_duration,
                language=lyric_lang or "Mandarin",
            )
            return True, f"preprocess {audio_path} done"
        except Exception as e:
            return False, f"preprocess failed: {e}"

    def run_svs(
        self,
        control: str,
        session_base: Path,
        auto_shift: bool,
        pitch_shift: int,
    ) -> Tuple[bool, str, Path | None, Path | None, Path | None]:
        if control not in ("melody", "score"):
            control = "score"
        save_dir = session_base / "generated"
        save_dir.mkdir(parents=True, exist_ok=True)
        class Args:
            pass
        args = Args()
        args.device = self.device
        args.model_path = "pretrained_models/SoulX-Singer/model.pt"
        args.config = "soulxsinger/config/soulxsinger.yaml"
        args.prompt_wav_path = str(session_base / "audio" / "prompt.wav")
        prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
        target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
        args.prompt_metadata_path = str(prompt_meta_path)
        args.target_metadata_path = str(target_meta_path)
        args.phoneset_path = self.phoneset_path
        args.save_dir = str(save_dir)
        args.auto_shift = auto_shift
        args.pitch_shift = int(pitch_shift)
        args.control = control
        try:
            svs_process(args, self.svs_config, self.svs_model)
            generated = save_dir / "generated.wav"
            if not generated.exists():
                return False, f"inference finished but {generated} not found", None, prompt_meta_path, target_meta_path
            return True, "svs inference done", generated, prompt_meta_path, target_meta_path
        except Exception as e:
            return False, f"svs inference failed: {e}", None, prompt_meta_path, target_meta_path

    def run_svs_from_paths(
        self,
        prompt_wav_path: str,
        prompt_metadata_path: str,
        target_metadata_path: str,
        control: str,
        auto_shift: bool,
        pitch_shift: int,
        save_dir: Path | None = None,
    ) -> Tuple[bool, str, Path | None]:
        """Run SVS from explicit prompt wav and metadata paths."""
        if save_dir is None:
            import uuid
            save_dir = ROOT / "outputs" / "gradio" / "synthesis" / str(uuid.uuid4())[:8]
        save_dir = Path(save_dir)
        audio_dir = save_dir / "audio"
        prompt_meta_dir = save_dir / "transcriptions" / "prompt"
        target_meta_dir = save_dir / "transcriptions" / "target"
        audio_dir.mkdir(parents=True, exist_ok=True)
        prompt_meta_dir.mkdir(parents=True, exist_ok=True)
        target_meta_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prompt_wav_path, audio_dir / "prompt.wav")
        shutil.copy2(prompt_metadata_path, prompt_meta_dir / "metadata.json")
        shutil.copy2(target_metadata_path, target_meta_dir / "metadata.json")
        ok, msg, merged, _, _ = self.run_svs(
            control=control,
            session_base=save_dir,
            auto_shift=auto_shift,
            pitch_shift=pitch_shift,
        )
        if not ok or merged is None:
            return False, msg or "svs failed", None
        return True, "svs inference done", merged


APP_STATE = AppState()

def _edit_metadata(
    meta,
    midi,
    audio,
    language: str = "Mandarin",
):
    try:
        meta = _resolve_file_path(meta)
        midi = _resolve_file_path(midi)
        if not midi:
            return meta
        audio = _normalize_audio_input(audio)

        if not meta:
            meta = str(Path(midi).with_name("metadata.json"))

        APP_STATE.midi_parser.midi2meta(midi, meta, audio, language=language)
        return meta
    except Exception:
        _print_exception("_edit_metadata")
        return meta


def _transcribe_prompt(
    prompt_audio,
    prompt_metadata,
    prompt_lyric_lang: str,
    prompt_vocal_sep,
    prompt_max_sec: int = PROMPT_MAX_SEC_DEFAULT,
):
    try:
        prompt_audio = _normalize_audio_input(prompt_audio)
        prompt_meta_resolved = _resolve_file_path(prompt_metadata)
        if prompt_audio is None and prompt_meta_resolved is None:
            return None, None

        session_base = _session_dir()
        prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
        prompt_midi_path = session_base / "transcriptions" / "prompt" / "vocal.mid"

        if prompt_audio is not None:
            audio_dir = session_base / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            transfer_prompt_path = audio_dir / "prompt.wav"
            _trim_and_save_audio(prompt_audio, transfer_prompt_path, prompt_max_sec)

            prompt_ok, prompt_msg = APP_STATE.run_preprocess(
                audio_path=transfer_prompt_path,
                save_path=session_base / "transcriptions" / "prompt",
                vocal_sep=_yes_no_to_bool(prompt_vocal_sep, default=False),
                lyric_lang=prompt_lyric_lang or "Mandarin",
                max_merge_duration=PROMPT_MAX_MERGE_DURATION_MS,
            )
            if not prompt_ok:
                print(prompt_msg, file=sys.stderr, flush=True)
                return None, None
        elif prompt_meta_resolved is not None:
            (session_base / "transcriptions" / "prompt").mkdir(parents=True, exist_ok=True)
            shutil.copy2(prompt_meta_resolved, prompt_meta_path)

        if prompt_meta_path.exists():
            APP_STATE.midi_parser.meta2midi(prompt_meta_path, prompt_midi_path)

        prompt_meta_file = str(prompt_meta_path) if prompt_meta_path.exists() else None
        prompt_midi_file = str(prompt_midi_path) if prompt_midi_path.exists() else None
        return prompt_meta_file, prompt_midi_file
    except Exception:
        _print_exception("_transcribe_prompt")
        return None, None


def _transcribe_target(
    target_audio,
    target_metadata,
    target_lyric_lang: str,
    target_vocal_sep,
    target_max_sec: int = TARGET_MAX_SEC_DEFAULT,
):
    try:
        target_audio = _normalize_audio_input(target_audio)
        target_meta_resolved = _resolve_file_path(target_metadata)
        if target_audio is None and target_meta_resolved is None:
            return None, None, None

        session_base = _session_dir()
        target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
        target_midi_path = session_base / "transcriptions" / "target" / "vocal.mid"
        target_vocal_path = session_base / "transcriptions" / "target" / "vocal.wav"

        if target_audio is not None:
            audio_dir = session_base / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            transfer_target_path = audio_dir / "target.wav"
            _trim_and_save_audio(target_audio, transfer_target_path, target_max_sec)

            target_ok, target_msg = APP_STATE.run_preprocess(
                audio_path=transfer_target_path,
                save_path=session_base / "transcriptions" / "target",
                vocal_sep=_yes_no_to_bool(target_vocal_sep, default=True),
                lyric_lang=target_lyric_lang or "Mandarin",
                max_merge_duration=TARGET_MAX_MERGE_DURATION_MS,
            )
            if not target_ok:
                print(target_msg, file=sys.stderr, flush=True)
                return None, None, None
        elif target_meta_resolved is not None:
            (session_base / "transcriptions" / "target").mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_meta_resolved, target_meta_path)

        if target_meta_path.exists():
            APP_STATE.midi_parser.meta2midi(target_meta_path, target_midi_path)

        target_meta_file = str(target_meta_path) if target_meta_path.exists() else None
        target_midi_file = str(target_midi_path) if target_midi_path.exists() else None
        target_vocal_file = str(target_vocal_path) if target_vocal_path.exists() else None
        return target_meta_file, target_midi_file, target_vocal_file
    except Exception:
        _print_exception("_transcribe_target")
        return None, None, None


def _run_synthesis(
    prompt_audio,
    prompt_metadata,
    target_metadata,
    control: str,
    auto_shift,
    pitch_shift,
    seed: int,
):
    """Run singing synthesis from prompt audio + prompt metadata + target metadata."""
    try:
        prompt_audio = _normalize_audio_input(prompt_audio)
        prompt_wav_path = prompt_audio
        prompt_meta_path = _resolve_file_path(prompt_metadata)
        target_meta_path = _resolve_file_path(target_metadata)
        if not prompt_wav_path or not os.path.isfile(prompt_wav_path):
            gr.Warning(message=_i18n("warn_missing_synthesis"))
            return None
        if not prompt_meta_path or not os.path.isfile(prompt_meta_path):
            gr.Warning(message=_i18n("warn_missing_synthesis"))
            return None
        if not target_meta_path or not os.path.isfile(target_meta_path):
            gr.Warning(message=_i18n("warn_missing_synthesis"))
            return None
        control = _control_to_internal(control)
        auto_shift = _yes_no_to_bool(auto_shift, default=True)
        seed = int(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        ok, msg, merged = APP_STATE.run_svs_from_paths(
            prompt_wav_path=prompt_wav_path,
            prompt_metadata_path=prompt_meta_path,
            target_metadata_path=target_meta_path,
            control=control,
            auto_shift=auto_shift,
            pitch_shift=int(pitch_shift),
        )
        if not ok or merged is None:
            print(msg or "synthesis failed", file=sys.stderr, flush=True)
            return None
        return str(merged)
    except Exception:
        _print_exception("_run_synthesis")
        return None


def _instruction_md() -> str:
    """Markdown content for the instruction panel (supports links)."""
    return "\n\n".join([
        f"**1.** {_i18n('instruction_p1')}",
        f"**2.** {_i18n('instruction_p2')}",
        f"**3.** {_i18n('instruction_p3')}",
    ])


def render_interface() -> gr.Blocks:
    with gr.Blocks(title="SoulX-Singer 歌声合成Demo", theme=gr.themes.Default()) as page:
        gr.HTML(
            '<div style="'
            'text-align: center; '
            'padding: 1.25rem 0 1.5rem; '
            'margin-bottom: 0.5rem;'
            '">'
            '<div style="'
            'display: inline-block; '
            'font-size: 1.75rem; '
            'font-weight: 700; '
            'letter-spacing: 0.02em; '
            'color: #1a1a2e; '
            'line-height: 1.3;'
            '">SoulX-Singer</div>'
            '<div style="'
            'width: 80px; '
            'height: 3px; '
            'margin: 1rem auto 0; '
            'background: linear-gradient(90deg, transparent, #6366f1, transparent); '
            'border-radius: 2px;'
            '"></div>'
            '</div>'
        )
        with gr.Row(equal_height=True):
            lang_choice = gr.Radio(
                choices=["中文", "English"],
                value="中文",
                label=_i18n("display_lang_label"),
                type="index",
                interactive=True,
                elem_id="lang_choice_radio",
            )

        # Instruction panel (usage workflow); updates on language change
        instruction_md = gr.Markdown(f"### {_i18n('instruction_title')}\n\n{_instruction_md()}")

        # Reference examples — at the front of operations (handler registered after components exist)
        skip_clear_metadata_count = gr.State(0)
        with gr.Row():
            _example_choices = [_i18n("example_choice_0"), _i18n("example_choice_1"), _i18n("example_choice_2"), _i18n("example_choice_3"), _i18n("example_choice_4")]
            example_choice = gr.Dropdown(
                label=_i18n("examples_label"),
                choices=_example_choices,
                value=_example_choices[0],
                interactive=True,
            )

        # Step 1: Transcription (audio → metadata)
        with gr.Accordion(_i18n("section_input_audio"), open=True) as accordion_input_audio:
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    prompt_audio = gr.Audio(
                        label=_i18n("prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                with gr.Column(scale=1):
                    target_audio = gr.Audio(
                        label=_i18n("target_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
        with gr.Accordion(_i18n("section_transcriptions"), open=True) as accordion_transcriptions:
            with gr.Row(equal_height=True):
                prompt_lyric_lang = gr.Dropdown(
                    label=_i18n("prompt_lyric_lang_label"),
                    choices=_get_lyric_lang_choices(),
                    value="Mandarin",
                    interactive=True,
                    scale=1,
                )
                prompt_vocal_sep = gr.Dropdown(
                    label=_i18n("prompt_vocal_sep_label"),
                    choices=[(_i18n("option_yes"), "yes"), (_i18n("option_no"), "no")],
                    value="no",
                    interactive=True,
                    scale=1,
                )
                target_lyric_lang = gr.Dropdown(
                    label=_i18n("target_lyric_lang_label"),
                    choices=_get_lyric_lang_choices(),
                    value="Mandarin",
                    interactive=True,
                    scale=1,
                )
                target_vocal_sep = gr.Dropdown(
                    label=_i18n("target_vocal_sep_label"),
                    choices=[(_i18n("option_yes"), "yes"), (_i18n("option_no"), "no")],
                    value="yes",
                    interactive=True,
                    scale=1,
                )

            with gr.Row(equal_height=True):
                prompt_metadata = gr.File(
                    label=_i18n("prompt_meta_label"),
                    type="filepath",
                    file_types=[".json"],
                    height=140,
                    interactive=True,
                )
                prompt_midi = gr.File(
                    label=_i18n("prompt_midi_label"),
                    type="filepath",
                    file_types=[".midi", ".mid"],
                    height=140,
                    interactive=True,
                )
                target_metadata = gr.File(
                    label=_i18n("target_meta_label"),
                    type="filepath",
                    file_types=[".json"],
                    height=140,
                    interactive=True,
                )
                target_midi = gr.File(
                    label=_i18n("target_midi_label"),
                    type="filepath",
                    file_types=[".midi", ".mid"],
                    height=140,
                    interactive=True,
                )
                target_vocal = gr.File(
                    type="filepath",
                    file_types=[".wav"],
                    interactive=False,
                    visible=False,
                )
        with gr.Accordion(_i18n("section_synthesis"), open=True) as accordion_synthesis:
            with gr.Row(equal_height=True):
                control_radio = gr.Dropdown(
                    choices=[(_i18n("control_melody"), "melody-controlled"), (_i18n("control_score"), "score-controlled")],
                    value="score-controlled",
                    label=_i18n("control_type_label"),
                    scale=1,
                )
                auto_shift = gr.Dropdown(
                    label=_i18n("auto_shift_label"),
                    choices=[(_i18n("option_yes"), "yes"), (_i18n("option_no"), "no")],
                    value="yes",
                    interactive=True,
                    scale=1,
                )
                pitch_shift = gr.Number(
                    label=_i18n("pitch_shift_label"),
                    value=0,
                    minimum=-36,
                    maximum=36,
                    step=1,
                    interactive=True,
                    scale=1,
                )
                seed_input = gr.Number(
                    label=_i18n("seed_label"),
                    value=12306,
                    step=1,
                    interactive=True,
                    scale=1,
                )
            with gr.Row():
                synthesis_btn = gr.Button(
                    value=_i18n("synthesis_btn_label"),
                    variant="primary",
                    size="lg",
                )
            with gr.Row():
                output_audio = gr.Audio(
                    label=_i18n("generated_audio_label"),
                    type="filepath",
                    interactive=False,
                )

        example_choice.change(
            fn=_load_example,
            inputs=[example_choice],
            outputs=[
                prompt_audio,
                target_audio,
                prompt_metadata,
                target_metadata,
                prompt_lyric_lang,
                target_lyric_lang,
                control_radio,
                prompt_vocal_sep,
                target_vocal_sep,
                auto_shift,
                pitch_shift,
                skip_clear_metadata_count,
            ],
        )

        def _change_component_language(lang):
            global _GLOBAL_LANG
            _GLOBAL_LANG = ["zh", "en"][lang]
            lyric_choices = _get_lyric_lang_choices()
            yes_no_choices = [(_i18n("option_yes"), "yes"), (_i18n("option_no"), "no")]
            control_choices = [(_i18n("control_melody"), "melody-controlled"), (_i18n("control_score"), "score-controlled")]
            return [
                gr.update(label=_i18n("prompt_audio_label")),
                gr.update(label=_i18n("target_audio_label")),
                gr.update(label=_i18n("prompt_lyric_lang_label"), choices=lyric_choices),
                gr.update(label=_i18n("target_lyric_lang_label"), choices=lyric_choices),
                gr.update(label=_i18n("prompt_vocal_sep_label"), choices=yes_no_choices),
                gr.update(label=_i18n("target_vocal_sep_label"), choices=yes_no_choices),
                gr.update(label=_i18n("prompt_meta_label")),
                gr.update(label=_i18n("target_meta_label")),
                gr.update(label=_i18n("control_type_label"), choices=control_choices),
                gr.update(label=_i18n("auto_shift_label"), choices=yes_no_choices),
                gr.update(label=_i18n("pitch_shift_label")),
                gr.update(label=_i18n("seed_label")),
                gr.update(value=_i18n("synthesis_btn_label")),
                gr.update(label=_i18n("generated_audio_label")),
                gr.update(label=_i18n("display_lang_label")),
                gr.update(
                    label=_i18n("examples_label"),
                    choices=[_i18n("example_choice_0"), _i18n("example_choice_1"), _i18n("example_choice_2"), _i18n("example_choice_3"), _i18n("example_choice_4")],
                    value=_i18n("example_choice_0"),
                ),
                gr.update(value=f"### {_i18n('instruction_title')}\n\n{_instruction_md()}"),
                gr.update(label=_i18n("section_input_audio")),
                gr.update(label=_i18n("section_transcriptions")),
                gr.update(label=_i18n("section_synthesis")),
            ]

        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice],
            outputs=[
                prompt_audio,
                target_audio,
                prompt_lyric_lang,
                target_lyric_lang,
                prompt_vocal_sep,
                target_vocal_sep,
                prompt_metadata,
                target_metadata,
                control_radio,
                auto_shift,
                pitch_shift,
                seed_input,
                synthesis_btn,
                output_audio,
                lang_choice,
                example_choice,
                instruction_md,
                accordion_input_audio,
                accordion_transcriptions,
                accordion_synthesis,
            ],
        )

        # Upload new prompt/target audio → clear corresponding metadata; skip clear when change came from load example
        prompt_audio.change(
            fn=_clear_prompt_meta_unless_example,
            inputs=[prompt_audio, skip_clear_metadata_count],
            outputs=[prompt_metadata, skip_clear_metadata_count],
        )
        prompt_audio.upload(
            fn=_transcribe_prompt,
            inputs=[prompt_audio, prompt_metadata, prompt_lyric_lang, prompt_vocal_sep],
            outputs=[prompt_metadata, prompt_midi],
        )
        prompt_midi.upload(
            fn=_edit_metadata,
            inputs=[prompt_metadata, prompt_midi, prompt_audio, prompt_lyric_lang],
            outputs=[prompt_metadata],
        )
        target_audio.change(
            fn=_clear_target_meta_unless_example,
            inputs=[target_audio, skip_clear_metadata_count],
            outputs=[target_metadata, skip_clear_metadata_count],
        )
        target_audio.upload(
            fn=_transcribe_target,
            inputs=[target_audio, target_metadata, target_lyric_lang, target_vocal_sep],
            outputs=[target_metadata, target_midi, target_vocal],
        )
        target_midi.upload(
            fn=_edit_metadata,
            inputs=[target_metadata, target_midi, target_vocal, target_lyric_lang],
            outputs=[target_metadata],
        )

        synthesis_btn.click(
            fn=_run_synthesis,
            inputs=[
                prompt_audio,
                prompt_metadata,
                target_metadata,
                control_radio,
                auto_shift,
                pitch_shift,
                seed_input,
            ],
            outputs=[output_audio],
        )

    return page


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    page = render_interface()
    page.queue()
    page.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
