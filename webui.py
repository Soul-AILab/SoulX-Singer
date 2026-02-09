import os
import re
import random
import shutil
import sys
import traceback
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import librosa
import soundfile as sf
import gradio as gr

from preprocess.pipeline import PreprocessPipeline
from soulxsinger.utils.file_utils import load_config
from cli.inference import build_model as build_svs_model, process as svs_process


ROOT = Path(__file__).parent

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

# Lyric language: value (Mandarin/Cantonese/English) is passed to PreprocessPipeline; display labels from i18n via get_lyric_lang_choices()

# Use absolute paths so Examples load correctly (including File components for metadata)
EXAMPLES_LIST = [
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / MANDARIN_EXAMPLE_TARGET_META),
        "Mandarin",
        "Mandarin",
        "melody",
        False,
        True,
        True,
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / CANTONESE_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / CANTONESE_EXAMPLE_TARGET_META),
        "Mandarin",
        "Cantonese",
        "melody",
        False,
        True,
        True,
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / ENGLISH_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / ENGLISH_EXAMPLE_TARGET_META),
        "Mandarin",
        "English",
        "melody",
        False,
        True,
        True,
        0,
    ],
    [
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_AUDIO),
        str(ROOT / MUSIC_EXAMPLE_TARGET_AUDIO),
        str(ROOT / MANDARIN_EXAMPLE_PROMPT_META),
        str(ROOT / MUSIC_EXAMPLE_TARGET_META),
        "Mandarin",
        "Mandarin",
        "melody",
        False,
        True,
        True,
        0,
    ],
]


def _load_example(choice_value):
    """Return 11 example values + skip_clear_count (2 when loading example so next 2 audio.change events don't clear metadata).
    choice_value: selected dropdown string (or index in older flow); map to example index 0/1/2."""
    if choice_value is None:
        return [gr.update()] * 11 + [0]
    idx = 0
    if isinstance(choice_value, int):
        idx = 0 if choice_value <= 0 else min(choice_value - 1, len(EXAMPLES_LIST) - 1)
    else:
        if choice_value == i18n("example_choice_1"):
            idx = 1
        elif choice_value == i18n("example_choice_2"):
            idx = 2
        elif choice_value == i18n("example_choice_3"):
            idx = 3
        elif choice_value == i18n("example_choice_4"):
            idx = 4
    if idx <= 0:
        return [gr.update()] * 11 + [0]
    list_idx = idx - 1
    if list_idx >= len(EXAMPLES_LIST):
        return [gr.update()] * 11 + [0]
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


def _session_dir_from_target(target_audio_path: str) -> Path:
    stem = Path(target_audio_path).stem
    safe = re.sub(r"[^\w\-]", "_", stem)
    safe = re.sub(r"_+", "_", safe).strip("_") or "session"
    return ROOT / "outputs" / "gradio" / safe[:64]


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

    def run_preprocess(
        self,
        prompt_path: Path,
        target_path: Path,
        session_base: Path,
        prompt_vocal_sep: bool,
        target_vocal_sep: bool,
        prompt_lyric_lang: str,
        target_lyric_lang: str,
    ) -> Tuple[bool, str]:
        try:
            self.preprocess_pipeline.save_dir = str(session_base / "transcriptions" / "prompt")
            self.preprocess_pipeline.run(
                audio_path=str(prompt_path),
                vocal_sep=prompt_vocal_sep,
                max_merge_duration=20000,
                language=prompt_lyric_lang or "Mandarin",
            )
            self.preprocess_pipeline.save_dir = str(session_base / "transcriptions" / "target")
            self.preprocess_pipeline.run(
                audio_path=str(target_path),
                vocal_sep=target_vocal_sep,
                max_merge_duration=60000,
                language=target_lyric_lang or "Mandarin",
            )
            return True, "preprocess done"
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


# i18n
_i18n_key2lang_dict = dict(
    display_lang_label=dict(en="Display Language", zh="显示语言"),
    seed_label=dict(en="Seed", zh="种子"),
    prompt_audio_label=dict(en="Prompt audio (reference voice), limit to 30 seconds", zh="Prompt 音频（参考音色），限制在 30 秒以内"),
    target_audio_label=dict(en="Target audio (melody / lyrics source), limit to 60 seconds", zh="Target 音频（旋律/歌词来源），限制在 60 秒以内"),
    generate_btn_label=dict(en="Start SVS", zh="开始 SVS"),
    transcription_btn_label=dict(en="Run singing transcription", zh="开始歌声转录"),
    synthesis_btn_label=dict(en="Run singing synthesis", zh="歌声合成"),
    prompt_meta_label=dict(en="Prompt metadata", zh="Prompt metadata"),
    target_meta_label=dict(en="Target metadata", zh="Target metadata"),
    
    edit_tutorial_html=dict(
        en='<p class="mb-0">Refer to <a href="https://github.com/Soul-AILab/SoulX-Singer/tree/main/preprocess#step-2-edit-in-the-midi-editor" target="_blank" rel="noopener">Edit Tutorial</a> for metadata editing (Important Note: The generated metadata may not perfectly align the singing audio with the corresponding lyrics and musical notes. For better results, we strongly recommend manually correcting the alignment. You can directly use <a href="https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor" target="_blank" rel="noopener">SoulX-Singer-Midi-Editor</a> to edit) </p>',
        zh='<p class="mb-0">metadata 编辑请参考 <a href="https://github.com/Soul-AILab/SoulX-Singer/tree/main/preprocess#step-2-edit-in-the-midi-editor" target="_blank" rel="noopener">编辑教程</a> (重要提示：自动生成的 metadata 在音频与歌词、音高对齐效果通常不理想。为了获得更好的结果，我们强烈建议手动纠正对齐，否则会导致合成效果不佳。 你可以直接使用 <a href="https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor" target="_blank" rel="noopener">SoulX-Singer-Midi-Editor</a> 进行编辑) </p>',
    ),
    prompt_wav_label=dict(en="Prompt WAV (reference)", zh="Prompt WAV（参考音色）"),
    generated_audio_label=dict(en="Generated merged audio", zh="合成结果音频"),
    prompt_lyric_lang_label=dict(en="Prompt lyric language", zh="Prompt 歌词语种"),
    target_lyric_lang_label=dict(en="Target lyric language", zh="Target 歌词语种"),
    lyric_lang_mandarin=dict(en="Mandarin", zh="普通话"),
    lyric_lang_cantonese=dict(en="Cantonese", zh="粤语"),
    lyric_lang_english=dict(en="English", zh="英语"),
    warn_missing_synthesis=dict(en="Please provide prompt WAV, prompt metadata, and target metadata", zh="请提供 Prompt WAV、Prompt metadata 与 Target metadata"),
    prompt_vocal_sep_label=dict(en="Prompt vocal separation", zh="Prompt人声分离"),
    target_vocal_sep_label=dict(en="Target vocal separation", zh="Target人声分离"),
    auto_shift_label=dict(en="Auto pitch shift", zh="自动变调"),
    pitch_shift_label=dict(en="Pitch shift (semitones)", zh="指定变调（半音）"),
    control_type_label=dict(en="Control type", zh="控制类型"),
    examples_label=dict(en="Reference examples (click to load)", zh="参考样例（点击加载）"),
    example_choice_0=dict(en="—", zh="—"),
    example_choice_1=dict(en="Example 1: Mandarin → Mandarin (melody), Start singing synthesis!", zh="样例 1: 普通话 → 普通话 (melody), 开始歌声合成吧!"),
    example_choice_2=dict(en="Example 2: Mandarin → Cantonese (melody), Start singing synthesis!", zh="样例 2: 普通话 → 粤语 (melody), 开始歌声合成吧!"),
    example_choice_3=dict(en="Example 3: Mandarin → English (melody), Start singing synthesis!", zh="样例 3: 普通话 → 英语 (melody), 开始歌声合成吧!"),
    example_choice_4=dict(en="Example 4: Mandarin → Music (score), Start singing synthesis!", zh="样例 4: 普通话 → 音乐 (score), 开始歌声合成吧!"),
    warn_missing_audio=dict(
        en="Please upload both prompt audio and target audio",
        zh="请上传 Prompt 音频与 Target 音频",
    ),
    # Instruction panel (workflow description)
    instruction_title=dict(en="Usage", zh="使用说明"),
    instruction_p1=dict(
        en="After uploading prompt and target audio and clicking **Run singing transcription**, the system generates two metadata files (prompt and target).",
        zh="上传 Prompt 与 Target 音频并点击「开始歌声转录」后，将生成 Prompt 与 Target 两份 metadata 文件。",
    ),
    instruction_p2=dict(
        en="Auto-transcribed lyrics and notes are often misaligned. For better results, import the generated metadata into the **MIDI Editor** for manual adjustment: [SoulX-Singer-Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor).",
        zh="自动转录的歌词与音高对齐效果通常不理想，建议将生成的 metadata 导入 **MIDI 编辑器** 进行手动调整：[SoulX-Singer-Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor)。",
    ),
    instruction_p3=dict(
        en="Re-upload the adjusted metadata to the corresponding Prompt / Target Meta fields, then click **Run singing synthesis** to generate the final audio.",
        zh="将调整后的 metadata 重新上传至对应的 Prompt / Target Meta 位置后，点击「歌声合成」开始最终生成。",
    ),
)

def _detect_initial_lang() -> Literal["zh", "en"]:
    """Detect initial UI language from server locale (browser language applied later via JS)."""
    try:
        import locale
        loc = (locale.getdefaultlocale()[0] or os.environ.get("LANG", "") or "").lower()
        return "en" if loc.startswith("en") else "zh"
    except Exception:
        return "zh"


global_lang: Literal["zh", "en"] = _detect_initial_lang()


def i18n(key: str) -> str:
    return _i18n_key2lang_dict[key][global_lang]


def get_lyric_lang_choices():
    """Lyric language dropdown (display, value) for current UI language."""
    return [
        (i18n("lyric_lang_mandarin"), "Mandarin"),
        (i18n("lyric_lang_cantonese"), "Cantonese"),
        (i18n("lyric_lang_english"), "English"),
    ]


def _resolve_file_path(x):
    """Gradio file input can be path string or (path, None) tuple."""
    if x is None:
        return None
    if isinstance(x, tuple):
        x = x[0]
    return x if (x and os.path.isfile(x)) else None


def transcription_function(
    prompt_audio,
    target_audio,
    prompt_metadata,
    target_metadata,
    prompt_lyric_lang: str,
    target_lyric_lang: str,
    prompt_vocal_sep: bool,
    target_vocal_sep: bool,
):
    """Step 1: Run transcription only; output (prompt_meta_path, target_meta_path)."""
    try:
        if isinstance(prompt_audio, tuple):
            prompt_audio = prompt_audio[0]
        if isinstance(target_audio, tuple):
            target_audio = target_audio[0]
        if prompt_audio is None or target_audio is None:
            gr.Warning(message=i18n("warn_missing_audio"))
            return None, None
        prompt_meta_resolved = _resolve_file_path(prompt_metadata)
        target_meta_resolved = _resolve_file_path(target_metadata)
        use_input_metadata = prompt_meta_resolved is not None and target_meta_resolved is not None

        session_base = _session_dir_from_target(target_audio)
        audio_dir = session_base / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        transfer_prompt_path = audio_dir / "prompt.wav"
        transfer_target_path = audio_dir / "target.wav"
        SR = 44100
        PROMPT_MAX_SEC = 30
        TARGET_MAX_SEC = 60
        prompt_audio_data, _ = librosa.load(prompt_audio, sr=SR, mono=True)
        target_audio_data, _ = librosa.load(target_audio, sr=SR, mono=True)
        prompt_audio_data = prompt_audio_data[: PROMPT_MAX_SEC * SR]
        target_audio_data = target_audio_data[: TARGET_MAX_SEC * SR]
        sf.write(transfer_prompt_path, prompt_audio_data, SR)
        sf.write(transfer_target_path, target_audio_data, SR)

        prompt_meta_path = session_base / "transcriptions" / "prompt" / "metadata.json"
        target_meta_path = session_base / "transcriptions" / "target" / "metadata.json"
        if use_input_metadata:
            (session_base / "transcriptions" / "prompt").mkdir(parents=True, exist_ok=True)
            (session_base / "transcriptions" / "target").mkdir(parents=True, exist_ok=True)
            shutil.copy2(prompt_meta_resolved, prompt_meta_path)
            shutil.copy2(target_meta_resolved, target_meta_path)
        else:
            ok, msg = APP_STATE.run_preprocess(
                transfer_prompt_path,
                transfer_target_path,
                session_base,
                prompt_vocal_sep=prompt_vocal_sep,
                target_vocal_sep=target_vocal_sep,
                prompt_lyric_lang=prompt_lyric_lang or "Mandarin",
                target_lyric_lang=target_lyric_lang or "Mandarin",
            )
            if not ok:
                print(msg, file=sys.stderr, flush=True)
                return None, None

        prompt_meta_file = str(prompt_meta_path) if prompt_meta_path.exists() else None
        target_meta_file = str(target_meta_path) if target_meta_path.exists() else None
        return prompt_meta_file, target_meta_file
    except Exception:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None, None


def synthesis_function(
    prompt_audio,
    prompt_metadata,
    target_metadata,
    control: str,
    auto_shift: bool,
    pitch_shift,
    seed: int,
):
    """Step 2: Run SVS from top prompt_audio + prompt_metadata + target_metadata."""
    try:
        if isinstance(prompt_audio, tuple):
            prompt_audio = prompt_audio[0]
        prompt_wav_path = prompt_audio
        prompt_meta_path = _resolve_file_path(prompt_metadata)
        target_meta_path = _resolve_file_path(target_metadata)
        if not prompt_wav_path or not os.path.isfile(prompt_wav_path):
            gr.Warning(message=i18n("warn_missing_synthesis"))
            return None
        if not prompt_meta_path or not os.path.isfile(prompt_meta_path):
            gr.Warning(message=i18n("warn_missing_synthesis"))
            return None
        if not target_meta_path or not os.path.isfile(target_meta_path):
            gr.Warning(message=i18n("warn_missing_synthesis"))
            return None
        if control not in ("melody", "score"):
            control = "score"
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
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return None


def _instruction_md() -> str:
    """Markdown content for the instruction panel (supports links)."""
    return "\n\n".join([
        f"**1.** {i18n('instruction_p1')}",
        f"**2.** {i18n('instruction_p2')}",
        f"**3.** {i18n('instruction_p3')}",
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
        # Auto-detect browser language: run after Gradio mounts
        gr.HTML(
            '<script type="text/javascript">'
            '(function(){'
            'function setLang(){'
            'var lang=(navigator.language||navigator.userLanguage||"").toLowerCase();'
            'if(lang.startsWith("en")){'
            'var inputs=document.querySelectorAll("#lang_choice_radio input");'
            'if(inputs.length>1)inputs[1].click();'
            '}'
            '}'
            'if(document.readyState==="complete")setTimeout(setLang,800);'
            'else window.addEventListener("load",function(){setTimeout(setLang,800);});'
            '})();'
            '</script>',
            visible=False,
        )
        with gr.Row(equal_height=True):
            lang_choice = gr.Radio(
                choices=["中文", "English"],
                value="中文",
                label=i18n("display_lang_label"),
                type="index",
                interactive=True,
                elem_id="lang_choice_radio",
            )

        # Instruction panel (usage workflow); updates on language change
        instruction_md = gr.Markdown(f"### {i18n('instruction_title')}\n\n{_instruction_md()}")

        # Reference examples — at the front of operations (handler registered after components exist)
        skip_clear_metadata_count = gr.State(0)
        with gr.Row():
            _example_choices = [i18n("example_choice_0"), i18n("example_choice_1"), i18n("example_choice_2"), i18n("example_choice_3"), i18n("example_choice_4")]
            example_choice = gr.Dropdown(
                label=i18n("examples_label"),
                choices=_example_choices,
                value=_example_choices[0],
                interactive=True,
            )

        # Step 1: Transcription (audio → metadata)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                prompt_audio = gr.Audio(
                    label=i18n("prompt_audio_label"),
                    type="filepath",
                    editable=False,
                    interactive=True,
                )
            with gr.Column(scale=1):
                target_audio = gr.Audio(
                    label=i18n("target_audio_label"),
                    type="filepath",
                    editable=False,
                    interactive=True,
                )
        with gr.Row(equal_height=True):
            prompt_lyric_lang = gr.Dropdown(
                label=i18n("prompt_lyric_lang_label"),
                choices=get_lyric_lang_choices(),
                value="Mandarin",
                interactive=True,
                scale=1,
            )
            target_lyric_lang = gr.Dropdown(
                label=i18n("target_lyric_lang_label"),
                choices=get_lyric_lang_choices(),
                value="Mandarin",
                interactive=True,
                scale=1,
            )
            prompt_vocal_sep = gr.Checkbox(
                label=i18n("prompt_vocal_sep_label"),
                value=False,
                interactive=True,
                scale=1,
            )
            target_vocal_sep = gr.Checkbox(
                label=i18n("target_vocal_sep_label"),
                value=True,
                interactive=True,
                scale=1,
            )
        with gr.Row():
            transcription_btn = gr.Button(
                value=i18n("transcription_btn_label"),
                variant="primary",
                size="lg",
            )

        # Edit tutorial link (gr.HTML supports links; component labels do not)
        metadata_tutorial_html = gr.HTML(value=i18n("edit_tutorial_html"))
        # Synthesis: params row, then synthesis button on next row
        with gr.Row(equal_height=True):
            prompt_metadata = gr.File(
                label=i18n("prompt_meta_label"),
                type="filepath",
                file_types=[".json"],
                interactive=True,
            )
            target_metadata = gr.File(
                label=i18n("target_meta_label"),
                type="filepath",
                file_types=[".json"],
                interactive=True,
            )
            control_radio = gr.Radio(
                choices=["melody", "score"],
                value="score",
                label=i18n("control_type_label"),
                scale=1,
            )
            auto_shift = gr.Checkbox(
                label=i18n("auto_shift_label"),
                value=True,
                interactive=True,
                scale=1,
            )
            pitch_shift = gr.Number(
                label=i18n("pitch_shift_label"),
                value=0,
                minimum=-36,
                maximum=36,
                step=1,
                interactive=True,
                scale=1,
            )
            seed_input = gr.Number(
                label=i18n("seed_label"),
                value=12306,
                step=1,
                interactive=True,
                scale=1,
            )
        with gr.Row():
            synthesis_btn = gr.Button(
                value=i18n("synthesis_btn_label"),
                variant="primary",
                size="lg",
            )
        with gr.Row():
            output_audio = gr.Audio(
                label=i18n("generated_audio_label"),
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
            global global_lang
            global_lang = ["zh", "en"][lang]
            choices = get_lyric_lang_choices()
            return [
                gr.update(label=i18n("prompt_audio_label")),
                gr.update(label=i18n("target_audio_label")),
                gr.update(label=i18n("prompt_lyric_lang_label"), choices=choices),
                gr.update(label=i18n("target_lyric_lang_label"), choices=choices),
                gr.update(label=i18n("prompt_vocal_sep_label")),
                gr.update(label=i18n("target_vocal_sep_label")),
                gr.update(value=i18n("transcription_btn_label")),
                gr.update(label=i18n("prompt_meta_label")),
                gr.update(label=i18n("target_meta_label")),
                gr.update(value=i18n("edit_tutorial_html")),
                gr.update(label=i18n("control_type_label")),
                gr.update(label=i18n("auto_shift_label")),
                gr.update(label=i18n("pitch_shift_label")),
                gr.update(label=i18n("seed_label")),
                gr.update(value=i18n("synthesis_btn_label")),
                gr.update(label=i18n("generated_audio_label")),
                gr.update(label=i18n("display_lang_label")),
                gr.update(
                    label=i18n("examples_label"),
                    choices=[i18n("example_choice_0"), i18n("example_choice_1"), i18n("example_choice_2"), i18n("example_choice_3"), i18n("example_choice_4")],
                    value=i18n("example_choice_0"),
                ),
                gr.update(value=f"### {i18n('instruction_title')}\n\n{_instruction_md()}"),
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
                transcription_btn,
                prompt_metadata,
                target_metadata,
                metadata_tutorial_html,
                control_radio,
                auto_shift,
                pitch_shift,
                seed_input,
                synthesis_btn,
                output_audio,
                lang_choice,
                example_choice,
                instruction_md,
            ],
        )

        # Upload new prompt/target audio → clear corresponding metadata; skip clear when change came from load example
        prompt_audio.change(
            fn=_clear_prompt_meta_unless_example,
            inputs=[prompt_audio, skip_clear_metadata_count],
            outputs=[prompt_metadata, skip_clear_metadata_count],
        )
        target_audio.change(
            fn=_clear_target_meta_unless_example,
            inputs=[target_audio, skip_clear_metadata_count],
            outputs=[target_metadata, skip_clear_metadata_count],
        )

        transcription_btn.click(
            fn=transcription_function,
            inputs=[
                prompt_audio,
                target_audio,
                prompt_metadata,
                target_metadata,
                prompt_lyric_lang,
                target_lyric_lang,
                prompt_vocal_sep,
                target_vocal_sep,
            ],
            outputs=[prompt_metadata, target_metadata],
        )

        synthesis_btn.click(
            fn=synthesis_function,
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
