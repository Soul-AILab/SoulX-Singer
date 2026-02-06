# 🎵 SoulX-Singer-Preprocess

This part offers a comprehensive **singing transcription and editing toolkit** for real-world music audio. It provides the pipeline from vocal extraction to high-level annotation optimized for SVS dataset construction. By integrating state-of-the-art models, it transforms raw audio into structured singing data and supports the **customizable creation and editing of lyric-aligned MIDI scores**.


## ✨ Features

The toolkit includes the following core modules:

- 🎤 **Clean Dry Vocal Extraction**  
  Extracts the lead vocal track from polyphonic music audio and dereverberation.

- 📝 **Lyrics Transcription**  
  Automatically transcribes lyrics from clean vocal.

- 🎶 **Note Transcription**  
  Converts singing voice into note-level representations for SVS.

- 🎼 **MIDI Editor**  
  Supports customizable creation and editing of MIDI scores integrated with lyrics.


## 🔧 Python Environment

Before running the pipeline, set up the Python environment as follows:

1. **Install Conda** (if not already installed): https://docs.conda.io/en/latest/miniconda.html

2. **Activate or create a conda environment** (recommended Python 3.10):

   - If you already have the `soulxsinger` environment:

     ```bash
     conda activate soulxsinger
     ```

   - Otherwise, create it first:

     ```bash
     conda create -n soulxsinger -y python=3.10
     conda activate soulxsinger
     ```

3. **Install dependencies** from the `preprocess` directory:

   ```bash
   cd preprocess
   pip install -r requirements.txt
   ```

## 📁 Data Preparation

Before running the pipeline, prepare the following inputs:

- **Prompt audio**  
  Reference audio that provides timbre and style

- **Target audio**  
  Original vocal or music audio to be processed and transcribed.

Configure the corresponding parameters in:

```
example/preprocess.sh
```

Typical configuration includes:
- Input / output paths
- Module enable switches

## 🚀 Usage

After configuring `preprocess.sh`, run the transcription pipeline with:

```bash
bash example/preprocess.sh
```

The script will automatically execute the following steps:

1. **Vocal separation and dereverberation**
2. **F0 extraction and voice activity detection (VAD)**
3. **Lyrics transcription**
4. **Note transcription**

---

After the pipeline completes, you will obtain **SoulX-Singer–style metadata** that can be directly used for Singing Voice Synthesis (SVS).

**Output paths:**
- The final metadata (**JSON file**) is written **in the same directory as your input audio**, with the **same filename** (e.g. `audio.mp3` → `audio.json`)
- All **intermediate results** (separated vocal and accompaniment, F0, VAD outputs, etc.) are also saved under the configured **`save_dir`**.

⚠️ **Important Note**

Transcription errors—especially in **lyrics** and **note annotations**—can significantly affect the final SVS quality. We **strongly recommend manually reviewing and correcting** the generated metadata before inference.

To support this, we provide a **MIDI Editor** for editing lyrics, phoneme alignment, note pitches, and durations. The workflow is:

**Export metadata to MIDI** → edit in the MIDI Editor → **Import edited MIDI back to metadata** for SVS.

---

#### Step 1: Metadata → MIDI (for editing)

Convert SoulX-Singer metadata to a MIDI file so you can open it in the MIDI Editor:

```bash
preprocess_root=example/transcriptions/music

python -m preprocess.tools.midi_parser \
    --meta2midi \
    --meta "${preprocess_root}/metadata.json" \
    --midi "${preprocess_root}/vocal.mid"
```

#### Step 2: Edit in the MIDI Editor

Open the MIDI Editor (see [MIDI Editor Tutorial](tools/midi_editor/README.md)), load `vocal.mid`, and correct lyrics, pitches, or durations as needed. Save the result as e.g. `vocal_edited.mid`.

#### Step 3: MIDI → Metadata (for SoulX-Singer inference)

Convert the edited MIDI back into SoulX-Singer-style metadata (and cut wavs) for SVS:

```bash
python -m preprocess.tools.midi_parser \
    --midi2meta \
    --midi "${preprocess_root}/vocal_edited.mid" \
    --meta "${preprocess_root}/edit_metadata.json" \
    --vocal "${preprocess_root}/vocal.wav" \
```

Use `edit_metadata.json` (and the wavs under `edit_cut_wavs`) as the target metadata in your inference pipeline.


## 🔗 References & Dependencies

This project builds upon the following excellent open-source works:

### 🎧 Vocal Separation & Dereverberation
- [Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [Lead Vocal Separation](https://huggingface.co/becruily/mel-band-roformer-karaoke)
- [Vocal Dereverberation](https://huggingface.co/anvuew/dereverb_mel_band_roformer)

### 🎼 F0 Extraction
- [RMVPE](https://github.com/Dream-High/RMVPE)

### 📝 Lyrics Transcription (ASR)
- [Paraformer](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
- [Parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)

### 🎶 Note Transcription
- [ROSVOT](https://github.com/RickyL-2000/ROSVOT)

We sincerely thank the authors of these repositories for their exceptional open-source contributions, which have been fundamental to the development of this toolkit.
