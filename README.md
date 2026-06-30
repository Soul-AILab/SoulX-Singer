<div align="center">
  <h1>🎤 SoulX-Singer</h1>
  <p>
    Official inference code for<br>
    <b><em>SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis</em></b>
  </p>
  <p>
    <img src="assets/soulx-logo.png" alt="SoulX-Logo" style="height:85px;">
  </p>
  <p>
    <a href="https://soul-ailab.github.io/soulx-singer/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="Demo Page"></a>
    <a href="https://huggingface.co/spaces/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Online%20Demo-ffda16" alt="HF Space Demo"></a>
    <a href="https://huggingface.co/Soul-AILab/SoulX-Singer"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF-model"></a>
    <a href="assets/technical-report.pdf"><img src="https://img.shields.io/badge/Report-Github-red" alt="Technical Report"></a>
    <a href="https://arxiv.org/abs/2602.07803"><img src="https://img.shields.io/badge/arXiv-2602.07803-b31b1b" alt="arXiv"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"></a>
  </p>
</div>

---

## 🎵 Overview

**SoulX-Singer** is a high-fidelity, zero-shot singing voice synthesis model that enables users to generate realistic singing voices for unseen singers. It supports **melody-conditioned (F0 contour)** and **score-conditioned (MIDI notes)** control for precise pitch, rhythm, and expression.

**SoulX-Singer-SVC** is a singing voice conversion (SVC) model finetuned from **SoulX-Singer**. Singing Voice Conversion aims to transform a source singing recording into the target singer’s voice while preserving the original melody, rhythm, and lyrical content. Based on the strong generative capability of SoulX-Singer, SoulX-Singer-SVC enables high-quality singing voice conversion directly from raw singing audio, without requiring lyric or MIDI transcriptions.

---

## ✨ Key Features

#### SoulX-Singer
- **🎤 Zero-Shot Singing** – Generate high-fidelity voices for unseen singers, no fine-tuning needed.  
- **🎵 Flexible Control Modes** – Melody (F0) and Score (MIDI) conditioning.  
- **📚 Large-Scale Dataset** – 42,000+ hours of aligned vocals, lyrics, notes across Mandarin, English, Cantonese.  
- **🧑‍🎤 Timbre Cloning** – Preserve singer identity across languages, styles, and edited lyrics.  
- **✏️ Singing Voice Editing** – Modify lyrics while keeping natural prosody.  
- **🌐 Cross-Lingual Synthesis** – High-fidelity synthesis by disentangling timbre from content.  

#### SoulX-Singer-SVC
- **🎙️ Zero-Shot Timbre and Style Transfer** – Transfer singer identity and style to unseen voices without per-speaker fine-tuning.
- **🌍 Language-Agnostic Conversion** – Works across multilingual singing content.
- **🔄 Transcription-Free Audio-to-Audio Conversion** – Convert target singing directly without lyrics transcription or MIDI inputs.

#### Smart Auto Prompt V2 (Local Customization)
- **📊 Data-Driven Prompt Extraction** – Significantly improved the automatic prompt extraction algorithm by profiling performance metrics.
- **🎯 F0 Stability Scoring** – Re-engineered the scoring logic to prioritize stable pitch segments (penalizing high F0 variance and range), which dramatically reduces F0 Frame Errors (FFE) and Word Error Rates (WER).
- **⏱️ Optimized Segment Selection** – Introduced a hard threshold for voiced ratio (>60%) and increased the minimum segment length to 3.5 seconds to preserve better breathing and natural prosody for robust voice cloning.

---

<p align="center">
  <img src="assets/performance_radar.png" width="80%" alt="Performance Radar"/>
</p>

---

## 🎬 Demo Examples

### Singing Voice Synthesis (SVS)
<div align="center">

<https://github.com/user-attachments/assets/13306f10-3a29-46ba-bcef-d6308d05cbcc>

</div>
<div align="center">

<https://github.com/user-attachments/assets/2eb260fe-6f0b-408c-aab8-5b81ddddb284>

</div>

### Singing Voice Conversion (SVC)
<div align="center">

<https://github.com/user-attachments/assets/aed15fc9-14c3-44fc-9146-f6d9fef894d3>

</div>

---

## 📰 News
- **[2026-03-16]** [SoulX-Singer-SVC](https://huggingface.co/Soul-AILab/SoulX-Singer/blob/main/model-svc.pt) is released, and [SoulX-Singer Online Demo](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer) has been updated to support singing voice conversion (SVC).
- **[2026-02-12]** [SoulX-Singer Eval Dataset](https://huggingface.co/datasets/Soul-AILab/SoulX-Singer-Eval-Dataset) is now available on Hugging Face Datasets.
- **[2026-02-09]** [SoulX-Singer Online Demo](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer) is live on Hugging Face Spaces — try singing voice synthesis in your browser.
- **[2026-02-08]** [MIDI Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor) is available on Hugging Face Spaces.
- **[2026-02-06]** SoulX-Singer inference code and models released.

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Soul-AILab/SoulX-Singer.git
cd SoulX-Singer
```

### 2. Set Up Environment

**1. Install Conda** (if not already installed): https://docs.conda.io/en/latest/miniconda.html

**2. Create and activate a Conda environment:**
```
conda create -n soulxsinger -y python=3.10
conda activate soulxsinger
```
**3. Install dependencies:**
```
pip install -r requirements.txt
```
⚠️ If you are in mainland China, use a PyPI mirror:
```
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```


---

### 3. Download Pretrained Models

Install Hugging Face Hub if needed:

```
pip install -U huggingface_hub
```

Download the SVS, SVC model and preprocessing models:
```sh
pip install -U huggingface_hub

# Download the SoulX-Singer SVS and SVC model
hf download Soul-AILab/SoulX-Singer --local-dir pretrained_models/SoulX-Singer

# Download models required for preprocessing
hf download Soul-AILab/SoulX-Singer-Preprocess --local-dir pretrained_models/SoulX-Singer-Preprocess
```


### 4. Run the Demo

#### Run the SVS inference demo
``` sh
bash example/infer.sh
```

This script relies on metadata generated from the preprocessing pipeline, including vocal separation and transcription. Users should follow the steps in [preprocess](preprocess/README.md) to prepare the necessary metadata before running the demo with their own data.

**⚠️ Important Note**
The metadata produced by the automatic preprocessing pipeline may not perfectly align the singing audio with the corresponding lyrics and musical notes. For best synthesis quality, we strongly recommend manually correcting the alignment using the 🎼 [Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor). 

How to use the Midi-Editor:
- [Eiditing Metadata with Midi-Editor](preprocess/README.md#L104-L105)

#### Run the SVC inference demo

```sh
bash example/infer_svc.sh
```

This example performs audio-to-audio SVC, converting the target singing into the prompt timbre using waveform and F0 inputs.
To prepare your own SVC data, run `example/preprocess.sh` with `midi_transcribe=False`.



### 🌐 WebUI

You can launch the interactive interface for SVS (Synthesised from lyrics and MIDI transcriptions) with:
```
python webui.py
```

For SVC WebUI (audio-to-audio conversion):

```
python webui_svc.py
```



## 🚧 Roadmap

- [x] 🖥️ Web-based UI for easy and interactive inference  
- [x] 🌐 Online MIDI Editor deployment on Hugging Face Spaces
- [x] 🌐 Online demo deployment on Hugging Face Spaces  
- [x] 📊 Release the SoulX-Singer-Eval benchmark  
- [ ] 🎹 Inference support for user-friendly MIDI-based input
- [ ] 📚 Comprehensive tutorials and usage documentation  
- [x] 🎵 Support for wav-to-wav singing voice conversion (without transcription)


## 🙏 Acknowledgements

Special thanks to the following open-source projects:

- [F5-TTS](https://github.com/SWivid/F5-TTS)
- [Amphion](https://github.com/open-mmlab/Amphion/tree/main)
- [Music Source Separation Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)
- [Lead Vocal Separation](https://huggingface.co/becruily/mel-band-roformer-karaoke)
- [Vocal Dereverberation](https://huggingface.co/anvuew/dereverb_mel_band_roformer)
- [RMVPE](https://github.com/Dream-High/RMVPE)
[Paraformer](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
- [Parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [ROSVOT](https://github.com/RickyL-2000/ROSVOT)



## 📄 License

We use the Apache 2.0 license. Researchers and developers are free to use the codes and model weights of our SoulX-Singer. Check the license at [LICENSE](LICENSE) for more details.


##  ⚠️ Usage Disclaimer

SoulX-Singer is intended for academic research, educational purposes, and legitimate applications such as personalized singing synthesis and assistive technologies.

Please note:

- 🎤 Respect intellectual property, privacy, and personal consent when generating singing content.
- 🚫 Do not use the model to impersonate individuals without authorization or to create deceptive audio.
- ⚠️ The developers assume no liability for any misuse of this model.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles. For ethics or misuse concerns, please contact us.


## 📄 Citation

If you use SoulX-Singer in your research, please cite:

```bibtex
@misc{soulxsinger,
      title={SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis}, 
      author={Jiale Qian and Hao Meng and Tian Zheng and Pengcheng Zhu and Haopeng Lin and Yuhang Dai and Hanke Xie and Wenxiao Cao and Ruixuan Shang and Jun Wu and Hongmei Liu and Hanlin Wen and Jian Zhao and Zhonglin Jiang and Yong Chen and Shunshun Yin and Ming Tao and Jianguo Wei and Lei Xie and Xinsheng Wang},
      year={2026},
      eprint={2602.07803},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2602.07803}, 
}
```


## 📬 Contact Us

We welcome your feedback, questions, and collaboration:

- **Email**: qianjiale@soulapp.cn | menghao@soulapp.cn | wangxinsheng@soulapp.cn

- **Join discussions**: WeChat or Soul APP groups for technical discussions and updates:

<p align="center">
  <!-- <em>Due to group limits, if you can't scan the QR code, please add my WeChat for group access  -->
      <!-- : <strong>Tiamo James</strong></em> -->
  <br>
  <span style="display: inline-block; margin-right: 10px;">
    <img src="assets/soul_wechat01.jpg" width="500" alt="WeChat Group QR Code"/>
  </span>
  <!-- <span style="display: inline-block;">
    <img src="assets/wechat_tiamo.jpg" width="300" alt="WeChat QR Code"/>
  </span> -->
</p>