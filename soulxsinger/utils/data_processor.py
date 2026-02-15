import json
import torch
import numpy as np
import torchaudio
from typing import List

from soulxsinger.utils.audio_utils import load_wav


class DataProcessor:
    """Data processor for SoulX-Singer
    """
    def __init__(
                self, 
                hop_size: int, 
                sample_rate: int, 
                phoneset_path: str = 'soulxsinger/utils/phoneme/phone_set.json',
                device: str = 'cuda',
                prompt_append_duration: float = 0.5):
        """Initialize data processor.

        Args:
            hop_size (int): Hop size in samples.
            sample_rate (int): Sample rate in Hz.
            phoneset_path (str): Path to phoneme set JSON file.
            device (str): Device to use for tensor operations.
            prompt_append_duration (float): Duration to append to prompt in seconds.
        """
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.device = device
        self.prompt_append_duration = prompt_append_duration
        self.prompt_append_length = int(prompt_append_duration * sample_rate / hop_size)
        self.load_phoneme_id_map(phoneset_path)

    def load_phoneme_id_map(self, phoneset_path: str):
        with open(phoneset_path, "r", encoding='utf-8') as f:
            phoneset = json.load(f)
        self.phone2idx = {ph: idx for idx, ph in enumerate(phoneset)}
    
    def merge_phoneme(self, meta):
        merged_items = []

        duration = [float(x) for x in meta["duration"].split()]
        phoneme = [str(x).replace("<AP>", "<SP>") for x in meta["phoneme"].split()]
        note_pitch = [int(x) for x in meta["note_pitch"].split()]
        note_type = [int(x) if phoneme[i] != "<SP>" else 1 for i, x in enumerate(meta["note_type"].split())]

        for i in range(len(phoneme)):
            if i > 0 and phoneme[i] == phoneme[i - 1] == "<SP>" and note_type[i] == note_type[i - 1] and note_pitch[i] == note_pitch[i - 1]:
                merged_items[-1][1] += duration[i]
            else:
                merged_items.append([phoneme[i], duration[i], note_pitch[i], note_type[i]])

        meta['phoneme'] =  [x[0] for x in merged_items]
        meta['duration'] = [x[1] for x in merged_items]
        meta['note_pitch'] = [x[2] for x in merged_items]
        meta['note_type'] = [x[3] for x in merged_items]

        return meta
    
    def preprocess(
        self,
        note_duration: List[float],
        phonemes: List[str],
        note_pitch: List[int],
        note_type: List[int],
    ):
        """
        Insert <BOW> and <EOW> for each note. 
        Get aligned indices for each frame.

        Args:
            note_duration: Duration of each note in seconds
            phonemes: Phoneme sequence for each note
            note_pitch: Pitch value for each note
            note_type: Type value for each note
        
        """
        sample_rate = self.sample_rate
        hop_size = self.hop_size
        duration = sum(note_duration) * sample_rate / hop_size
        mel2note = torch.zeros(int(duration), dtype=torch.long)

        ph_locations = []   # idx at mel scale and length
        new_phonemes = []
        dur_sum = 0

        note2origin = []

        for ph_idx in range(len(phonemes)):
            dur = int(np.round(dur_sum * sample_rate / hop_size))
            dur = min(dur, len(mel2note) - 1)
            new_phonemes.append("<BOW>")
            note2origin.append(ph_idx)
            if phonemes[ph_idx][:3] == "en_":
                en_phs = ['en_' + x for x in phonemes[ph_idx][3:].split('-')] + ['<SEP>']     # <sep> between en words in one note
                ph_locations.append([dur, max(1, len(en_phs))])
                new_phonemes.extend(en_phs)
                note2origin.extend([ph_idx] * len(en_phs))
            else:
                ph_locations.append([dur, 1])
                new_phonemes.append(phonemes[ph_idx])
                note2origin.append(ph_idx)
            new_phonemes.append("<EOW>")
            note2origin.append(ph_idx)
            dur_sum += note_duration[ph_idx]

        ph_idx = 1
        for idx, (i, j) in enumerate(ph_locations):
            next_phoneme_start = ph_locations[idx + 1][0] if idx < len(ph_locations) - 1 else len(mel2note)
            if i >= len(mel2note) or i + j > len(mel2note):
                break
            if i < len(mel2note) and mel2note[i] > 0:
                # print(f"warning: overlap of {idx}: {mel2note[i]}")
                while i < len(mel2note) and mel2note[i] > 0:
                    i += 1
            mel2note[i] = ph_idx
            k = i + 1
            while k + j < next_phoneme_start:
                mel2note[k : k + j] = torch.arange(ph_idx, ph_idx + j) + 1
                k += j
            mel2note[next_phoneme_start - 1] = ph_idx + j + 1
            ph_idx += j + 2     # <BOW> + ph repeats + <EOW>

        new_phonemes = ["<PAD>"] + new_phonemes
        new_note_pitch = [0] + [note_pitch[k] for k in note2origin]
        new_note_type = [1] + [note_type[k] for k in note2origin]

        return {
            "phoneme": torch.tensor([self.phone2idx[x] for x in new_phonemes], device=self.device).unsqueeze(0),
            "note_pitch": torch.tensor(new_note_pitch, device=self.device).unsqueeze(0),
            "note_type": torch.tensor(new_note_type, device=self.device).unsqueeze(0),
            "mel2note": mel2note.clone().detach().to(self.device).unsqueeze(0),
        }
    
    def process(
        self,
        meta: dict,
        wav_path: str = None
    ):

        meta = self.merge_phoneme(meta)
        
        item = self.preprocess(
            meta["duration"],
            meta["phoneme"],
            meta["note_pitch"],
            meta["note_type"],
        )

        f0 = [float(x) for x in meta.get("f0", "").split()]
        min_frame = min(item["mel2note"].shape[1], len(f0)) if len(f0) > 0 else item["mel2note"].shape[1]
        item['f0'] = torch.tensor(f0)[:min_frame].unsqueeze(0).float().to(self.device) if len(f0) > 0 else None
        item["mel2note"] = item["mel2note"][:, :min_frame]

        if wav_path is not None:
            waveform = load_wav(wav_path, self.sample_rate)
            item["waveform"] = waveform.to(self.device)[:, :min_frame * self.hop_size]
        
        return item


# test
if __name__ == "__main__":
    import json
    with open("example/metadata/zh_prompt.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, list):
        meta = meta[0]
    processor = DataProcessor(hop_size=480, sample_rate=24000)
    item = processor.process(meta, "example/audio/zh_prompt.wav")
    print(item.keys())