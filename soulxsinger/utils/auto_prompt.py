import torch
import torchaudio
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class AutoPromptExtractor:
    def __init__(self, target_length_sec: float = 15.0, crossfade_ms: int = 50, min_segment_sec: float = 3.0):
        """
        Initialize the AutoPromptExtractor.
        Args:
            target_length_sec: Target maximum length of the combined prompt in seconds.
            crossfade_ms: Duration of crossfade in milliseconds to prevent popping noise.
            min_segment_sec: Minimum duration of a single segment in seconds to preserve prosody.
        """
        self.target_length_sec = target_length_sec
        self.crossfade_ms = crossfade_ms
        self.min_segment_sec = min_segment_sec

    def _split_into_segments(self, wav_tensor: torch.Tensor, sample_rate: int, energy_threshold: float = 0.01) -> list[torch.Tensor]:
        """
        Split the audio tensor into segments based on silence (RMS energy).
        Args:
            wav_tensor: 1D or 2D audio tensor of shape (channels, length)
            sample_rate: Sample rate of the audio
            energy_threshold: RMS energy threshold below which it's considered silence
        Returns:
            A list of audio tensors, each representing a valid vocal segment.
        """
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
            
        # Calculate RMS energy in frames
        frame_length = int(sample_rate * 0.05) # 50ms frames
        hop_length = int(sample_rate * 0.025) # 25ms hop
        
        # Calculate squared signal, then unfold to frames
        squared_sig = wav_tensor.pow(2).squeeze(0)
        
        # Pad to avoid dropping the end
        pad_size = hop_length - (squared_sig.shape[0] % hop_length)
        if pad_size > 0:
            squared_sig = torch.nn.functional.pad(squared_sig, (0, pad_size))
            
        frames = squared_sig.unfold(0, frame_length, hop_length)
        rms_energy = torch.sqrt(frames.mean(dim=1))
        
        # Create voice activity mask
        is_voiced = rms_energy > energy_threshold
        
        # Find continuous voiced regions
        segments = []
        in_segment = False
        start_frame = 0
        
        for i, voiced in enumerate(is_voiced):
            if voiced and not in_segment:
                start_frame = i
                in_segment = True
            elif not voiced and in_segment:
                end_frame = i
                in_segment = False
                
                # Convert frame indices to samples
                start_sample = start_frame * hop_length
                end_sample = end_frame * hop_length
                segment_sec = (end_sample - start_sample) / sample_rate
                
                segments.append({
                    "start": start_sample,
                    "end": end_sample,
                    "duration": segment_sec
                })
                
        # Handle the case where audio ends while voiced
        if in_segment:
            start_sample = start_frame * hop_length
            end_sample = wav_tensor.shape[1]
            segment_sec = (end_sample - start_sample) / sample_rate
            segments.append({
                "start": start_sample,
                "end": end_sample,
                "duration": segment_sec
            })
            
        logger.debug(f"[_split_into_segments] Found {len(segments)} raw segments before merging.")
            
        # Merge segments to satisfy minimum length constraint
        merged_segments = []
        current_segment = None
        
        for seg in segments:
            if current_segment is None:
                current_segment = seg
            else:
                # Merge current_segment with seg (including the silence in between to keep flow)
                current_segment["end"] = seg["end"]
                current_segment["duration"] = (current_segment["end"] - current_segment["start"]) / sample_rate
            
            # If the merged segment satisfies the min duration, add to list and reset
            if current_segment["duration"] >= self.min_segment_sec:
                merged_segments.append(current_segment)
                current_segment = None
                
        # If there's a leftover segment smaller than min_segment_sec, try merging it with the last valid segment
        if current_segment is not None:
            if merged_segments:
                merged_segments[-1]["end"] = current_segment["end"]
                merged_segments[-1]["duration"] = (merged_segments[-1]["end"] - merged_segments[-1]["start"]) / sample_rate
            else:
                # If it's the only segment, just append it even if it's short
                merged_segments.append(current_segment)
                
        avg_dur = sum(s['duration'] for s in merged_segments)/len(merged_segments) if merged_segments else 0
        logger.info(f"[_split_into_segments] Final merged segments: {len(merged_segments)} (Avg duration: {avg_dur:.2f}s)")
                
        # Instead of extracting tensor chunks here, we return the segment dictionaries
        # so that downstream methods can slice both wav and f0 tensors.
        return merged_segments

    def _calculate_segment_scores(self, segments: list[dict], wav_tensor: torch.Tensor, f0_tensor: torch.Tensor, sample_rate: int, f0_rate: int = 50):
        """
        Calculate acoustic and phonetic scores for each segment.
        """
        global_f0 = f0_tensor[f0_tensor > 0]
        global_f0_median = torch.median(global_f0) if len(global_f0) > 0 else torch.tensor(0.0)
        logger.info(f"[_calculate_segment_scores] Global F0 Median calculated: {global_f0_median.item():.2f} Hz")
        
        scored_segments = []
        for seg in segments:
            start_f0_idx = int(seg["start"] / sample_rate * f0_rate)
            end_f0_idx = int(seg["end"] / sample_rate * f0_rate)
            
            seg_f0 = f0_tensor[..., start_f0_idx:end_f0_idx]
            voiced_f0 = seg_f0[seg_f0 > 0]
            
            total_f0_frames = max(1, end_f0_idx - start_f0_idx)
            voiced_ratio = len(voiced_f0) / total_f0_frames
            
            if len(voiced_f0) > 0:
                seg_f0_median = torch.median(voiced_f0)
                seg_f0_var = torch.var(voiced_f0) if len(voiced_f0) > 1 else torch.tensor(0.0)
                f0_distance = abs(seg_f0_median.item() - global_f0_median.item())
            else:
                seg_f0_var = torch.tensor(0.0)
                f0_distance = 9999.0
                
            seg_wav = wav_tensor[..., seg["start"]:seg["end"]]
            frame_length = int(sample_rate * 0.05)
            hop_length = int(sample_rate * 0.025)
            if seg_wav.shape[-1] >= frame_length:
                frames = seg_wav.pow(2).squeeze(0).unfold(0, frame_length, hop_length)
                rms = torch.sqrt(frames.mean(dim=1))
                rms_var = torch.var(rms).item()
            else:
                rms_var = 0.0
                
            seg_copy = seg.copy()
            seg_copy.update({
                "voiced_ratio": voiced_ratio,
                "f0_var": seg_f0_var.item() if isinstance(seg_f0_var, torch.Tensor) else seg_f0_var,
                "rms_var": rms_var,
                "f0_distance": f0_distance
            })
            scored_segments.append(seg_copy)
            
        def normalize(key):
            vals = [s[key] for s in scored_segments]
            if not vals: return
            min_v, max_v = min(vals), max(vals)
            if max_v > min_v:
                for s in scored_segments:
                    s[f"{key}_norm"] = (s[key] - min_v) / (max_v - min_v)
            else:
                for s in scored_segments:
                    s[f"{key}_norm"] = 0.5
                    
        normalize("f0_var")
        normalize("rms_var")
        normalize("voiced_ratio")
        
        for s in scored_segments:
            # Weighted sum: prefer high dynamic variance and high voice density
            s["score"] = s.get("f0_var_norm", 0) * 0.4 + s.get("rms_var_norm", 0) * 0.4 + s.get("voiced_ratio_norm", 0) * 0.2
            logger.debug(f"[_calculate_segment_scores] Seg {s['start']}~{s['end']}: voiced_ratio={s.get('voiced_ratio',0):.2f}, rms_var={s.get('rms_var',0):.2f}, f0_diff={s.get('f0_distance',0):.2f} -> Score={s['score']:.4f}")
            
        return scored_segments, global_f0_median.item()

    def _find_best_combination(self, scored_segments: list[dict], global_f0_median: float):
        """
        Find the best combination of segments that fits within the target length.
        """
        # Sort by score descending
        scored_segments.sort(key=lambda x: x["score"], reverse=True)
        
        chosen = []
        current_len = 0.0
        
        for seg in scored_segments:
            if current_len + seg["duration"] <= self.target_length_sec:
                # Constraint: F0 median of segment should not deviate more than ~15% from global median
                # (roughly 2.5 semitones)
                if seg["f0_distance"] < (global_f0_median * 0.15):
                    chosen.append(seg)
                    current_len += seg["duration"]
                    
        # Restore chronological order
        chosen.sort(key=lambda x: x["start"])
        
        avg_dist = sum(c["f0_distance"] for c in chosen)/len(chosen) if chosen else 0
        logger.info(f"[_find_best_combination] Selected {len(chosen)} segments. Total duration: {current_len:.2f}s, Avg F0 Deviation: {avg_dist:.2f} Hz")
        return chosen

    def _concatenate_with_crossfade(self, chosen_segments: list[dict], wav_tensor: torch.Tensor, f0_tensor: torch.Tensor, sample_rate: int, f0_rate: int = 50):
        """
        Concatenate the chosen segments applying a crossfade at the boundaries.
        Also concatenate F0 appropriately.
        """
        if not chosen_segments:
            return torch.zeros((1, int(sample_rate * 3)), device=wav_tensor.device), f0_tensor
            
        crossfade_samples = int(sample_rate * self.crossfade_ms / 1000)
        crossfade_f0_frames = int(f0_rate * self.crossfade_ms / 1000)
        
        final_wav = None
        final_f0 = None
        
        for seg in chosen_segments:
            seg_wav = wav_tensor[:, seg["start"]:seg["end"]]
            start_f0 = int(seg["start"] / sample_rate * f0_rate)
            end_f0 = int(seg["end"] / sample_rate * f0_rate)
            seg_f0 = f0_tensor[:, start_f0:end_f0]
            
            if final_wav is None:
                final_wav = seg_wav
                final_f0 = seg_f0
            else:
                cf_len = min(crossfade_samples, final_wav.shape[1] // 2, seg_wav.shape[1] // 2)
                cf_f0_len = min(crossfade_f0_frames, final_f0.shape[1] // 2, seg_f0.shape[1] // 2)
                
                logger.debug(f"[_concatenate_with_crossfade] Applying crossfade: {cf_len} samples, {cf_f0_len} F0 frames.")
                
                if cf_len > 0:
                    fade_out = final_wav[:, -cf_len:] * torch.linspace(1, 0, cf_len, device=wav_tensor.device)
                    fade_in = seg_wav[:, :cf_len] * torch.linspace(0, 1, cf_len, device=wav_tensor.device)
                    
                    mixed = fade_out + fade_in
                    final_wav = torch.cat([
                        final_wav[:, :-cf_len],
                        mixed,
                        seg_wav[:, cf_len:]
                    ], dim=1)
                    
                    if cf_f0_len > 0:
                        mixed_f0 = torch.max(final_f0[:, -cf_f0_len:], seg_f0[:, :cf_f0_len])
                        final_f0 = torch.cat([
                            final_f0[:, :-cf_f0_len],
                            mixed_f0,
                            seg_f0[:, cf_f0_len:]
                        ], dim=1)
                    else:
                        final_f0 = torch.cat([final_f0, seg_f0], dim=1)
                else:
                    final_wav = torch.cat([final_wav, seg_wav], dim=1)
                    final_f0 = torch.cat([final_f0, seg_f0], dim=1)
                    
        logger.info(f"[_concatenate_with_crossfade] Concatenation complete. Final wav samples: {final_wav.shape[1] if final_wav is not None else 0}, Final F0 frames: {final_f0.shape[1] if final_f0 is not None else 0}")
        return final_wav, final_f0

    def extract(self, wav_tensor: torch.Tensor, f0_tensor: torch.Tensor, sample_rate: int, f0_rate: int = 50):
        """
        Main entrypoint to extract and combine the best prompt from the given audio.
        """
        original_sec = wav_tensor.shape[-1] / sample_rate
        logger.info("--- AutoPromptExtractor started ---")
        logger.info(f"[extract] Original Audio Length: {original_sec:.2f}s, F0 Frames: {f0_tensor.shape[-1]}")
        
        segments = self._split_into_segments(wav_tensor, sample_rate)
        if not segments:
            return wav_tensor, f0_tensor # fallback
            
        scored_segments, global_f0_median = self._calculate_segment_scores(segments, wav_tensor, f0_tensor, sample_rate, f0_rate)
        chosen_segments = self._find_best_combination(scored_segments, global_f0_median)
        
        # fallback if constraint was too strict
        if not chosen_segments and scored_segments:
            chosen_segments = [scored_segments[0]] 
            
        final_wav, final_f0 = self._concatenate_with_crossfade(chosen_segments, wav_tensor, f0_tensor, sample_rate, f0_rate)
        
        final_sec = final_wav.shape[-1] / sample_rate
        logger.info(f"[extract] Extracted Prompt Length: {final_sec:.2f}s")
        logger.info("--- AutoPromptExtractor finished ---")
        
        return final_wav, final_f0
