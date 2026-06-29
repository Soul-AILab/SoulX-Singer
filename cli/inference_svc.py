import os
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
from collections import OrderedDict
from omegaconf import DictConfig

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
from soulxsinger.utils.audio_utils import load_wav
from soulxsinger.utils.auto_prompt import AutoPromptExtractor


def build_model(
    model_path: str,
    config: DictConfig,
    device: str = "cuda",
    use_fp16: bool = False,
):
    """
    Build the model from the pre-trained model path and model configuration.

    Args:
        model_path (str): Path to the checkpoint file.
        config (DictConfig): Model configuration.
        device (str, optional): Device to use. Defaults to "cuda".
        use_fp16 (bool, optional): If True and device is CUDA, convert model to FP16 after load. Defaults to False.

    Returns:
        SoulXSingerSVC: The initialized model.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Please download the pretrained model and place it at the path, or set --model_path."
        )
    model = SoulXSingerSVC(config).to(device)
    print("Model initialized.")
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {model_path} has no 'state_dict' key. "
            "Expected a checkpoint saved with model.state_dict()."
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    if use_fp16 and ((isinstance(device, str) and device.startswith("cuda")) or (hasattr(device, "type") and getattr(device, "type", None) == "cuda")):
        model.half()
        model.mel.float()
        print("Model converted to FP16 (mel kept in FP32).")
    print("Model checkpoint loaded.")
    model.eval()
    model.to(device)

    return model


def process(args, config, model: torch.nn.Module):
    """Run the full inference pipeline given a data_processor and model.
    """

    os.makedirs(args.save_dir, exist_ok=True)
    pt_wav = load_wav(args.prompt_wav_path, config.audio.sample_rate).to(args.device)
    gt_wav = load_wav(args.target_wav_path, config.audio.sample_rate).to(args.device)
    pt_f0 = torch.from_numpy(np.load(args.prompt_f0_path)).unsqueeze(0).to(args.device)
    gt_f0 = torch.from_numpy(np.load(args.target_f0_path)).unsqueeze(0).to(args.device)

    if getattr(args, "auto_prompt", False):
        print(f"Extracting optimal prompt (max length: {args.auto_prompt_length}s)...")
        extractor = AutoPromptExtractor(target_length_sec=args.auto_prompt_length)
        # Move back to CPU for extraction to avoid memory issues during processing, then back to device
        pt_wav_cpu, pt_f0_cpu = extractor.extract(pt_wav.cpu(), pt_f0.cpu(), config.audio.sample_rate)
        pt_wav = pt_wav_cpu.to(args.device)
        pt_f0 = pt_f0_cpu.to(args.device)
        print(f"Optimal prompt extracted. Final length: {pt_wav.shape[-1] / config.audio.sample_rate:.2f}s")
        
        # Save the combined prompt for user inspection
        import torchaudio
        combined_wav_path = os.path.join(args.save_dir, "combined_prompt.wav")
        torchaudio.save(combined_wav_path, pt_wav.cpu(), config.audio.sample_rate)
        print(f"Saved optimal prompt to {combined_wav_path}")

    n_step = args.n_steps if hasattr(args, "n_steps") else config.infer.n_steps
    cfg = args.cfg if hasattr(args, "cfg") else config.infer.cfg

    with torch.no_grad():
        generated_audio, generated_shift = model.infer(
            pt_wav=pt_wav,
            gt_wav=gt_wav,
            pt_f0=pt_f0,
            gt_f0=gt_f0,
            auto_shift=args.auto_shift, 
            pitch_shift=args.pitch_shift, 
            n_steps=n_step, 
            cfg=cfg,
            use_fp16=args.use_fp16,
        )
    generated_audio = generated_audio.squeeze().float().cpu().numpy()
    if args.pitch_shift != generated_shift:
        args.pitch_shift = generated_shift
        # print(f"Applied pitch shift of {generated_shift} semitones to match GT F0 contour.")

    sf.write(os.path.join(args.save_dir, "generated.wav"), generated_audio, config.audio.sample_rate)
    print(f"Generated audio saved to {os.path.join(args.save_dir, 'generated.wav')}")


def main(args, config):
    model = build_model(
        model_path=args.model_path,
        config=config,
        device=args.device,
        use_fp16=getattr(args, "use_fp16", False),
    )
    process(args, config, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default='pretrained_models/soulx-singer/model.pt')
    parser.add_argument("--config", type=str, default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument("--prompt_wav_path", type=str, default='example/audio/zh_prompt.wav')
    parser.add_argument("--target_wav_path", type=str, default='example/audio/zh_target.wav')
    parser.add_argument("--prompt_f0_path", type=str, default='example/audio/zh_prompt_f0.npy')
    parser.add_argument("--target_f0_path", type=str, default='example/audio/zh_target_f0.npy')
    parser.add_argument("--save_dir", type=str, default='outputs')
    parser.add_argument("--auto_shift", action="store_true")
    parser.add_argument("--pitch_shift", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 inference (faster on GPU)",
    )
    parser.add_argument("--auto_prompt", action="store_true", help="Enable automatic optimal prompt extraction.")
    parser.add_argument("--auto_prompt_length", type=float, default=15.0, help="Maximum length of the auto-extracted prompt in seconds.")
    args = parser.parse_args()
    args.use_fp16 = args.fp16

    config = load_config(args.config)
    main(args, config)
