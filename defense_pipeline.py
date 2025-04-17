#!/usr/bin/env python
"""
defense_pipeline.py

This script implements a model‐integrated adversarial defense for AudioSeal using a high‐frequency–selective adversarial training (F-SAT style) method.
Two modes are supported:
  1. Training mode ("train"): Fine‐tunes the baseline AudioSeal detector using adversarial examples that perturb only the high‐frequency spectrum.
  2. Evaluation mode ("eval"): Evaluates both the baseline and defended models under a high‐frequency adversarial attack.
  
In this version, we assume a regression setup where the detector’s output should match the clean audio (using MSE loss).
Metrics logged include:
  - MSE loss on clean and adversarial examples.
  - SNR (Signal-to-Noise Ratio) of the adversarial attack.
  - PESQ (Perceptual Evaluation of Speech Quality) of the adversarial audio.

Usage examples:
  • Training: 
      python defense_pipeline.py --mode train --train_folder path/to/train_data --gpu 0
  • Evaluation:
      python defense_pipeline.py --mode eval --test_folder path/to/test_data --defended_model_path audioseal_defended.pth --gpu 0
"""

import os, argparse, time, fnmatch
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from pesq import pesq  # PESQ library for audio quality evaluation
from audioseal import AudioSeal  # AudioSeal package must be installed and in PYTHONPATH
from torch.utils.data import Dataset, DataLoader
import multiprocessing

# Set multiprocessing start method to "spawn" to avoid CUDA reinitialization issues.
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

# ----------------------- Global Settings -----------------------
sample_rate = 16000          # AudioSeal works at 16kHz.
epsilon = 1e-4               # Maximum perturbation (L_inf bound).
step_size = 2e-5             # PGD step size.
pgd_steps = 2                # Number of PGD iterations.
high_freq_cutoff = 4000      # Hz: focus on perturbing high-frequency components.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- Utility Functions -----------------------
def get_logits(model, audio):
    """Call the model on audio and extract output if it's a tuple."""
    out = model(audio)
    if isinstance(out, tuple):
        return out[0]
    return out

def highfreq_filter(signal_fft, cutoff_hz, sr):
    """
    Zeroes out low-frequency FFT coefficients.
    :param signal_fft: FFT coefficients (complex tensor)
    :param cutoff_hz: Frequency (Hz) below which to zero out coefficients.
    :param sr: Sample rate.
    :return: Filtered FFT coefficients.
    """
    n = (signal_fft.shape[-1]-1) * 2  
    freq_res = sr / float(n)
    cutoff_bin = int(cutoff_hz / freq_res)
    filtered_fft = signal_fft.clone()
    filtered_fft[..., :cutoff_bin] = 0
    return filtered_fft

def generate_adversarial(audio, label, model):
    """
    Generates an adversarial example using PGD that perturbs only the high-frequency components.
    Here we assume the detector's output (e.g. a reconstructed audio signal) is compared against the clean input with MSE.
    :param audio: Tensor with shape (1, samples)
    :param label: Unused in this regression setup.
    :param model: AudioSeal detector model.
    :return: Adversarial audio sample tensor.
    """
    adv_audio = audio.clone().detach()
    adv_audio.requires_grad_(True)
    for _ in range(pgd_steps):
        output = get_logits(model, adv_audio)
        loss = nn.MSELoss()(output, audio)
        loss.backward()
        grad = adv_audio.grad
        # Filter gradient to keep only high-frequency components.
        grad_fft = torch.fft.rfft(grad, dim=-1)
        filtered_grad_fft = highfreq_filter(grad_fft, high_freq_cutoff, sample_rate)
        filtered_grad = torch.fft.irfft(filtered_grad_fft, n=audio.shape[-1])
        # Update adversarial audio.
        adv_audio.data = adv_audio.data + step_size * torch.sign(filtered_grad)
        delta = torch.clamp(adv_audio.data - audio, min=-epsilon, max=epsilon)
        adv_audio.data = torch.clamp(audio + delta, -1.0, 1.0)
        adv_audio.grad.zero_()
    return adv_audio.detach()

def attack_highfreq(audio, model):
    """
    Generates a one-step FGSM high-frequency attack on a given audio sample.
    During evaluation, because the model includes an RNN the backward pass
    requires the model to be in training mode. We temporarily switch modes.
    :param audio: Tensor with shape (1, samples)
    :param model: AudioSeal detector model.
    :return: Adversarial audio sample tensor.
    """
    was_training = model.training  # Save current mode.
    model.train()  # Switch to training mode to enable RNN backward.
    audio_adv = audio.clone().detach()
    audio_adv.requires_grad_(True)
    output = get_logits(model, audio_adv)
    loss = nn.MSELoss()(output, audio)
    loss.backward()
    grad = audio_adv.grad
    grad_fft = torch.fft.rfft(grad, dim=-1)
    filtered_grad_fft = highfreq_filter(grad_fft, high_freq_cutoff, sample_rate)
    filtered_grad = torch.fft.irfft(filtered_grad_fft, n=audio.shape[-1])
    adv_audio = audio_adv + epsilon * torch.sign(filtered_grad)
    delta = torch.clamp(adv_audio - audio, -epsilon, epsilon)
    adv_audio = torch.clamp(audio + delta, -1.0, 1.0)
    if not was_training:
        model.eval()  # Restore mode.
    return adv_audio.detach()

def compute_snr(original, perturbed):
    """
    Compute the Signal-to-Noise Ratio (SNR) between the original and perturbed audio.
    :param original: Clean audio tensor.
    :param perturbed: Perturbed audio tensor.
    :return: SNR value in dB.
    """
    noise = perturbed - original
    snr = 10 * torch.log10(torch.sum(original ** 2) / (torch.sum(noise ** 2) + 1e-10))
    return snr.item()

# ----------------------- Dataset Classes -----------------------
class AudioTrainDataset(Dataset):
    def __init__(self, folder):
        """
        Expects WAV files in the folder.
        Label information is parsed from the filename.
        (E.g., filenames containing '_label1' will have label 1; otherwise label 0.)
        In this regression setup, the label is unused.
        """
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        waveform, sr = torchaudio.load(file)
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.clamp(-1.0, 1.0)
        label = 1 if '_label1' in file else 0  # Unused.
        return waveform.to(device), torch.tensor([label], dtype=torch.long).to(device)

class AudioTestDataset(Dataset):
    def __init__(self, folder):
        """
        Test dataset: reads WAV files from the folder.
        """
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        waveform, sr = torchaudio.load(file)
        if sr != sample_rate:
            resampler = T.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waveform = waveform.clamp(-1.0, 1.0)
        label = 1 if '_label1' in file else 0
        return waveform.to(device), torch.tensor([label], dtype=torch.long).to(device), os.path.basename(file)

# ----------------------- Training Function -----------------------
def train_defense(args):
    print("Loading baseline AudioSeal detector...")
    model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(os.listdir(args.train_folder))
    dataset = AudioTrainDataset(args.train_folder)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    loss_fn = nn.MSELoss()
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for audio, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            output_clean = get_logits(model, audio)
            loss_clean = loss_fn(output_clean, audio)
            
            adv_audio_list = []
            for i in range(audio.size(0)):
                adv = generate_adversarial(audio[i:i+1], label[i:i+1], model)
                adv_audio_list.append(adv)
            adv_audio = torch.cat(adv_audio_list, dim=0)
            output_adv = get_logits(model, adv_audio)
            loss_adv = loss_fn(output_adv, audio)
            
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * audio.size(0)
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), args.defended_model_path)
    print(f"Defended model saved to {args.defended_model_path}")

# ----------------------- Evaluation Function -----------------------
def evaluate_defense(args):
    print("Loading original (baseline) AudioSeal detector...")
    orig_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    orig_model.eval()
    print("Loading defended AudioSeal detector...")
    defended_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    defended_model.load_state_dict(torch.load(args.defended_model_path, map_location=device))
    defended_model.eval()
    
    dataset = AudioTestDataset(args.test_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    os.makedirs(args.eval_output_folder, exist_ok=True)
    log_path = os.path.join(args.eval_output_folder, "defense_evaluation.csv")
    log_file = open(log_path, "w")
    log_file.write("idx, orig_clean_loss, orig_adv_loss, def_clean_loss, def_adv_loss, snr, pesq\n")
    
    loss_fn = nn.MSELoss()
    for audio, label, fname in tqdm(dataloader, desc="Evaluating defense"):
        with torch.no_grad():
            orig_clean_output = get_logits(orig_model, audio)
            def_clean_output = get_logits(defended_model, audio)
            orig_clean_loss = loss_fn(orig_clean_output, audio).item()
            def_clean_loss = loss_fn(def_clean_output, audio).item()
        
        adv_audio = attack_highfreq(audio, orig_model)
        snr_val = compute_snr(audio, adv_audio)
        clean_np = audio.squeeze().cpu().numpy().astype(np.float64)
        adv_np = adv_audio.squeeze().cpu().numpy().astype(np.float64)
        try:
            pesq_val = pesq(sample_rate, clean_np, adv_np, 'wb')
        except Exception as e:
            pesq_val = -1
        
        with torch.no_grad():
            orig_adv_output = get_logits(orig_model, adv_audio)
            def_adv_output = get_logits(defended_model, adv_audio)
            orig_adv_loss = loss_fn(orig_adv_output, audio).item()
            def_adv_loss = loss_fn(def_adv_output, audio).item()
        
        log_file.write(f"{fname},{orig_clean_loss:.6f},{orig_adv_loss:.6f},{def_clean_loss:.6f},{def_adv_loss:.6f},{snr_val:.2f},{pesq_val:.2f}\n")
        print(f"File: {fname} | Orig Clean Loss: {orig_clean_loss:.6f} | Orig Adv Loss: {orig_adv_loss:.6f} | " +
              f"Def Clean Loss: {def_clean_loss:.6f} | Def Adv Loss: {def_adv_loss:.6f} | SNR: {snr_val:.2f} dB | PESQ: {pesq_val:.2f}")
    log_file.close()
    print(f"Evaluation results saved to {log_path}")

# ----------------------- Argument Parsing -----------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Defense for AudioSeal using F-SAT")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], required=True,
                        help="Mode: 'train' to perform adversarial defense training; 'eval' to evaluate the defense under attack.")
    parser.add_argument("--train_folder", type=str, default="audiomarkdata_audioseal_train",
                        help="Folder with training WAV files for defense training.")
    parser.add_argument("--test_folder", type=str, default="audiomarkdata_audioseal_test",
                        help="Folder with test WAV files for evaluation.")
    parser.add_argument("--defended_model_path", type=str, default="audioseal_defended.pth",
                        help="Path to save (or load) the defended model weights.")
    parser.add_argument("--eval_output_folder", type=str, default="defense_eval_results",
                        help="Folder to save evaluation logs (CSV file).")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for adversarial training.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use (if available).")
    return parser.parse_args()

# ----------------------- Main Function -----------------------
def main():
    args = parse_arguments()
    global device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    if args.mode == "train":
        train_defense(args)
    elif args.mode == "eval":
        evaluate_defense(args)

if __name__ == "__main__":
    main()
