#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torchaudio
from audioseal import AudioSeal
import wavmark
import yaml
from pesq import pesq  # Using PESQ for audio quality evaluation
from tqdm import tqdm
import torchaudio.transforms as T
import time
from art.estimators.classification import PyTorchClassifier

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ALIF Blackbox Attack on AudioSeal Watermarked Audio"
    )
    parser.add_argument("--testset_size", type=int, default=200,
                        help="Number of samples from the test set to process")
    parser.add_argument("--encode", action="store_true",
                        help="Run the encoding process before decoding")
    parser.add_argument("--length", type=int, default=5 * 16000,
                        help="Length of the audio samples")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index to use")
    parser.add_argument("--save_pert", action="store_true",
                        help="If set, saves the perturbed waveform")
    parser.add_argument("--query_budget", type=int, default=10,
                        help="Query budget for the attack")
    parser.add_argument("--blackbox_folder", type=str, default="alif_10k",
                        help="Folder to save the blackbox attack results")
    parser.add_argument("--eps", type=float, default=0.05,
                        help="Epsilon for the attack")
    parser.add_argument("--tau", type=float, default=0.15,
                        help="(Unused for ALIF) Parameter placeholder")
    parser.add_argument("--snr", type=list, default=[0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30],
                        help="Signal-to-noise ratio for the attack")
    parser.add_argument("--norm", type=str, default='linf',
                        help="Norm for the attack")
    parser.add_argument("--attack_bitstring", action="store_true",
                        help="If set, perturb the bitstring instead of the detection probability")
    parser.add_argument("--attack_type", type=str, default='both', choices=['amplitude', 'phase', 'both'],
                        help="Type of attack")
    parser.add_argument("--model", type=str, default='audioseal', choices=['audioseal', 'wavmark', 'timbre'],
                        help="Model to be attacked")
    parser.add_argument("--defended_model_path", type=str, default='audioseal_defended.pth',
                        help="Defence model path")
    parser.add_argument("--def_ok", type=bool, default=False,
                        help="Defence model path")
    args = parser.parse_args()
    print("Arguments: ", args)
    return args

# ----------------------------------------------------------------
# ALIF Attack (inspired by ALIF – IEEE S&P 2024, Cheng et al.)
#
# This attack computes a surrogate gradient by converting the adversarial 
# spectrogram into a waveform and passing it through the model. To avoid shape 
# mismatches, we ensure the waveform has an explicit channel dimension. If the 
# model returns a tuple, we extract the first element. We then define the loss 
# as the mean of the model’s output.
# ----------------------------------------------------------------
def alif_attack(detector, orig_spectrogram: np.ndarray, eps: float, max_queries: int, tau: float = None):
    """
    Args:
      detector: Black-box detector (providing surrogate gradients via its model).
      orig_spectrogram: Input spectrogram as a numpy array of shape (1, C, F, T).
                        (For attack_type 'both', C=2; otherwise, C=1.)
      eps: Maximum per-element perturbation (L_inf bound).
      max_queries: Query budget (typically very low for ALIF; we use 1 query for verification).
      tau: Unused parameter for ALIF.
    Returns:
      queries_used: Number of queries used (numpy array).
      adv_spectrogram: Adversarial spectrogram (numpy array; same shape as input, without the batch dimension).
    """
    # Remove batch dimension and convert to tensor
    orig_tensor = torch.tensor(orig_spectrogram[0], dtype=torch.float32).clone()
    if orig_tensor.ndim == 2:  # shape: (F, T)
        freq_bins, time_frames = orig_tensor.shape
    elif orig_tensor.ndim == 3:  # shape: (C, F, T)
        channels, freq_bins, time_frames = orig_tensor.shape
    else:
        raise ValueError("Unexpected spectrogram shape.")
    
    # Set spectrogram variable to require gradient
    adv_spec = orig_tensor.clone().detach().requires_grad_(True)
    device = adv_spec.device

    # --- SURROGATE GRADIENT COMPUTATION ---
    # Convert adversarial spectrogram to waveform (explicitly pass sample_rate=16000)
    waveform = detector.transform.spectrogram2signal(adv_spec.unsqueeze(0), sample_rate=16000)
    # Ensure waveform has shape (B, C, T); if waveform is of shape (B, T), add a channel dimension.
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(1)
    # Pass waveform through the model
    output = detector.model(waveform)
    # If output is a tuple, extract the first element
    if isinstance(output, tuple):
        output = output[0]
    # Use mean loss to push the detection score downward
    loss = output.mean()
    detector.model.zero_grad()
    loss.backward()
    grad = adv_spec.grad
    if grad is None:
        grad = torch.zeros_like(adv_spec)
    grad_sign = grad.sign()
    perturbed_spec = adv_spec.detach() - eps * grad_sign
    perturbed_spec = torch.clamp(perturbed_spec, min=adv_spec - eps, max=adv_spec + eps)
    queries = 1  # ALIF typically uses one query for verification

    with torch.no_grad():
        result_candidate = detector.get_detection_result(perturbed_spec.unsqueeze(0))
    try:
        new_label = 1 if float(result_candidate[0]) >= 0.5 else 0
    except Exception:
        new_label = 1
    # Detach the tensor before converting to NumPy
    adv_spec_np = perturbed_spec.detach().unsqueeze(0).cpu().numpy()
    return np.array(queries), adv_spec_np

# ----------------------------------------------------------------
# Detector Wrapper and Spectrogram Conversion Class
# Adapted from your original code.
# ----------------------------------------------------------------
class WatermarkDetectorWrapper(PyTorchClassifier):
    def __init__(self, model, message, detector_type, on_bitstring, transform, th, input_size, model_type, device):
        super(WatermarkDetectorWrapper, self).__init__(model=model,
                                                       input_shape=input_size,
                                                       nb_classes=2,
                                                       channels_first=True,
                                                       loss=None)
        self._device = device
        self.message = message.to(self._device)
        self.detector_type = detector_type
        self.th = th
        self.on_bitstring = on_bitstring
        self.transform = transform
        self.model.to(self._device)
        if model_type == 'timbre':
            self.get_detection_result = self.get_detection_result_timbre
        elif model_type == 'wavmark':
            self.get_detection_result = self.get_detection_result_wavmark
        elif model_type == 'audioseal':
            self.get_detection_result = self.get_detection_result_audioseal

    def get_detection_result_audioseal(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(self._device)
        signal = self.transform.spectrogram2signal(spectrogram, sample_rate=16000)
        result, msg_decoded = self.model.detect_watermark(signal.unsqueeze(0), sample_rate=16000)
        if self.on_bitstring:
            if msg_decoded is None:
                return np.array([0])
            else:
                bitacc = 1 - torch.sum(torch.abs(self.message - msg_decoded)) / len(self.message)
                return np.array([bitacc.item()])
        else:
            return np.array([result])

    def get_detection_result_wavmark(self, spectrogram):
        spectrogram = torch.tensor(spectrogram).to(self._device)
        signal = self.transform.spectrogram2signal(spectrogram).squeeze(0).detach().cpu()
        payload, info = wavmark.decode_watermark(self.model, signal)
        if payload is None:
            return np.array([0])
        else:
            payload = torch.tensor(payload).to(self.message.device)
            bit_acc = 1 - torch.sum(torch.abs(payload - self.message)) / self.message.shape[0]
            return np.array([bit_acc.item()])
        
    def get_detection_result_timbre(self, spectrogram, batch_size=1):
        spectrogram = torch.tensor(spectrogram).to(self._device)
        signal = self.transform.spectrogram2signal(spectrogram)
        payload = self.model.test_forward(signal.unsqueeze(0))
        message = self.message * 2 - 1
        payload = payload.to(message.device)
        bitacc = (payload >= 0).eq(message >= 0).sum().float() / message.numel()
        return np.array([bitacc.item()])

class signal22spectrogram:
    def __init__(self, signal, low_frequency, high_frequency, device, attack_type):
        self.signal = signal
        self.attack_type = attack_type
        self.sig2spec = T.Spectrogram(n_fft=400, power=None).to(device)
        self.spec2sig = T.InverseSpectrogram(n_fft=400).to(device)
        self.spectrogram = self.sig2spec(signal)
        self.amplitude = torch.abs(self.spectrogram)
        self.phase = torch.angle(self.spectrogram)
        self.lf = low_frequency
        self.hf = high_frequency
        self.attack_shape = self.spectrogram[..., low_frequency:high_frequency, :].shape
        self.length = signal.shape[-1]

    def signal2spectrogram(self, signal):
        """
        Convert a waveform (tensor) to a spectrogram.
        For attack_type:
          - 'amplitude': amplitude only.
          - 'phase': phase only.
          - 'both': concatenated amplitude and phase along the channel dimension.
        """
        spectro_complex = self.sig2spec(signal)
        spectro_amplitude = torch.abs(spectro_complex)
        spectro_phase = torch.angle(spectro_complex)
        if self.attack_type == 'amplitude':
            return spectro_amplitude[..., self.lf:self.hf, :]
        elif self.attack_type == 'phase':
            return spectro_phase[..., self.lf:self.hf, :]
        elif self.attack_type == 'both':
            return torch.cat([spectro_amplitude, spectro_phase], dim=0)[..., self.lf:self.hf, :]

    def spectrogram2signal(self, spectrogram, sample_rate=16000):
        """
        Convert a spectrogram back to a waveform signal.
        """
        spectrogram = spectrogram.squeeze()
        if self.attack_type == 'both':
            padding_amp = self.amplitude
            padding_amp[..., self.lf:self.hf, :] = spectrogram[0]
            padding_phase = self.phase
            padding_phase[..., self.lf:self.hf, :] = spectrogram[1]
            spectro_complex = padding_amp * torch.exp(1j * padding_phase)
        elif self.attack_type == 'amplitude':
            padding_amp = self.amplitude
            padding_amp[..., self.lf:self.hf, :] = spectrogram
            spectro_complex = padding_amp * torch.exp(1j * self.phase)
        elif self.attack_type == 'phase':
            padding_phase = self.phase
            padding_phase[..., self.lf:self.hf, :] = spectrogram
            spectro_complex = self.amplitude * torch.exp(1j * padding_phase)
        signal = self.spec2sig(spectro_complex, self.length)
        return signal

def decode_audio_files_perturb_blackbox(model, output_dir, args, device, attack_func):
    watermarked_dir = os.path.join(output_dir, 'watermarked_{}'.format(args.testset_size))
    watermarked_files = os.listdir(watermarked_dir)
    progress_bar = tqdm(watermarked_files, desc="Decoding Watermarks under ALIF Attack")
    save_path = os.path.join(output_dir, args.blackbox_folder)
    os.makedirs(save_path, exist_ok=True)
    
    filename = os.path.join(save_path, f'alif_spectrogram.csv')
    log = open(filename, 'a' if os.path.exists(filename) else 'w')
    log.write('idx, query, acc, snr, pesq\n')
    
    for watermarked_file in progress_bar:
        parts = watermarked_file.split('_')
        idx = '_'.join(watermarked_file.split('_')[:-2])
        waveform, sample_rate = torchaudio.load(os.path.join(watermarked_dir, watermarked_file))
        waveform = waveform.to(device=device)
        original_payload_str = watermarked_file.split('_')[-2]
        original_payload = torch.tensor(list(map(int, original_payload_str)), dtype=torch.int)
        
        transform = signal22spectrogram(waveform, 0, 201, device, args.attack_type)
        detector = WatermarkDetectorWrapper(model, original_payload, 'single-tailed',
                                            args.attack_bitstring, transform, args.tau,
                                            model_type=args.model, input_size=transform.attack_shape, device=device)
        
        watermarked_spectrogram = transform.signal2spectrogram(waveform).unsqueeze(0).detach().cpu().numpy()
        queries_used, adv_spectrogram = attack_func(detector, watermarked_spectrogram, args.eps, args.query_budget, args.tau)
        adv_signal = transform.spectrogram2signal(torch.tensor(adv_spectrogram).to(device), sample_rate=16000)
        acc = detector.get_detection_result(adv_spectrogram)[0]
        snr = 10 * torch.log10(torch.sum(waveform ** 2) / torch.sum((adv_signal - waveform) ** 2))
        pesq_score = pesq(sample_rate, 
                          np.array(waveform.squeeze().detach().cpu(), dtype=np.float64), 
                          np.array(adv_signal.squeeze().detach().cpu(), dtype=np.float64), 'wb')
        print(f'idx: {idx}, query: {int(queries_used)}, acc: {acc:.3f}, snr: {snr:.1f}, pesq: {pesq_score:.3f}')
        log.write(f'{idx}, {int(queries_used)}, {acc}, {snr}, {pesq_score}\n')
        torchaudio.save(os.path.join(save_path, 
            f"{idx}_tau{args.tau}_query{int(queries_used)}_snr{snr:.1f}_acc{acc:.1f}_pesq{pesq_score:.1f}.wav"),
            adv_signal.detach().cpu(), sample_rate)
    log.close()

def main():
    args = parse_arguments()
    if args.norm == 'l2':
        args.norm = 2
    else:
        args.norm = np.inf
    np.random.seed(42)
    torch.manual_seed(42)
    
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    if args.model == 'audioseal':
        model = AudioSeal.load_detector("audioseal_detector_16bits").to(device=device)
        
        if args.def_ok:
            print("Saved model loaded --------------------------------------------------------------")
            model.load_state_dict(torch.load(args.defended_model_path, map_location=device))
        output_dir = 'audiomarkdata_audioseal_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'wavmark':
        model = wavmark.load_model().to(device)
        output_dir = 'audiomarkdata_wavmark_max_5s'
        os.makedirs(output_dir, exist_ok=True)
    elif args.model == 'timbre':
        process_config = yaml.load(open("timbre/config/process.yaml", "r"), Loader=yaml.FullLoader)
        model_config = yaml.load(open("timbre/config/model.yaml", "r"), Loader=yaml.FullLoader)
        print("Timbre model not fully supported in this demo. Exiting.")
        return

    decode_audio_files_perturb_blackbox(model, output_dir, args, device, alif_attack)

if __name__ == "__main__":
    main()
