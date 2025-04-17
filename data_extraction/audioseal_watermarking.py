# import os
# import torch
# import torchaudio
# import numpy as np
# from pydub import AudioSegment
# from audioseal import AudioSeal

# # --- Configuration ---

# # Update these paths as needed:
# train_dir = "train_folder"             # Folder with original train MP3 files
# test_dir = "test_folder"               # Folder with original test MP3 files
# watermarked_train_dir = "audiomarkdata_audioseal_train"
# watermarked_test_dir = "audiomarkdata_audioseal_test"

# # Create destination directories if they don't exist.
# os.makedirs(watermarked_train_dir, exist_ok=True)
# os.makedirs(watermarked_test_dir, exist_ok=True)

# # Desired sample rate (16 kHz)
# desired_sr = 16000

# # Set device (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load the AudioSeal watermark generator model and move it to device.
# # Replace "audioseal_wm_16bits" below with your valid checkpoint or local path if needed.
# try:
#     model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)
# model.eval()  # Set model to evaluation mode

# # --- Functions ---

# def process_audio_file(input_path, output_path, model, device, desired_sr=16000):
#     """
#     Loads an MP3 file, resamples to the desired sample rate if needed,
#     moves the audio tensor to the specified device, generates a watermark using
#     AudioSeal, adds the watermark to the audio, and exports the result as an MP3.
#     """
#     try:
#         # Load audio (returns waveform with shape [channels, samples]) and sample rate.
#         waveform, sr = torchaudio.load(input_path)
#     except Exception as e:
#         print(f"Error loading {input_path}: {e}")
#         return

#     # Resample if necessary.
#     if sr != desired_sr:
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sr)
#         waveform = resampler(waveform)
#         sr = desired_sr

#     # Move waveform to device and add batch dimension.
#     waveform_batch = waveform.unsqueeze(0).to(device)  # Shape: (1, channels, samples)

#     # Generate the watermark (ensure no gradient computation during inference).
#     with torch.no_grad():
#         watermark = model.get_watermark(waveform_batch, sr)

#     # Add the watermark to the original waveform.
#     watermarked_batch = waveform_batch + watermark

#     # Remove batch dimension and move back to CPU.
#     watermarked = watermarked_batch.squeeze(0).cpu()

#     # Convert tensor to NumPy array.
#     watermarked_np = watermarked.numpy()

#     # Convert float tensor values in [-1.0, 1.0] to 16-bit PCM format.
#     watermarked_int16 = np.int16(watermarked_np * 32767)

#     # Determine the number of channels.
#     channels = watermarked_int16.shape[0]

#     # If multi-channel, interleave the channels.
#     if channels > 1:
#         interleaved = watermarked_int16.T.flatten()
#     else:
#         interleaved = watermarked_int16.flatten()

#     # Create a pydub AudioSegment from the raw bytes.
#     audio_seg = AudioSegment(
#         interleaved.tobytes(),
#         frame_rate=sr,
#         sample_width=2,   # 16-bit audio = 2 bytes per sample.
#         channels=channels
#     )

#     # Export the watermarked audio as an MP3 file.
#     try:
#         audio_seg.export(output_path, format="mp3")
#         print(f"Watermarked file saved to {output_path}")
#     except Exception as e:
#         print(f"Error saving {output_path}: {e}")

# def process_folder(input_folder, output_folder, model, device, desired_sr=16000):
#     """
#     Processes all MP3 files in input_folder: generates watermarked
#     audio and saves the output in output_folder.
#     """
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".mp3"):
#             input_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, filename)
#             process_audio_file(input_path, output_path, model, device, desired_sr)

# # --- Process Folders ---

# print("Processing train folder:")
# process_folder(train_dir, watermarked_train_dir, model, device, desired_sr)

# print("\nProcessing test folder:")
# process_folder(test_dir, watermarked_test_dir, model, device, desired_sr)

# import os
# import torch
# import torchaudio
# import numpy as np
# from pydub import AudioSegment
# from audioseal import AudioSeal

# # --- Set local ffmpeg path (if needed for pydub) ---
# # Uncomment and adjust the line below if ffmpeg is not in your PATH.
# # AudioSegment.converter = os.path.expanduser("~/ffmpeg_local/ffmpeg-4.4.2-64bit-static/ffmpeg")

# # --- Configuration ---

# # Update these paths as needed:
# train_dir = "train_folder"             # Folder with original train MP3 files
# test_dir = "test_folder"               # Folder with original test MP3 files
# watermarked_train_dir = "audiomarkdata_audioseal_train"
# watermarked_test_dir = "audiomarkdata_audioseal_test"

# # Create destination directories if they don't exist.
# os.makedirs(watermarked_train_dir, exist_ok=True)
# os.makedirs(watermarked_test_dir, exist_ok=True)

# # Desired sample rate (16 kHz)
# desired_sr = 16000

# # Set device (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load the AudioSeal watermark generator model and move it to the device.
# # Use a valid checkpoint identifier; adjust if necessary.
# try:
#     model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)
# model.eval()  # Set model to evaluation mode

# # --- Functions ---

# def process_audio_file(input_path, output_path, model, device, desired_sr=16000):
#     """
#     Loads an MP3 file, resamples to the desired sample rate if needed,
#     moves the audio tensor to the specified device, generates a watermark
#     using AudioSeal, adds the watermark to the audio, and exports the result as a WAV file.
#     """
#     try:
#         # Load audio (returns waveform with shape [channels, samples]) and its sampling rate.
#         waveform, sr = torchaudio.load(input_path)
#     except Exception as e:
#         print(f"Error loading {input_path}: {e}")
#         return

#     # Resample if necessary.
#     if sr != desired_sr:
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sr)
#         waveform = resampler(waveform)
#         sr = desired_sr

#     # Move waveform to device and add a batch dimension.
#     waveform_batch = waveform.unsqueeze(0).to(device)  # Shape: (1, channels, samples)

#     # Generate the watermark (disable gradient computation for inference).
#     with torch.no_grad():
#         watermark = model.get_watermark(waveform_batch, sr)

#     # Add the watermark to the original waveform.
#     watermarked_batch = waveform_batch + watermark

#     # Remove the batch dimension and move the result back to CPU.
#     watermarked = watermarked_batch.squeeze(0).cpu()

#     # Convert the tensor to a NumPy array.
#     watermarked_np = watermarked.numpy()

#     # Convert float values in [-1.0, 1.0] to 16-bit PCM integers.
#     watermarked_int16 = np.int16(watermarked_np * 32767)

#     # Determine the number of channels.
#     channels = watermarked_int16.shape[0]

#     # If multi-channel, interleave the channels.
#     if channels > 1:
#         interleaved = watermarked_int16.T.flatten()
#     else:
#         interleaved = watermarked_int16.flatten()

#     # Create a pydub AudioSegment from the raw bytes.
#     audio_seg = AudioSegment(
#         interleaved.tobytes(),
#         frame_rate=sr,
#         sample_width=2,  # 16-bit audio = 2 bytes per sample.
#         channels=channels
#     )

#     # Export the watermarked audio as a WAV file.
#     try:
#         audio_seg.export(output_path, format="wav")
#         print(f"Watermarked file saved to {output_path}")
#     except Exception as e:
#         print(f"Error saving {output_path}: {e}")

# def process_folder(input_folder, output_folder, model, device, desired_sr=16000):
#     """
#     Processes all MP3 files in input_folder: generates watermarked audio
#     and saves the output in output_folder as WAV files.
#     """
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".mp3"):
#             input_path = os.path.join(input_folder, filename)
#             # Change the file extension to .wav for output.
#             output_filename = os.path.splitext(filename)[0] + ".wav"
#             output_path = os.path.join(output_folder, output_filename)
#             process_audio_file(input_path, output_path, model, device, desired_sr)

# # --- Process Folders ---

# print("Processing train folder:")
# process_folder(train_dir, watermarked_train_dir, model, device, desired_sr)

# print("\nProcessing test folder:")
# process_folder(test_dir, watermarked_test_dir, model, device, desired_sr)


# import os
# import torch
# import torchaudio
# import numpy as np
# from audioseal import AudioSeal

# # --- Configuration ---

# # Input folders containing original MP3 files
# train_dir = "train_folder"             
# test_dir = "test_folder"               

# # Output folders to store watermarked WAV files
# watermarked_train_dir = "watermarked_train"
# watermarked_test_dir = "watermarked_test"

# # Create output directories if they don't exist.
# os.makedirs(watermarked_train_dir, exist_ok=True)
# os.makedirs(watermarked_test_dir, exist_ok=True)

# # Desired sample rate (16 kHz)
# desired_sr = 16000

# # Set device (GPU if available, else CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Load the AudioSeal watermark generator model and move it to the device.
# # Make sure you use a valid checkpoint identifier; here we assume "audioseal_wm_16bits"
# try:
#     model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit(1)
# model.eval()  # Evaluation mode

# # --- Function to Process and Save Watermarked Audio as WAV ---

# def process_audio_file(input_path, output_path, model, device, desired_sr=16000):
#     """
#     Loads an MP3 file, resamples to desired_sr if needed, moves the audio tensor to the device,
#     generates a watermark using AudioSeal, adds the watermark to the audio, and writes the result
#     as a WAV file using torchaudio.save (preserving PCM data for consistent watermark decoding).
#     """
#     try:
#         # Load the audio file; waveform shape: [channels, samples]
#         waveform, sr = torchaudio.load(input_path)
#     except Exception as e:
#         print(f"Error loading {input_path}: {e}")
#         return

#     # Resample if sample rate differs from desired_sr.
#     if sr != desired_sr:
#         resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sr)
#         waveform = resampler(waveform)
#         sr = desired_sr

#     # Move waveform to the specified device and add a batch dimension (needed by AudioSeal).
#     waveform_batch = waveform.unsqueeze(0).to(device)  # Shape: (1, channels, samples)

#     # Generate the watermark with inference mode (disable gradients).
#     with torch.no_grad():
#         watermark = model.get_watermark(waveform_batch, sr)

#     # Add the watermark to the original audio.
#     watermarked_batch = waveform_batch + watermark

#     # Remove the batch dimension and move the result back to CPU.
#     watermarked = watermarked_batch.squeeze(0).cpu()

#     # Convert audio to 16-bit PCM. The original waveform is assumed to be in the range [-1, 1].
#     watermarked_int16 = (watermarked.numpy() * 32767).astype(np.int16)

#     # Convert numpy array to a torch tensor of type torch.int16.
#     pcm_tensor = torch.from_numpy(watermarked_int16)

#     # Save the watermarked audio to a WAV file using torchaudio.save.
#     # This writes a 16-bit PCM WAV file preserving the raw audio data.
#     try:
#         torchaudio.save(output_path, pcm_tensor, sr, encoding="PCM_S", bits_per_sample=16)
#         print(f"Watermarked file saved to {output_path}")
#     except Exception as e:
#         print(f"Error saving {output_path}: {e}")

# def process_folder(input_folder, output_folder, model, device, desired_sr=16000):
#     """
#     Iterates over all MP3 files in input_folder, processes each file to add a watermark,
#     and saves the output as a WAV file in output_folder.
#     """
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(".mp3"):
#             input_path = os.path.join(input_folder, filename)
#             # Replace file extension with .wav for output.
#             output_filename = os.path.splitext(filename)[0] + ".wav"
#             output_path = os.path.join(output_folder, output_filename)
#             process_audio_file(input_path, output_path, model, device, desired_sr)

# # --- Process the Train and Test Folders ---

# print("Processing train folder:")
# process_folder(train_dir, watermarked_train_dir, model, device, desired_sr)

# print("\nProcessing test folder:")
# process_folder(test_dir, watermarked_test_dir, model, device, desired_sr)


import os
import torch
import torchaudio
import numpy as np
from audioseal import AudioSeal

# --- Configuration ---

# Folders for input MP3s and output watermarked WAV files.
train_dir = "train_folder"             # Folder with original train MP3 files
test_dir = "test_folder"               # Folder with original test MP3 files
watermarked_train_dir = "audiomarkdata_audioseal_train"
watermarked_test_dir = "audiomarkdata_audioseal_test"

# Create output directories if they don't exist.
os.makedirs(watermarked_train_dir, exist_ok=True)
os.makedirs(watermarked_test_dir, exist_ok=True)

# Desired sample rate and target length.
desired_sr = 16000       # 16 kHz
target_length = 80000    # 80000 samples, corresponding to 5 seconds at 16 kHz

# Set device (GPU if available, else CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the AudioSeal watermark generator model and move it to the device.
# Use the pre-trained 16-bit model checkpoint.
try:
    model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
model.eval()  # Set model to evaluation mode

# --- Helper Functions ---

def pad_or_truncate(waveform, target_length):
    """
    Pads with zeros or truncates the waveform (assumed shape: [channels, samples])
    so that its number of samples equals target_length.
    """
    _, n_samples = waveform.shape
    if n_samples < target_length:
        pad_amount = target_length - n_samples
        padding = torch.zeros((waveform.size(0), pad_amount), dtype=waveform.dtype)
        waveform = torch.cat([waveform, padding], dim=1)
    elif n_samples > target_length:
        waveform = waveform[:, :target_length]
    return waveform

def downmix_to_mono(waveform):
    """
    Downmix multi-channel audio to mono by taking the mean across channels.
    """
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

def generate_payload(nbits=16):
    """
    Generate a random binary payload of length nbits.
    Returns a tensor of shape [1, nbits] with values 0 or 1.
    """
    payload = torch.randint(0, 2, (1, nbits), dtype=torch.int64)
    return payload

def payload_to_string(payload):
    """
    Convert a payload tensor (shape: [1, nbits]) into a string of bits.
    """
    bit_list = payload.squeeze(0).tolist()
    return ''.join(str(b) for b in bit_list)

def extract_watermarked_audio_and_payload(input_path, model, device, desired_sr=16000, target_length=80000):
    """
    Loads an MP3 file, preprocesses it (downmix, resample, pad/truncate),
    generates a watermark with an explicit 16-bit payload using AudioSeal, and returns:
      - The watermarked audio tensor (shape: [channels, samples])
      - The payload string (16-bit)
      - The sample rate (should be desired_sr)
    """
    try:
        waveform, sr = torchaudio.load(input_path)  # waveform shape: [channels, samples]
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return None, None, None

    # Downmix to mono.
    waveform = downmix_to_mono(waveform)
    # Resample if needed.
    if sr != desired_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=desired_sr)
        waveform = resampler(waveform)
        sr = desired_sr
    # Pad or truncate.
    waveform = pad_or_truncate(waveform, target_length)
    # Move to device and add batch dimension.
    waveform_batch = waveform.unsqueeze(0).to(device)  # Shape: [1, channels, samples]
    
    # Generate a random payload.
    nbits = model.msg_processor.nbits if hasattr(model, "msg_processor") else 16
    payload = generate_payload(nbits)
    payload_str = payload_to_string(payload)
    
    # Generate watermark with the given payload.
    with torch.no_grad():
        watermark = model.get_watermark(waveform_batch, sr, message=payload)
    
    # Compute watermarked audio.
    watermarked_batch = waveform_batch + watermark
    watermarked = watermarked_batch.squeeze(0).cpu()  # Move back to CPU
    
    return watermarked, payload_str, sr

# --- Main Processing Function ---

def process_folder(input_folder, output_folder, model, device, desired_sr=16000, target_length=80000):
    """
    Iterates over all MP3 files in input_folder, extracts the watermarked audio
    and payload, constructs the output filename to include the payload, and saves
    the watermarked audio as a WAV file using torchaudio.save.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            watermarked, payload_str, sr = extract_watermarked_audio_and_payload(input_path, model, device, desired_sr, target_length)
            if watermarked is None:
                continue
            base = os.path.splitext(filename)[0]
            # Construct filename: originalName_<payload>.wav
            output_filename = f"{base}_{payload_str}.wav"
            output_path = os.path.join(output_folder, output_filename)
            try:
                torchaudio.save(output_path, watermarked, sr, encoding="PCM_S", bits_per_sample=16)
                print(f"Watermarked file saved to {output_path}")
            except Exception as e:
                print(f"Error saving {output_path}: {e}")

# --- Process Train and Test Folders ---

print("Processing train folder:")
process_folder(train_dir, watermarked_train_dir, model, device, desired_sr, target_length)

print("\nProcessing test folder:")
process_folder(test_dir, watermarked_test_dir, model, device, desired_sr, target_length)
