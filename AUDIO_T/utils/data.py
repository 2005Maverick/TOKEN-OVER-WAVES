import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings

class AudioTransforms:
    def __init__(self, sample_rate, max_length, n_fft, n_mels, hop_length, training=True):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.training = training
        
        self.expected_time_length = int((sample_rate * max_length) / hop_length) + 1
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        
        if training:
            self.time_stretch = torchaudio.transforms.TimeStretch()
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=20)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)

    def _prepare_audio(self, waveform, sample_rate):
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if waveform.numel() == 0 or torch.max(torch.abs(waveform)) == 0:
            return torch.zeros(1, int(self.sample_rate * self.max_length))
            
        waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        target_length = int(self.sample_rate * self.max_length)
        current_length = waveform.shape[1]
        
        if current_length >= target_length:
            if self.training:
                start = torch.randint(0, current_length - target_length + 1, (1,)).item()
                waveform = waveform[:, start:start + target_length]
            else:
                start = (current_length - target_length) // 2
                waveform = waveform[:, start:start + target_length]
        else:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        if waveform.numel() == 0:
            return torch.zeros(1, int(self.sample_rate * self.max_length))
        
        return waveform

    def __call__(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            waveform_np, sample_rate = librosa.load(audio_path, sr=None, mono=False)
            waveform = torch.from_numpy(waveform_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.numel() == 0:
                waveform = torch.zeros(1, int(self.sample_rate * self.max_length))
                sample_rate = self.sample_rate

        waveform = self._prepare_audio(waveform, sample_rate)
        
        if waveform.numel() == 0:
            mel_spec = torch.zeros(1, self.n_mels, self.expected_time_length)
            waveform = torch.zeros(1, int(self.sample_rate * self.max_length))
            return waveform, mel_spec

        if self.training:
            if torch.rand(1) < 0.5:
                gain_factor = 1.0 + torch.randn(1).clamp(-0.2, 0.2)
                waveform = waveform * gain_factor

        mel_spec = self.mel_spectrogram(waveform)
        mel_spec = (mel_spec + 1e-8).log()
        
        if mel_spec.shape[2] != self.expected_time_length:
            mel_spec = torch.nn.functional.interpolate(
                mel_spec,
                size=(self.n_mels, self.expected_time_length),
                mode='bilinear',
                align_corners=False
            )
        
        std_val = mel_spec.std()
        if std_val < 1e-8:
            mel_spec = mel_spec - mel_spec.mean()
        else:
            mel_spec = (mel_spec - mel_spec.mean()) / (std_val + 1e-8)
        
        if self.training:
            if torch.rand(1) < 0.5:
                mel_spec = self.freq_mask(mel_spec)
            if torch.rand(1) < 0.5:
                mel_spec = self.time_mask(mel_spec)

        return waveform, mel_spec

class DeepfakeAudioDataset(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.files = []
        self.labels = []
        
        split_dir = os.path.join(root_dir, split)
        
        if os.path.exists(split_dir):
            for label_name in ['real', 'fake']:
                label_dir = os.path.join(split_dir, label_name)
                if os.path.exists(label_dir):
                    label_value = 0 if label_name == 'real' else 1
                    for audio_file in os.listdir(label_dir):
                        if audio_file.endswith(('.wav', '.mp3')):
                            file_path = os.path.join(label_dir, audio_file)
                            if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                                self.files.append(file_path)
                                self.labels.append(label_value)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        
        try:
            if self.transform:
                waveform, mel_spec = self.transform(audio_path)
            else:
                waveform, _ = torchaudio.load(audio_path)
                mel_transform = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                mel_spec = mel_transform(waveform)
            
            return {
                'waveform': waveform,
                'mel_spec': mel_spec,
                'label': torch.tensor(label, dtype=torch.long),
                'path': audio_path
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return {
                'waveform': torch.zeros(1, self.expected_waveform_length),
                'mel_spec': torch.zeros(1, self.n_mels, self.expected_time_length),
                'label': torch.tensor(label, dtype=torch.long),
                'path': audio_path
            }
