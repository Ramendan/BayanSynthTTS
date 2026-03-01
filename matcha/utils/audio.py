"""Real mel_spectrogram using torch.stft — replaces Matcha-TTS dependency.

This is a self-contained reimplementation of librosa-style mel spectrogram
using only PyTorch + torchaudio (both already required by CosyVoice).
"""
import torch
import torch.nn.functional as F
import torchaudio


def mel_spectrogram(
    y,
    n_fft=1024,
    num_mels=80,
    sampling_rate=22050,
    hop_size=256,
    win_size=1024,
    fmin=0,
    fmax=None,
    center=False,
):
    """Compute log-mel spectrogram from waveform tensor.

    Args:
        y: Waveform tensor of shape (B, T) or (T,).
        n_fft: FFT size.
        num_mels: Number of mel filterbank channels.
        sampling_rate: Audio sample rate in Hz.
        hop_size: Hop length in samples.
        win_size: Window length in samples.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank (None = sampling_rate/2).
        center: Whether to pad signal for centered STFT.

    Returns:
        Mel spectrogram tensor of shape (B, num_mels, T_frames).
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # Pad signal if needed (match librosa behavior when center=False)
    if not center:
        pad_amount = (n_fft - hop_size) // 2
        y = F.pad(y, (pad_amount, pad_amount), mode="reflect")

    # Hann window
    window = torch.hann_window(win_size, device=y.device, dtype=y.dtype)

    # STFT
    spec = torch.stft(
        y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
        window=window, center=center, return_complex=True,
    )
    # Power spectrogram
    spec = spec.abs().pow(2)  # (B, n_fft//2+1, T_frames)

    # Mel filterbank via torchaudio
    if fmax is None:
        fmax = sampling_rate / 2.0
    mel_fb = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=float(fmin),
        f_max=float(fmax),
        n_mels=num_mels,
        sample_rate=sampling_rate,
    ).to(device=y.device, dtype=y.dtype)  # (n_fft//2+1, num_mels)

    # Apply filterbank: (B, n_fft//2+1, T) @ (n_fft//2+1, num_mels) → (B, num_mels, T)
    mel = torch.matmul(spec.transpose(1, 2), mel_fb).transpose(1, 2)

    # Log scale (clamp to avoid log(0))
    mel = torch.clamp(mel, min=1e-5).log()

    return mel
