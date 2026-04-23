import numpy as np
from scipy.signal import resample_poly


def to_pcm16_8k(audio: np.ndarray, src_sr: int, dst_sr: int = 8000) -> bytes:
    """Convert mono float audio to raw 16-bit LE PCM at `dst_sr`.

    OmniVoice emits float32/float16 in [-1, 1] at 24 kHz. Telephony expects
    int16 LE mono at 8 kHz.
    """
    if audio.ndim != 1:
        audio = audio.squeeze()
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape {audio.shape}")

    audio = audio.astype(np.float32, copy=False)

    if src_sr != dst_sr:
        # resample_poly uses polyphase FIR; anti-aliased and stable for fixed ratios.
        from math import gcd

        g = gcd(src_sr, dst_sr)
        audio = resample_poly(audio, dst_sr // g, src_sr // g)

    np.clip(audio, -1.0, 1.0, out=audio)
    pcm = (audio * 32767.0).astype(np.int16)
    return pcm.tobytes()


def sine_pcm(duration_s: float, sr: int = 8000, freq: float = 440.0) -> bytes:
    """Return a sine wave as raw int16 LE PCM — used by the mock engine."""
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    audio = 0.3 * np.sin(2.0 * np.pi * freq * t)
    return (audio * 32767.0).astype(np.int16).tobytes()
