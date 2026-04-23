import numpy as np

from tts_service.audio import sine_pcm, to_pcm16_8k


def test_to_pcm16_8k_downsamples_correctly():
    sr_in = 24000
    duration = 0.5
    t = np.arange(int(sr_in * duration), dtype=np.float32) / sr_in
    audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)

    pcm = to_pcm16_8k(audio, src_sr=sr_in, dst_sr=8000)

    # 0.5s * 8000 Hz * 2 bytes
    assert len(pcm) == 8000
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples.dtype == np.int16
    # sanity: non-silent
    assert np.abs(samples).mean() > 1000


def test_to_pcm16_clips_out_of_range():
    audio = np.array([2.0, -2.0, 0.0, 0.5], dtype=np.float32)
    pcm = to_pcm16_8k(audio, src_sr=8000, dst_sr=8000)
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples[0] == 32767  # clipped
    assert samples[1] == -32767  # clipped
    assert samples[2] == 0


def test_to_pcm16_handles_stereo_input():
    stereo = np.zeros((100, 2), dtype=np.float32)
    # squeeze via .squeeze() won't collapse (100, 2); should raise
    try:
        to_pcm16_8k(stereo, src_sr=8000)
    except ValueError:
        return
    raise AssertionError("expected ValueError for stereo input")


def test_sine_pcm_length():
    pcm = sine_pcm(duration_s=1.0, sr=8000)
    assert len(pcm) == 16000
