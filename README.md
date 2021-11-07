# Paddle-Librosa: Paddle implementation of Librosa

This codebase provides Paddle implementation of some librosa functions. If users previously used for training cpu-extracted features from librosa, but want to add GPU acceleration during training and evaluation, Paddle-Librosa will provide almost identical features to standard paddlelibrosa functions (numerical difference less than 1e-5).

## Install
```bash
$ git clone https://github.com/AgentMaker/Paddle-Librosa.git
$ pip install Paddle-Librosa/
```

## Examples 1

Extract Log mel spectrogram with Paddle-Librosa.

```python
import paddle
import paddlelibrosa as pl

batch_size = 16
sample_rate = 22050
win_length = 2048
hop_length = 512
n_mels = 128

batch_audio = paddle.uniform((batch_size, sample_rate))  # (batch_size, sample_rate)

# Paddle-Librosa feature extractor the same as librosa.feature.melspectrogram()
feature_extractor = paddle.nn.Sequential(
    pl.Spectrogram(
        hop_length=hop_length,
        win_length=win_length,
    ), pl.LogmelFilterBank(
        sr=sample_rate,
        n_mels=n_mels,
        is_log=False, # Default is true
    ))
batch_feature = feature_extractor(batch_audio) # (batch_size, 1, time_steps, mel_bins)
```

## Examples 2

Extracting spectrogram, then log mel spectrogram, STFT and ISTFT with Paddle-Librosa.

```python
import paddle
import paddlelibrosa as pl

batch_size = 16
sample_rate = 22050
win_length = 2048
hop_length = 512
n_mels = 128

batch_audio = paddle.empty(batch_size, sample_rate).uniform_(-1, 1)  # (batch_size, sample_rate)

# Spectrogram
spectrogram_extractor = pl.Spectrogram(n_fft=win_length, hop_length=hop_length)
sp = spectrogram_extractor.forward(batch_audio)   # (batch_size, 1, time_steps, freq_bins)

# Log mel spectrogram
logmel_extractor = pl.LogmelFilterBank(sr=sample_rate, n_fft=win_length, n_mels=n_mels)
logmel = logmel_extractor.forward(sp)   # (batch_size, 1, time_steps, mel_bins)

# STFT
stft_extractor = pl.STFT(n_fft=win_length, hop_length=hop_length)
(real, imag) = stft_extractor.forward(batch_audio)
# real: (batch_size, 1, time_steps, freq_bins), imag: (batch_size, 1, time_steps, freq_bins) #

# ISTFT
istft_extractor = pl.ISTFT(n_fft=win_length, hop_length=hop_length)
y = istft_extractor.forward(real, imag, length=batch_audio.shape[-1])    # (batch_size, samples_num)
```

## Contact
AgentMaker: AgentMaker@163.com

## External links
Other related repos include:

torchlibrosa: https://github.com/qiuqiangkong/torchlibrosa

