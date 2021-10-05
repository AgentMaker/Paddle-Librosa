import paddle
import paddlelibrosa as pl

batch_size = 16
sample_rate = 22050
win_length = 2048
hop_length = 512
n_mels = 128

batch_audio = paddle.uniform((batch_size, sample_rate))  # (batch_size, sample_rate)

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