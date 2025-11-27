import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft

# ====== 1. โหลดไฟล์เสียง ======
sample_rate, data = wavfile.read("sound/original.wav")

# หากเป็น stereo → แปลงเป็น mono
if len(data.shape) > 1:
    data = data.mean(axis=1)

# เก็บไฟล์ original
wavfile.write("sound/original.wav", sample_rate, data.astype(np.int16))

# ====== 2. ทำ FFT ======
N = len(data)
signal_fft = fft(data)

# ====== 3. ตัดความถี่สูง ======
cutoff = 4000   # Hz (กำหนดเองได้)
freqs = np.fft.fftfreq(N, 1 / sample_rate)

# สร้าง mask
mask = np.abs(freqs) <= cutoff
filtered_fft = signal_fft * mask

# ====== 4. inverse FFT กลับเป็นสัญญาณ ======
filtered_signal = np.real(ifft(filtered_fft))

# Normalize ป้องกันเสียงแตก
filtered_signal = filtered_signal / np.max(np.abs(filtered_signal))
filtered_signal = (filtered_signal * 32767).astype(np.int16)

# ====== 5. บันทึกไฟล์ ======
wavfile.write("filtered.wav", sample_rate, filtered_signal)

print("✔ สร้างไฟล์ original.wav และ filtered.wav เรียบร้อย")
