import numpy as np
import matplotlib.pyplot as plt

# 1. Create dummy signal data
fs = 20000  # Sample rate (Hz)
t = np.linspace(0, 1, fs)  # 1 second duration
spindle_freq = 50  # 3000 RPM / 60 = 50 Hz

# Generate "New Tool" signal (Clean, mostly fundamental freq)
sig_new = 0.2 * np.sin(2 * np.pi * spindle_freq * t) + \
          0.05 * np.sin(2 * np.pi * 2 * spindle_freq * t) + \
          np.random.normal(0, 0.05, len(t))

# Generate "Worn Tool" signal (Noisy, high harmonics, sidebands)
sig_worn = 0.5 * np.sin(2 * np.pi * spindle_freq * t) + \
           0.4 * np.sin(2 * np.pi * 2 * spindle_freq * t) + \
           0.3 * np.sin(2 * np.pi * 3 * spindle_freq * t) + \
           np.random.normal(0, 0.2, len(t)) # More noise (friction)

# 2. Perform FFT
n = len(t)
freq = np.fft.rfftfreq(n, d=1/fs)
fft_new = np.abs(np.fft.rfft(sig_new)) / n
fft_worn = np.abs(np.fft.rfft(sig_worn)) / n

# 3. Plot the FFT Spectrum
plt.figure(figsize=(10, 5))
plt.plot(freq, fft_worn, 'r', alpha=0.7, label='Worn Tool')
plt.plot(freq, fft_new, 'g', alpha=0.9, label='New Tool')

plt.xlim(0, 500)  # Zoom in on 0-500 Hz range
plt.ylim(0, 0.3)
plt.title("Figure 4-3: Frequency Domain (FFT) Analysis")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()