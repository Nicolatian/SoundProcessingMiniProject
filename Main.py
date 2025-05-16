import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import sounddevice as sd
from scipy.signal import get_window

# === Change parameters ===
file_path = 'Bôa - Duvet (Lyrics).wav'
sampling_rate = 44100 # check in sonic visualzer, mine is 44100
fft_size = 4096 #points used in FFT
window_length = 4096 #samples pr segment for spectogram
window_type = 'hann' # hann', 'boxcar', "hammin"
overlap_size = window_length // 2 #how much overlap?
use_db_scaling = True #makes it logaritmics

original_sampling_rate, audio_data = wavfile.read(file_path)

#Normalizes the audio to fit between -1 and 1.
if audio_data.ndim == 2:
    audio_data = audio_data.mean(axis=1)
audio_data = audio_data / np.max(np.abs(audio_data))

# TIME DOmain plot
duration_in_seconds = len(audio_data) / sampling_rate
time_axis = np.linspace(0, duration_in_seconds, len(audio_data))

plt.figure(figsize=(12, 4))
plt.plot(time_axis, audio_data)
plt.title("Time Domain Signal")
plt.xlabel("Time in seconds")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# FFT
fft_result = np.fft.fft(audio_data, n=fft_size)
magnitude_spectrum = np.abs(fft_result)[:fft_size // 2]
if use_db_scaling:
    magnitude_spectrum = 20 * np.log10(magnitude_spectrum + 1e-10)  #Magnitude is 20 linaer

frequency_axis = np.fft.fftfreq(fft_size, d=1 / sampling_rate)[:fft_size // 2]

plt.figure(figsize=(12, 4))
plt.plot(frequency_axis, magnitude_spectrum)
plt.xlim(0, 5000)
plt.title("Frequency Domain")
plt.xlabel("Frequency in [Hz]")
plt.ylabel("Magnitude [{}]".format("dB" if use_db_scaling else "Linear"))
plt.grid(True)
plt.tight_layout()
plt.show()

# Long term spectrum is crated here.
# Break audio into overlapping segments
step_size = window_length - overlap_size
num_segments = (len(audio_data) - window_length) // step_size + 1
window = get_window('hann', window_length)
store_spectrum = np.zeros(fft_size // 2)

#loops thorugh segments.
for i in range(num_segments):
    start = i * step_size
    segment = audio_data[start:start + window_length] * window
    segment_fft = np.fft.fft(segment, n=fft_size)
    magnitude = np.abs(segment_fft[:fft_size // 2])
    store_spectrum += magnitude

average_spectrum = store_spectrum / num_segments
if use_db_scaling:
    average_spectrum = 10 * np.log10(average_spectrum + 1e-10)

plt.figure(figsize=(12, 4))
plt.plot(frequency_axis, average_spectrum)
plt.xlim(0, 5000)
plt.title("Long-Term Average Spectrum")
plt.xlabel("Frequency in [Hz]")
plt.ylabel("Magnitude [{}]".format("dB" if use_db_scaling else "Linear"))
plt.grid(True)
plt.tight_layout()
plt.show()

# SPEcTROGRAM
frequencies, times, spectogram_energy = spectrogram(
    audio_data,
    fs=sampling_rate,
    window=window_type,
    nperseg=window_length,
    noverlap=overlap_size
)
# does the decibel scaling... if active, else logarhythm
if use_db_scaling:
    spectogram_energy = 10 * np.log10(spectogram_energy + 1e-10)  # Power is 10

plt.figure(figsize=(12, 5))
plt.ylim(0, 5000)
plt.pcolormesh(times, frequencies, spectogram_energy, shading='gouraud', cmap='jet')
plt.title("Spectogram")
plt.xlabel("Time in sec")
plt.ylabel("Frequency in [Hz]")
plt.colorbar(label='Power [{}]'.format("dB" if use_db_scaling else "Linear"))
plt.tight_layout()
plt.show()


# play le audio here?
sd.play(audio_data, samplerate=sampling_rate)
sd.wait()