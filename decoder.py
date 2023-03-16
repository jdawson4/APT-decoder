# Author: Jacob Dawson
#
# This file does the decoding of an already-resampled (?) APT recording.

# just going to copy the Medium post by Dmitrii Eliuseev to begin with
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

fs, data = wav.read('recordings/noaa18-march-9.wav')
if len(data.shape)==2:
    data = data[:,0] # only need 1 channel
#data_crop = data[20*fs:21*fs]
#plt.figure(figsize=(12,4))
#plt.plot(data_crop)
#plt.xlabel("Samples")
#plt.ylabel("Amplitude")
#plt.title("Signal")
#plt.show()

resample = 4
data = data[::resample]
fs = fs//resample

def hilbert(data):
    analytical_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope
data_am = hilbert(data)

frame_width = int(0.5*fs)
w, h = frame_width, data_am.shape[0]//frame_width
image = Image.new('RGB', (w, h))
px, py = 0, 0
for p in range(data_am.shape[0]):
    lum = int(data_am[p]//32 - 32)
    if lum < 0: lum = 0
    if lum > 255: lum = 255
    image.putpixel((px, py), (lum, lum, lum))
    px += 1
    if px >= w:
        if (py % 50) == 0:
            print(f"Line saved {py} of {h}")
        px = 0
        py += 1
        if py >= h:
            break

#image = image.resize((w, 4*h))
#plt.imshow(image)
#plt.savefig('images/img.png', dpi=500)
#plt.show()
image.save('images/img.png')
