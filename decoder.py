# Author: Jacob Dawson
#
# This file does the decoding of an already-resampled (?) APT recording.

# just going to copy the Medium post by Dmitrii Eliuseev to begin with
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np

from PIL import Image

fs, data = wav.read('recordings/noaa18-march-9.wav')
if len(data.shape)==2:
    print(f"Data in two channels, using the first one")
    data = data[:,0] # only need 1 channel

expectedRate = 11025
if fs!=expectedRate:
    print(f"Resampling from {fs} to {expectedRate}")
    factor = fs // expectedRate
    stop = (len(data) // factor) * factor
    # resample to new rate:
    data = data[0:stop:factor]
    fs = expectedRate
else:
    print(f"No need to resample, recording is already at {fs}")

def hilbert(data):
    analytical_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope
data_am = hilbert(data)

frame_width = int(0.5*fs)
w, h = frame_width, data_am.shape[0]//frame_width
image = Image.new('RGB', (w, h))
px, py = 0, 0
for p in range((data_am.shape[0]//frame_width) * frame_width):
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
