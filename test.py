# Author: Jacob Dawson
#
# This file is just testing stuff out. You'll find all kinds of loose code in
# here, I hope.

import scipy.io.wavfile as wav

fs, data = wav.read("recordings/noaa18-march-9.wav")
print(fs, data.shape)

fs, data = wav.read("recordings/noaa18-march-9-resampled.wav")
print(fs, data.shape)
