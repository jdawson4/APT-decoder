# Author: Jacob Dawson
#
# This file does the decoding of an already-resampled (?) APT recording.

# just going to copy the Medium post by Dmitrii Eliuseev to begin with
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np

from os import listdir, path, makedirs

from PIL import Image

def modified(filename, outputFolder):
    print("Reading", filename)
    fs, data = wav.read(filename)
    if len(data.shape) == 2:
        print(f"Data in two channels, using the first one")
        data = data[:, 0]  # only need 1 channel

    # we worry first about sample rate:
    expectedRate = 20800
    if fs != expectedRate:
        print(f"Resampling from {fs} to {expectedRate}")
        data = signal.resample(data, int((data.shape[0] / fs) * expectedRate))

        # sample rate is now expected
        fs = expectedRate
    else:
        print(f"No need to resample, recording is already at {fs}")

    def hilbert(data):
        analytical_signal = signal.hilbert(data)
        amplitude_envelope = np.abs(analytical_signal)
        return amplitude_envelope

    data_am = hilbert(data)

    # this is complicated. It applies a median filter kernel size 5,
    # and then keeps only the signal at each 3rd position.
    data_am = data_am[:((data_am.size // 5) * 5)] # signal's size a factor of 5
    data_am = signal.medfilt(data_am, 5)
    data_am = data_am.reshape(len(data_am) // 5, 5)[:, 3]
    fs = fs // 5

    frame_width = fs // 2  # should now be 2080, if my math is correct

    print("Determining pixel values")
    # erring on the side of caution and just saving the raw image, so let's
    # simply scale everything:
    max = np.amax(data_am)
    min = np.amin(data_am)
    data_am = ((data_am - min) / max) * 255  # values are in (0,255)
    # and just to be safe:
    data_am = np.clip(data_am, 0, 255)

    print("Aligning signal for image")
    # https://github.com/zacstewart/apt-decoder has some great code for this,
    # copying it and changing things slightly. It's very smart to make a little
    # "example" and then compare slices of our real signal to figure out how
    # similar the potential slice is. Unfortunately it needs to be done with
    # some python iteration, but I don't think this is avoidable

    # we compare this platonic ideal of a telemetry signal to our real data:
    syncA = [0, 128, 255, 128]*7 + [0]*7

    # list of maximum correlations found: (index, value)
    peaks = [(0, 0)]

    # minimum distance between peaks
    minDistance = 2000

    # need to shift the values down to get meaningful correlation values
    signalshifted = [x-128 for x in data_am]
    syncA = [x-128 for x in syncA]
    for i in range(len(data_am)-len(syncA)):
        corr = np.dot(syncA, signalshifted[i : i+len(syncA)])

        # if previous peak is too far, keep it and add this value to the
        # list as a new peak
        if i - peaks[-1][0] > minDistance:
            peaks.append((i, corr))

        # else if this value is bigger than the previous maximum, set this one
        elif corr > peaks[-1][1]:
            peaks[-1] = (i, corr)

    print("Creating image from array")
    # create image matrix starting each line on the peaks found
    matrix = []
    for ind, _ in peaks:
        row = data_am[ind : ind + frame_width]
        if row.size != frame_width:
            break
        matrix.append(row)
    data_am = np.array(matrix)

    image = Image.fromarray(data_am)

    if image.mode != "RGB":
        image = image.convert("RGB")

    output_path = outputFolder + "/" + filename.split("\\")[-1].split(".")[0] + ".png"
    print(f"Writing image to {output_path}")
    image.save(output_path)

if __name__ == "__main__":
    # filename = 'recordings/noaa18-march-9.wav'
    # original(filename)
    # modified(filename)
    recordings = "recordings"
    outputRawImages = "rawImages"
    if not path.isdir(recordings):
        makedirs(recordings)
    if not path.isdir(outputRawImages):
        makedirs(outputRawImages)
    if len(listdir(recordings)) == 0:
        print("Place recoded APT signals in the 'recordings' folder!")
    for filename in listdir(recordings):
        print("\n")
        f = path.join(recordings, filename)
        modified(f, outputRawImages)
