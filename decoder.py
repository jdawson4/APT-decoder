# Author: Jacob Dawson
#
# This file does the decoding of an already-resampled (?) APT recording.

# just going to copy the Medium post by Dmitrii Eliuseev to begin with
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np

from os import listdir, path, makedirs

from PIL import Image

# from PIL import ImageEnhance


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

        # pretty bad method, only works for certain imports:
        """factor = fs // expectedRate
        # resample to new rate:
        data = data[::factor]
        fs = fs // factor
        print(f"Factor: {factor}\nNew rate: {fs}")"""

        # slow, but probably more correct:
        data = signal.resample(data, int((data.shape[0] / fs) * expectedRate))

        # and truncate because we have reshaping to do:
        truncate = expectedRate * int(len(data) // expectedRate)
        data = data[:truncate]

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
    data_am = signal.medfilt(data_am, 5)
    data_am = data_am.reshape(len(data_am) // 5, 5)[:, 3]
    fs = fs // 5

    frame_width = fs // 2  # should now be 2080, if my math is correct

    print(f"Limiting data to a factor of {frame_width}")
    data_am = data_am[: (data_am.shape[0] // frame_width) * frame_width]

    w, h = frame_width, data_am.shape[0] // frame_width  # shape of final img

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
    # but it relies on python iteration rather than numpy functions
    # aligning and matrixizing
    syncA = [0, 128, 255, 128]*7 + [0]*7

    # list of maximum correlations found: (index, value)
    peaks = [(0, 0)]

    # minimum distance between peaks
    mindistance = 2000

    # need to shift the values down to get meaningful correlation values
    signalshifted = [x-128 for x in data_am]
    syncA = [x-128 for x in syncA]
    for i in range(len(data_am)-len(syncA)):
        corr = np.dot(syncA, signalshifted[i : i+len(syncA)])

        # if previous peak is too far, keep it and add this value to the
        # list as a new peak
        if i - peaks[-1][0] > mindistance:
            peaks.append((i, corr))

        # else if this value is bigger than the previous maximum, set this
        # one
        elif corr > peaks[-1][1]:
            peaks[-1] = (i, corr)

    # create image matrix starting each line on the peaks found
    matrix = []
    for i in range(len(peaks) - 1):
        matrix.append(data_am[peaks[i][0] : peaks[i][0] + 2080])
    data_am = np.array(matrix)

    print("Creating image from array")
    image = Image.fromarray(data_am)

    if image.mode != "RGB":
        image = image.convert("RGB")

    # print("Enhancing contrast")
    # image = ImageEnhance.Contrast(image).enhance(1.65) # this value changes things quite a bit. I think it looks best here

    output_path = outputFolder + "/" + filename.split("\\")[-1].split(".")[0] + ".png"
    print(f"Writing image to {output_path}")
    image.save(output_path)


def original(filename):
    fs, data = wav.read(filename)
    if len(data.shape) == 2:
        print(f"Data in two channels, using the first one")
        data = data[:, 0]  # only need 1 channel

    resample = 4
    data = data[::resample]
    fs = fs // resample

    def hilbert(data):
        analytical_signal = signal.hilbert(data)
        amplitude_envelope = np.abs(analytical_signal)
        return amplitude_envelope

    data_am = hilbert(data)
    frame_width = int(0.5 * fs)
    w, h = frame_width, data_am.shape[0] // frame_width
    image = Image.new("RGB", (w, h))
    px, py = 0, 0
    for p in range(data_am.shape[0]):
        lum = int(data_am[p] // 32 - 32)
        if lum < 0:
            lum = 0
        if lum > 255:
            lum = 255
        # image.putpixel((px, py), (0, lum, 0))# green map
        image.putpixel((px, py), (lum, lum, lum))  # greyscale
        px += 1
        if px >= w:
            if (py % 50) == 0:
                print(f"Line saved {py} of {h}")
            px = 0
            py += 1
            if py >= h:
                break
    image = image.resize((w, 4 * h))
    output_path = "images/" + filename.split("\\")[-1].split(".")[0] + ".png"
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
