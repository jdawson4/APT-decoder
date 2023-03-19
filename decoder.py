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
    # we have a few options:

    # not sure if I love this. It might filter out some noise, but we'll
    # deliberately be clipping out useful signal.
    # data_am = np.clip(data_am, np.percentile(data_am, 0.0), np.percentile(data_am, 90.0))
    # these bounds matter a lot! Mess around with this ^!

    # simply scale everything:
    max = np.amax(data_am)
    min = np.amin(data_am)
    data_am = ((data_am - min) / max) * 255  # scale values (0,255)

    # this is how the original article does it. He says it might be better
    # for different antennae
    # data_am = (data_am // 32) - 32
    # data_am = data_am // 32

    # this no matter what:
    data_am = np.clip(data_am, 0, 255)

    print("Generating image array")
    data_am = np.reshape(data_am, (h, w))

    print("Aligning image")
    # the plan here is to simply find the brightest column, and then make that
    # the center. This column will presumably be the center of the white strip
    # down the middle which is (I think) meant to be for telemetry
    columnAvs = np.mean(data_am, axis=1).flatten()
    darkestColumnVal = 0
    darkestColumnInd = 0
    i = 0
    for columnAv in columnAvs:
        if columnAv < darkestColumnVal:
            darkestColumnVal = columnAv
            darkestColumnInd = i
        i+=1
    print(darkestColumnInd)
    data_am = np.concatenate((data_am[:,darkestColumnInd:], data_am[:,:darkestColumnInd]), axis=1)

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
