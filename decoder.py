# Author: Jacob Dawson
#
# This file does the decoding of an APT recording.

import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np

from os import listdir, path, makedirs

from PIL import Image
from PIL.ImageOps import autocontrast


def readFile(filename):
    # this function takes a .wav file and returns its frequency and data.
    # Because we only need the signal in mono, we only return one channel if
    # the given recording has stereo sound.

    print("Reading", filename)
    fs, data = wav.read(filename)
    if len(data.shape) == 2:
        print(f"Data in two channels, using the first one")
        data = data[:, 0]  # only need 1 channel

    return fs, data


def resample(fs, data):
    # this function uses scipy's resample function to resample a signal
    # into the correct sample rate. Normally this will be a downsample,
    # so don't worry about any aliasing issues or whatever

    expectedRate = 20800
    if fs != expectedRate:
        print(f"Resampling from {fs} to {expectedRate}")
        data = signal.resample(data, int((data.shape[0] / fs) * expectedRate))

        # sample rate is now expected
        fs = expectedRate
    else:
        print(f"No need to resample, recording is already at {fs}")

    return fs, data


def hilbert(data):
    # This function performs the hilbert transform and returns its absolute
    # value on our data. Scipy has a good visualization of this at the bottom
    # of the page:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

    analytical_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytical_signal)

    return amplitude_envelope


def kernelFilter(fs, data_am):
    # this applies a median filter kernel size 5,
    # and then keeps only the signal at each 3rd position.

    data_am = data_am[: ((data_am.size // 5) * 5)]  # signal's size a factor of 5
    data_am = signal.medfilt(data_am, 5)
    data_am = data_am.reshape(len(data_am) // 5, 5)[:, 3]
    fs = fs // 5

    return fs, data_am


def toImgValues(data_am):
    # this function simply normalizes our raw data into a range of (0,255),
    # the range of a greyscale image. Note that data_am, both given and
    # returned, is expected to be in the shape (x,), that is a one-dimensional
    # array. However, I think it *should* work otherwise.

    print("Determining pixel values")
    # erring on the side of caution and just saving the raw image, so let's
    # simply scale everything:
    max = np.amax(data_am)
    min = np.amin(data_am)
    data_am = ((data_am - min) / max) * 255  # values are in (0,255)
    # and just to be safe:
    data_am = np.clip(data_am, 0, 255)

    return data_am


def alignSignal(data_am):
    # this function takes the signal and returns data on how the signal ought
    # to be aligned.

    print("Aligning signal for image")
    # https://github.com/zacstewart/apt-decoder has some great code for this,
    # copying it and changing things slightly. It's very smart to make a little
    # "example" and then compare slices of our real signal to figure out how
    # similar the potential slice is. Unfortunately it needs to be done with
    # some python iteration, but I don't think this is avoidable

    # we compare this platonic ideal of a telemetry signal to our real data:
    syncA = [0, 128, 255, 128] * 7 + [0] * 7

    # list of maximum correlations found: (index, value)
    peaks = [(0, 0)]

    # minimum distance between peaks
    minDistance = 2000

    # need to shift the values down to get meaningful correlation values
    signalshifted = [x - 128 for x in data_am]
    syncA = [x - 128 for x in syncA]
    for i in range(len(data_am) - len(syncA)):
        corr = np.dot(syncA, signalshifted[i : i + len(syncA)])

        # if previous peak is too far, keep it and add this value to the
        # list as a new peak
        if i - peaks[-1][0] > minDistance:
            peaks.append((i, corr))

        # else if this value is bigger than the previous maximum, set this one
        elif corr > peaks[-1][1]:
            peaks[-1] = (i, corr)

    return peaks


def createGreyscaleImg(data_am, peaks, frame_width):
    # given a signal with shape (x,) and some info on how wide the image is
    # expected to be and how the signal ought to be aligned, create an aligned
    # greyscale image and return it.

    print("Creating image from array")
    # create image matrix starting each line on the peaks found
    matrix = []
    for ind, _ in peaks:
        row = data_am[ind : ind + frame_width]
        if row.size != frame_width:
            break
        matrix.append(row)
    img = np.array(matrix)

    return img


def saveImg(img, outputFolder, filename, enhanceContrast=False):
    # this function simply takes an ndarray which may be greyscale or RGB and
    # saves it to a png file

    # first, we want to change the image from 32-bit floats to 8-bit integers:
    img = img.astype(np.int8)

    if len(img.shape) == 2:
        image = Image.fromarray(img, mode="L")
    elif len(img.shape) == 3:
        image = Image.fromarray(img, mode="RGB")
    else:
        print("Unknown format, cannot save image")
        return
        # dunno!

    if image.mode != "RGB":
        image = image.convert("RGB")

    if enhanceContrast:
        image = autocontrast(image)

    output_path = outputFolder + "/" + filename.split("\\")[-1].split(".")[0] + ".png"
    print(f"Writing image to {output_path}")
    image.save(output_path)


def process(filename, outputFolderRawImgs, outputFalseColorImages):
    # the "main" function. Given a filename and an expected output folder, this
    # function calls the above functions to turn the given .wav file into an
    # image from a weather sat!

    # read the given file:
    fs, data = readFile(filename)

    # resample to 20800
    fs, data = resample(fs, data)

    # perform hilbert transform
    data_am = hilbert(data)

    # kernel filter, reducing sample rate and improving signal (?)
    fs, data_am = kernelFilter(fs, data_am)

    frame_width = fs // 2  # should now be 2080, if my math is correct

    # transform values into the range (0, 255)
    data_am = toImgValues(data_am)

    # determine how to align image
    peaks = alignSignal(data_am)

    # turn signal into a 2D ndarray based on the alignment numbers above
    img = createGreyscaleImg(data_am, peaks, frame_width)

    # let's save that greyscale to a file, call it rawImages
    saveImg(img, outputFolderRawImgs, filename)

    print("Combining channels and creating a false color image")
    chA = img[:, :1040]
    chB = img[:, 1040:]
    blankChannel = np.zeros(chA.shape)

    # let's just see what overlaying the two channels does for now
    stackedImg = np.stack((chA*1.5, (chA*.7)+(chB*.3), chB*0.5), axis=-1)
    stackedImg = toImgValues(stackedImg)
    saveImg(stackedImg, outputFalseColorImages, filename, enhanceContrast=True)


if __name__ == "__main__":
    recordings = "recordings"
    outputRawImages = "rawImages"
    outputFalseColorImages = "falseColorImages"
    if not path.isdir(recordings):
        makedirs(recordings)
    if not path.isdir(outputRawImages):
        makedirs(outputRawImages)
    if not path.isdir(outputFalseColorImages):
        makedirs(outputFalseColorImages)
    if len(listdir(recordings)) == 0:
        print("Place recoded APT signals in the 'recordings' folder!")
    for filename in listdir(recordings):
        print("\n")
        f = path.join(recordings, filename)
        process(f, outputRawImages, outputFalseColorImages)
