# Description:
APT (Automatic Picture Transmission) is a radio transmission system used by weather satellites, such as the NOAA-15, -18, and -19, which can all easily be received using a simple dipole antenna and a cheap SDR receiver. Recording a sample of one of these transmissions, it should be fairly easy for us to decode the image being transmitted using some Python!

Of course, there's easier ways of doing this. NOAA-APT (https://noaa-apt.mbernardi.com.ar/) is my favorite way of doing this, but others seem to prefer WXtoIMG (https://wxtoimgrestored.xyz/), which is somewhat abandoned and hard to get to work.

While these are good solutions, I wanted to see how the process works by hand, in order to get some experience with scipy and digital signal processing in general. At least initially, I'm inspired by this Medium post https://medium.com/swlh/decoding-noaa-satellite-images-using-50-lines-of-code-3c5d1d0a08da by Dmitrii Eliuseev. I want to do my own take on this afterwards, though.
