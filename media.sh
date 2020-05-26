#!/usr/bin/env bash
# first rename files to match frame pattern
ffmpeg -framerate 10 -i "i (%d)*.png" out.gif
ffmpeg -i SpaceGAN-training.gif -c:v libvpx -crf 4 -b:v 8000K -auto-alt-ref 0 output.webm