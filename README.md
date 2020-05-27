<div align="center">
  <img src="https://hubblesite.org/files/live/sites/hubble/files/home/_images/hs-images-hubble-side12x12.jpg?t=tn370">
</div>


## SpaceGan
Generative Adversarial Network on Hubble deep space scans from NASA


## Examples

Samples:
<div align="center">
  <img src="https://github.com/Supermaxman/SpaceGAN/blob/master/samples/independent.png?raw=true">
</div>

Interpolated Samples:
<div align="center">
  <img src="https://github.com/Supermaxman/SpaceGAN/blob/master/samples/interpolate.png?raw=true">
</div>

Anchor Interpolated Examples:
<div align="center">
  <img src="https://github.com/Supermaxman/SpaceGAN/blob/master/samples/anchor_interpolate.png?raw=true">
</div>


## Training

Training Time-Lapse:
<div align="center">
  <img src="https://github.com/Supermaxman/SpaceGAN/blob/master/samples/SpaceGAN-training.gif?raw=true">
</div>

## TODO
 - InfoGAN Continuous noise would be cool
    - Number of stars
    - Colors
    - Brightness
 - Log interpolation of random noise variables
   - More images with fixed z and changing InfoGan c
 - Larger images
   - 256x256 or 512x512 unlikely on GTX 1080, will need better hardware
 - Write up technical paper for writing experience, post on arxiv
   - Include discussion on dataset construction and preparation
   - Discuss challenges with dataset
   - Include experiments and samples
   - Cite relevant prior work and papers implemented
