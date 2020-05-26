<div align="center">
  <img src="https://raw.githubusercontent.com/Supermaxman/SpaceGAN/master/samples/SpaceGAN-images-30500-128x128.png">
</div>


## SpaceGan
Generative Adversarial Network on Hubble deep space scans from NASA


## Examples

Training Time-Lapse:
<div align="center">
  <img src="https://github.com/Supermaxman/SpaceGAN/blob/master/samples/SpaceGAN-training.gif?raw=true">
</div>


Best Results:
<div align="center">
  <img src="https://raw.githubusercontent.com/Supermaxman/SpaceGAN/master/samples/SpaceGAN-images-30500-128x128.png">
</div>

## TODO
 - InfoGAN Continuous noise would be cool
    - Number of stars
    - Colors
    - Brightness
 - Log interpolation of random noise variables
   - More images with fixed z and changing InfoGan c,
   - Fixed z except one, interpolate
   - Interpolate all of z on a sphere, see tips: https://github.com/dribnet/plat
 - Larger images
   - 256x256 or 512x512 unlikely on GTX 1080, will need better hardware
 - Write up technical paper for writing experience, post on arxiv
   - Include discussion on dataset construction and preparation
   - Discuss challenges with dataset
   - Include experiments and samples
   - Cite relevant prior work and papers implemented
