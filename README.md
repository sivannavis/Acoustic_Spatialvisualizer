# Acoustic_Spatialvisualizer

## Description
The Acoustic Spatial-Visualizer can generate spatial audio and visualize its moving track.

The spatializer can spatialize mono audio with continuous trajectory in 3D space given RIRs in different locations.

The visualizer uses APGD algorithm to takes in a 32-channel spatialized audio and outputs an energy map of azimuth and elevation every 100ms.

## Setup
Run `pip install -r requirements.txt`

## Spatialize
1. Set up `path_to_irs` and `IRS` in `spatialization.py`
2. Set up paths to read and write audio files
3. Run `python3 spatialization.py`

## Visualize
1. This script is based on the application of [this repo](https://github.com/adrianSRoman/DeepWaveTorch).
2. Download the submodule using `git clone https://github.com/adrianSRoman/DeepWaveTorch`
3. Set up current working directory to DeepWaveTorch in `visualization.py`
4. Set up `file_path` as the input audio file location and output folder in the end of the script
5. Set up `N_max_frames` as the number of frames to visualize (100ms per frame)
6. Run `python3 visualization.py`
7. Check out the images generated in the output map and generate .gif file of them using `convert -delay 20 -loop 0 $(ls -1 viz_output/*.jpg | sort -V) animated.gif`
