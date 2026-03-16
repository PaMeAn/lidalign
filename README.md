# Lidalign Python Package
Welcome to the Lidalign Python package! This package contains some python scripts for the alignment and pointing calibration of scanning wind lidars offshore! 

<img src="docs/figures/ScanningLidar_atTP_nolidar.jpg" alt="Installation procedure" width="300"/>



Mainly, the package contains the following scripts:
- North alignment using azimuth hard target mapping [see Rott et al. 2022](https://wes.copernicus.org/articles/7/283/2022/)
- External Lidar alignment (pitch/roll) determination using Sea-Surface Leveling (SSL) [see Rott et al. 2022](https://wes.copernicus.org/articles/7/283/2022/)
- Determination of systematic elevation offset from Sea-Surface Calibration (SSC) --> publication currently in the making (and see [Gramitzky et al. 2025](https://wes.copernicus.org/preprints/wes-2025-191/))
- Scripts for the evaluation and visualisation of conventional hard target mapping

We do not claim full completeness and hope for contributions from the community.

## Documentation:
For the documentation of the code and some examples, see the [HTML docs of lidalign]().

## How to use:
For the installation, we recommend the use of [Astral uv](https://docs.astral.sh/uv/), a fast python package manager. Alternatively, you can replace the `uv pip` commands with `python -m pip` in the following. 

### Installation for direct use
Installation of the package (including the codes for the evaluation for the publication) should be possible through:

```
SSH:
uv pip install "git+ssh://git@..."

HTTPS:
uv pip install "git+https://gitlab."
```
### Development
If you want to develop the packages, clone the repository first:
```
git clone https://
```
and then install editable:
```
cd lidalign
uv pip install -e .
```
or 
```
uv pip install -e .[dev]
```
The "-e" indicates, the package is editable, so scripts can be changed and updated. If you do not plan to contribute, you can install without -e.

If you made edits, please create branches and create merge requests, so the code keeps developing!

# How to cite
If used, please cite with the following:
```
Paul Meyer, Andreas Rott, Jörge Schneemann, Janna K. Seifer et al. from ForWind (2026). Lidalign python package (DOI to be added)
```

The general work please cite with:
```
Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kuhn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).
```
and 
```
Rott, A., Schneemann, J., Theuer, F., Trujillo Quintero, J. J., & Kühn, M. (2022). Alignment of scanning lidars in offshore wind farms. Wind Energy Science, 7(1), 283–297. https://doi.org/10.5194/wes-7-283-2022
```


