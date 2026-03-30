# Lidalign Python Package
Welcome to the Lidalign Python package! This package contains some python scripts for the alignment and pointing calibration of scanning wind lidars offshore! 

<img src="docs/source/_static/pictures/ScanningLidar_atTP_nolidar.png" alt="Installation procedure" width="300"/>



Mainly, the package contains python code for the following lidar alignment actions:
- **North alignment** using azimuth hard target mapping [see Rott et al. 2022](https://wes.copernicus.org/articles/7/283/2022/)
- **External Lidar alignment** (pitch/roll) determination using Sea-Surface Leveling (SSL) [see Rott et al. 2022](https://wes.copernicus.org/articles/7/283/2022/)
- Determination of a **static elevation offset** from Sea-Surface Calibration (SSC) --> publication currently in the making (and see [Gramitzky et al. 2025](https://wes.copernicus.org/preprints/wes-2025-191/))
- Scripts for the evaluation and visualisation of **conventional hard target mapping**

We do not claim full completeness and hope for contributions from the community.

The work is heavily based on previous work by Rott et al.22, which can be found [in this zenodo repository](https://zenodo.org/records/5654919).

## Documentation:
For the documentation of the code and some examples, see the [HTML docs of lidalign](https://pamean.github.io/lidalign/). This includes:
- [Hard Target elevation mapping](https://pamean.github.io/lidalign/notebooks/Example_HTM_Elevation.html)
- [Sea Surface Leveling and Calibration](https://pamean.github.io/lidalign/notebooks/Example_SSC.html)
- [Lidar north alignment](https://pamean.github.io/lidalign/notebooks/Example_Azimuth_NorthAlignment.html)

## Experimental Validation of the SSC:
Also the experimental validation of Meyer et al. 2026 is documented [as HTML version of the jupyter notebooks](https://pamean.github.io/lidalign/ValidationCampaign.html).

## How to use:
For the installation, we recommend the use of [Astral uv](https://docs.astral.sh/uv/), a fast python package manager. Alternatively, you can replace the `uv pip` commands with `python -m pip` in the following. 

### Installation for direct use
Installation of the package (including the codes for the evaluation for the publication) is possible through:
```
uv pip install "uv pip install git+https://github.com/PaMeAn/lidalign.git"
```
### Development
If you want to develop the packages, clone the repository first:
```
git clone https://github.com/PaMeAn/lidalign.git
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

# Contributing
If you made edits, please create branches and create merge requests, so the code keeps developing! If you find issues or you are missing functionalities, let us know in the issues!

# How to cite
If used, please cite with the following:
```
Meyer, P. J., Rott, A., Seifert, J. K., & Schneemann, J. (2026). Lidalign python package (V.2.0). Zenodo. https://doi.org/10.5281/zenodo.19056909
```

The general work please cite with:
```
Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kühn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).
```
and 
```
Rott, A., Schneemann, J., Theuer, F., Trujillo Quintero, J. J., & Kühn, M. (2022). Alignment of scanning lidars in offshore wind farms. Wind Energy Science, 7(1), 283–297. https://doi.org/10.5194/wes-7-283-2022
```


