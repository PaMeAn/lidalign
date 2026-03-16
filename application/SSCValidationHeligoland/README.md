# Readme for SSC Validation at Heligoland
This directory contains the scripts for the generation of results for the SSC paper:
> Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kuhn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation)
> 

More information can be found 
- in the parallel publication
- some documentation here
- The remaining scripts in this folder


## Content of this directory:
- [00_TestSetup_Heligoland.ipynb](00_TestSetup_Heligoland.ipynb) - show the setup and perform hard target mapping for scanner offset determination (reference)
- [10_StarringTests_Heligoland.ipynb](10_StarringTests_Heligoland.ipynb) - Test (a) investigations, starring with the lidar into the water
- [20_CornerScanTest_Heligoland.ipynb](20_CornerScanTest_Heligoland.ipynb) - Test (b) investigations, Scan of a corner that is partially in the water 
- [30_SSC_Scan_Evaluation.ipynb](30_SSC_Scan_Evaluation.ipynb) - Test (c) investigations, SSC scans of the surrounding water 
  
- [98_SSC_sensitivity_numerical.ipynb](98_SSC_sensitivity_numerical.ipynb) - investigates the sensititivy of the SSC to possible errors in the range (purely numerically)
- [99_Importance_of_accurate_elevation.ipynb](99_Importance_of_accurate_elevation.ipynb) - demonstrates, why accurate elevation measurements are necessary
