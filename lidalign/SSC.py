# %%
"""This module contains functions for the Sea-Surfac-Leveling (following Rott et al. 2022) and Sea-Surface-Calibration (following Meyer et al. 2026) from volumetric scans of the surrounding water surface.
Many parts of this module are directly based on the published scripts by Rott et. al 2021 (see https://zenodo.org/record/5654866).

SSL: Used to align the scanning lidar offshore (reduce roll and pitch as much as possible)
SSC: Calibration for scanning lidar elevation offset determination (determine offset and correct for afterwards in the software)

Authors: Paul Meyer, Andreas Rott
Date: 2026-06-01
"""
import numpy as np
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, curve_fit, least_squares
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass
from scipy.optimize import brentq
from numpy.typing import NDArray
from scipy.optimize import curve_fit


def print_welcome():
    print(
        r"""
     __              __             __                                                         
    / _\ ___  __ _  / _\_   _ _ __ / _| __ _  ___ ___                                          
    \ \ / _ \/ _` | \ \| | | | '__| |_ / _` |/ __/ _ \                                         
    _\ \  __/ (_| | _\ \ |_| | |  |  _| (_| | (_|  __/                                         
    \__/\___|\__,_| \__/\__,_|_|  |_|  \__,_|\___\___|                                         
       ___      _ _ _               _   _                                                      
      / __\__ _| (_) |__  _ __ __ _| |_(_) ___  _ __                                           
     / /  / _` | | | '_ \| '__/ _` | __| |/ _ \| '_ \                                          
    / /__| (_| | | | |_) | | | (_| | |_| | (_) | | | |                                         
    \____/\__,_|_|_|_.__/|_|  \__,_|\__|_|\___/|_| |_|   
    
    Scripts for the leveling (SSL) of wind lidars and calibration (SSC) of inherent scanner offsets.
    Authors: Paul Meyer, Andreas Rott et al. (ForWind)
    Version: V2.0
    Date: 2026-03-15
    """
    )


if not globals().get("_WELCOME_SHOWN"):
    print_welcome()
    _WELCOME_SHOWN = True


class PulseShape:
    """Base Class for pulse shapes of pulses used for wind lidar. Here, the pulse shape means the Range Weighting Function.

    Must have a "get_weighting" method implemented in subclass.
    """

    def get_inverse_cdf(self, dr: NDArray[np.float64]) -> NDArray[np.float64]:
        """get the inverse cummulative distribution function (CDF) of a pulse shape.

        Args:
            dr (NDArray[float]): radial distance to the pulse/range gate center. Resolution of r should be high enough for proper integration. The weighting function must be centered around 0
        Returns:
            NDArray[float]: CDF weighting function (discretized)
        """
        weighting = self.get_weighting(dr)
        cdf = np.cumsum(weighting)
        return 1 - (cdf / cdf.max())

    def get_weighting(self, dr: NDArray[np.float64]) -> NDArray[np.float64]:
        """get normalized weighting function

        Args:
            r (NDArray[np.float64]): distance from range gate center
        Returns:
            NDArray[np.float64]: function weighting
        """
        raise NotImplementedError(
            "get_weighting method must be implemented in subclass"
        )

    def __repr__(self):
        pass


class GaussianTruncatedPulse(PulseShape):
    """Pulse Shape of a truncated Gaussian Pulse. Child class of PulseShape.

    Pulse shape definition following Schlipf, D. (2015). PhD Thesis: Lidar-Assisted Control Concepts for Wind Turbines.
    """

    FWHM: float = 100
    FWHM_Width_ratio: float = 2.6

    def __init__(self, FWHM: float = 100, FWHM_Width_ratio: float = 2.6):
        """Initialize Pulse Object

        Args:
            FWHM (float, optional): Full Width at half maximum of the gaussian form. Defaults to 100m .
            FWHM_Width_ratio (float, optional): Width of the truncated gaussian, relative to the FWHM. Defaults to 2.6. Values for r>|FWHM * ratio / 2| are set to zero
        """
        self.FWHM = FWHM
        self.FWHM_Width_ratio = FWHM_Width_ratio

        # get the full gate length, where we truncate the Gaussian
        self.gate_length = self.FWHM * self.FWHM_Width_ratio

    def __repr__(self):
        return f"GaussianTruncatedPulse(FWHM={self.FWHM}, FWHM_Width_ratio={self.FWHM_Width_ratio}, gate_length={self.gate_length})"

    def get_weighting(self, dr: NDArray[np.float64]) -> NDArray[np.float64]:
        """get normalized weighting function

        Args:
            dr (np.array|float): distance from range gate center
        Returns:
            np.array|float: function weighting at dr
        """
        # ---------------------------- calculate parameter --------------------------- #
        sigma_L = self.FWHM / (2 * np.sqrt(2 * np.log(2)))
        func = 1 / (sigma_L * np.sqrt(2 * np.pi)) * np.exp(-(dr**2) / (2 * sigma_L**2))

        # -------------------------- truncate the function -------------------------- #
        truncated_func = np.where(np.abs(dr) < self.gate_length / 2, func, 0)
        truncated_func_norm = truncated_func / np.max(truncated_func)

        return truncated_func_norm

    @staticmethod
    def fit_weighting_to_data(
        r_data: NDArray[np.float64], signal_strength: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Fit the Pulse shape to given radial distribution of Signal strength. Can be used e.g., to fit the truncated gaussian to the CNR signal of a hard target.

        Args:
            r_data (NDArray[np.float64]): Radial distance from the lidar to the range gate center
            signal_strength (NDArray[np.float64]): Signal strength of the lidar signal (e.g., CNR) at the radial positions r_data
        Returns:
            NDArray[np.float64]: Parameters from the fit: [Rmax, CNRmax, FWHM]
        """

        pulse = GaussianTruncatedPulse()

        def _fit_func(r, *params):
            """function to calculate the modeled Pulse for the fit"""
            Rmax, CNRmax, FWHM = params
            r = r_data
            dr = r - Rmax
            pulse.FWHM = FWHM
            weight = pulse.get_weighting(dr)
            return CNRmax * weight

        # --------- Optimize the parameters for the best fit (least squares) --------- #
        params_opt, covs = curve_fit(
            _fit_func,
            r_data,
            signal_strength,
            p0=[r_data[np.argmax(signal_strength)], 10, 50],
        )

        return params_opt


def _calculate_FWHM(x, y):
    """Calculate the full FWHM of a given function y(x)"""
    ymin = np.min(y)
    ymax = np.max(y)
    ydelta = ymax - ymin
    yhalf = ymin + ydelta / 2
    x0 = x[y > yhalf][0]
    x1 = x[(x > x0) & (y < yhalf)][0]
    xdelta = x1 - x0
    return xdelta, x0, x1, yhalf


class PeakPulse(PulseShape):
    """Peak pulse shape (Peak to 1 at dr=0). Not realistic, only used for demonstration purposes."""

    def __init__(self, gate_length: float = 50):
        self.gate_length = gate_length

    def get_weighting(self, dr: float) -> float:
        """get normalized weighting function

        Args:
            dr (np.array|float): distance from range gate center
        Returns:
            np.array|float: function weighting
        """
        return np.where((dr < -0.01) | (dr > 0.01), 0, 1)


# ------------------------------------ xx ------------------------------------ #
# ------------------------------------ xx ------------------------------------ #
# ------------------------------- CNR functions ------------------------------ #
# ------------------------------------ xx ------------------------------------ #
# ------------------------------------ xx ------------------------------------ #


def linear2db(linearsignal: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert signal in linear scale into dB scale: S_{dB} = 10 log_{10}(S_{lin}),

    Args:
        linearsignal (float | NDArray[np.float64]): signal strength in linear scale

    Returns:
        float|NDArray[np.float64]: signal strength in dB scale
    """
    return 10 * np.log10(linearsignal)


def db2linear(dbsignal: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Convert signal in dB scale into linear scale: S_{lin} = 10^{S_{dB}/10},

    Args:
        dbsignal (float | NDArray[np.float64]): signal strength in dB scale

    Returns:
        float|NDArray[np.float64]: signal strength in linear scale
    """
    return 10 ** (dbsignal / 10)


def inverse_sigmoid(
    r: float | NDArray[np.float64], mid: float, down: float, up: float, growth: float
) -> float | NDArray[np.float64]:
    """Sigmoid function for fitting CNR data for water range detection.

    Following:
    Rott, A., Schneemann, J., Theuer, F., Trujillo Quintero, J. J., & Kühn, M. (2022). Alignment of scanning lidars in offshore wind farms. Wind Energy Science, 7(1), 283–297. https://doi.org/10.5194/wes-7-283-2022


    Args:
        r (float|NDArray[np.float64]): radial distance of range gate center
        up (float): upper CNR level before drop [dB]
        down (float): lower CNR level after drop [dB]
        mid (float): radial distance of the inflection point of the sigmoid -> corresponds to water range [m]
        growth (float): growth rate of the sigmoid function [-]

    Returns:
        Sigmoid function (r)
    """

    exponent = (r - mid) * growth
    exponent = np.clip(
        exponent, -700, 700
    )  # ensure no overflow (overflow starts at 709)
    return (up - down) / (1 + np.exp(exponent)) + down


def inverse_sigmoid_Gra24(
    r: float | NDArray[np.float64],
    mid: float,
    down: float,
    up: float,
    growth: float,
    lin_fac: float,
) -> NDArray[np.float64]:
    """Sigmoid function for fitting CNR data for water range detection, using a linear decay/increase before the drop.

    In contrast to the original function from Rott et al. (2022), a linear factor is added to account for a linear decrease before the drop.

    Following:
    Gramitzky, K., Jäger, F., Hildebrand, T., Gloria, N., Riechert, J., Steger, M., & Pauscher, L. (2024). Alignment calibration and correction for offshore wind measurements using scanning lidars. Journal of Physics: Conference Series, 2767(4), 042014. https://doi.org/10.1088/1742-6596/2767/4/042014

    Args:
        r (float|NDArray[np.float64]): radial distance of range gate center
        mid (float): radial distance of the inflection point of the sigmoid -> corresponds to water range [m]
        down (float): lower CNR level after drop [dB]
        up (float): upper CNR level before drop [dB]
        growth (float): growth rate of the sigmoid function [-]
        lin_fac (float): linear factor to adjust linear decrease before the drop [dB/m]

    Returns:
        sigmoid value, depending on input type of position
    """
    exponent = (r - mid) * growth
    exponent = np.clip(exponent, -700, 700)  #  overflow starts at 709
    return (up - down) * (1 + lin_fac * (r - mid)) / (1 + np.exp(exponent)) + down


# def improved_inverse_sigmoid(position, mid: float, down: float, up: float, growth: float, lin_fac: float):
#     return (1 + lin_fac * (position - mid)) * inverse_sigmoid(position, mid, down, up, growth)


def inverse_sigmoid_dbscale(
    r: float | NDArray[np.float64],
    mid: float,
    down: float,
    up: float,
    growth: float,
    lin_fac: float,
) -> NDArray[np.float64]:
    """Inverse sigmoid in DB scale. All inputs must be in dB scale.

    we do not recommend using this function

    Following:
    Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kühn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).

    Args:
        r (float | NDArray[np.float64]): radial distance of range gate center
        up (float): upper CNR level before drop [dB]
        down (float):  lower CNR level after drop [dB]
        mid (float): radial distance of the inflection point of the sigmoid -> corresponds to water range [m]
        growth (float): growth rate of the sigmoid function [-]
        lin_fac (float): linear factor to adjust linear decrease before the drop [dB/m]

    Returns:
        NDArray[np.float64]: Sigmoid function (r) in dB scale
    """
    # ---------- get inverse sigmoid in linear scale, then convert to dB --------- #
    inv_sigmo_db = linear2db(inverse_sigmoid(r, mid, 0, db2linear(up), growth))

    # ---------- add the prefactor for linear decay/increase in dB scale --------- #
    prefactor_db = 1 + lin_fac * (r - mid)

    # ------------ cut prefactor, where r is larger than the mid point ----------- #
    signal = np.where(r < mid, inv_sigmo_db * prefactor_db, inv_sigmo_db)
    return np.where(signal > down, signal, down)


def inverse_sigmoid_linscale(
    r: float | NDArray[np.float64],
    mid: float,
    down: float,
    up: float,
    growth: float,
    lin_fac: float,
) -> NDArray[np.float64]:
    """Inverse sigmoid in linear scale. Same as inverse_sigmoid_dB but output format is linear scale. INPUT MUST BE dB SCALE

    Following:
        Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kühn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).

    Args:
        r (float | NDArray[np.float64]): radial distance of range gate center
        up (float): upper CNR level before drop [dB]
        down (float):  lower CNR level after drop [dB]
        mid (float): radial distance of the inflection point of the sigmoid -> corresponds to water range [m]
        growth (float): growth rate of the sigmoid function [-]
        lin_fac (float): linear factor to adjust linear decrease before the drop [dB/m]

    Returns:
        NDArray[np.float64]: Sigmoid function (r) in linear scale
    """
    inv_sigmo = inverse_sigmoid(
        r, mid, 0, db2linear(up), growth
    )  # sigmoid falls down to zero, as we assume a perfect signal
    prefactor = db2linear(
        lin_fac * (r - mid)
    )  # remove the 1 from Gramitzky et al. (2024)
    signal = np.where(r < mid, inv_sigmo * prefactor, inv_sigmo)
    return np.where(
        signal > db2linear(down), signal, db2linear(down)
    )  # lower limit of CNR (noise)


def model_cnr_signal_CDF(
    r,
    R_water,
    uppervalue: float,
    lowervalue: float,
    lin_fac: float,
    pulse_object: PulseShape,
):
    """Model the CNR signal using the inverse CDF of a pulse shape. Not really used anymore"""
    cdf = pulse_object.get_inverse_cdf(r - R_water)
    cdf = cdf * (uppervalue - lowervalue) + lowervalue  # scale CNR
    cdf = (1 + lin_fac * (r - R_water)) * cdf
    return cdf


@dataclass
class CNRFitResult:
    """result dataclass from CNR Sigmoid fits"""

    success: bool
    r_water: float
    residuals: float = np.inf
    x: list = None  # parameter values
    params: dict = None
    fit_data: xr.Dataset = None
    fit: plt.Figure = None


# ------------------------------------ xx ------------------------------------ #
# ------------------------------------ xx ------------------------------------ #
# -------------------- Actual Water Range Detection class -------------------- #
# ------------------------------------ xx ------------------------------------ #
# ------------------------------------ xx ------------------------------------ #


class WaterRangeDetection:
    """Base class for the detection of the lidar-water range r_w"""

    def __init__(
        self,
        data_db: xr.Dataset,
        pulse: PulseShape = None,
        verbose: int = 0,
        input_in_db: bool = True,
    ):
        """Initialize Water Range Detection

        Args:
            data_db (xr.Dataset): Dataset of CNR data with dimension "range". Only single timestamp is evaluated here
            pulse (PulseShape, optional): Pulse shape object to describe the pulse shape. Only use for convolution models If None, it is not considered. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 0.
            input_in_db (bool, optional): Whether the input data is in decibel scale. Defaults to True.

        Raises:
            ValueError: _description_
        """
        # ------------------------------ prepare dataset ----------------------------- #
        if not len(data_db.dims) == 1 or "range" not in data_db.dims:
            raise ValueError("data_db must be 1D xarray Dataset with dimension 'range'")

        if not "cnr" in data_db.data_vars:
            raise ValueError(
                f"data_db must contain variable 'cnr' with the CNR data, if name is CNR or similar, try ds.rename({...})"
            )

        if not input_in_db:
            data_db = data_db.copy()
            data_db["cnr"] = linear2db(data_db["cnr"])
            data_db["cnr"].attrs["unit"] = "dB"
        self.data_db = data_db.copy()
        self.pulse = pulse
        self.verbose = verbose

    @staticmethod
    def _linear_signal_decay(
        r: NDArray[np.float64], R_water: float, cnr_slope: float, cnr_offset: float
    ) -> NDArray[np.float64]:
        """Model the radial signal strength of a linear decay with a drop at R_water -- in decibel scale

        Args:
            r (np.ndarray): range array [m]
            R_water (float): lidar-water range [m]
            cnr_slope (float): slope of the linear decay [dB/m]
            cnr_offset (float): offset of the linear decay [dB]

        Returns:
            np.ndarray: modeled radial signal strength
        """
        cnr_signal_undisturbed = (r - R_water) * cnr_slope + cnr_offset
        cnr_signal_water = np.where(r < R_water, cnr_signal_undisturbed, -np.inf)
        return cnr_signal_water

    @staticmethod
    def model_cnr_signal_convolution(
        r: np.ndarray,
        R_water: float,
        cnr_slope: float,
        cnr_offset: float,
        pulse_object: PulseShape,
        cnr_noise: float = -30,
        return_dB: bool = False,
    ):
        """Model the cnr signal convolution of a linear decay with a drop at R_water and a pulse shape, given by a pulse_object --> the returned signal is given in the linear scale!

        Args:
            r (np.ndarray): Range array where convolution is calculated at - must be equidistant [m]
            R_water (float): Range to water surface [m]
            cnr_slope (float): Slope of the linear decay [dB/m]
            cnr_offset (float): Offset of the linear decay [dB]
            pulse_object (PulseShape): Pulse shape object
            cnr_noise (float, optional): Noise level in CNR. Defaults to -28. [dB]

        Returns:
            NDArray[np.float64]: Modeled CNR signal at range r
        """

        # ----------------------------- pad the cnr data ----------------------------- #
        rdiff = np.diff(r)
        dr = np.round(np.min(rdiff), 3)  # range gate resolution
        r_range_gate = (
            np.arange(0, pulse_object.gate_length, dr) + dr
        )  # discretize the range gate
        n_values_pad = len(r_range_gate)
        r_padded = np.concat(
            [r[0] - r_range_gate[::-1], r, r[-1] + r_range_gate]
        )  # padd range array for convolution

        # ------------------------- estimate water cnr curve ------------------------- #
        cnr_signal_water_db = WaterRangeDetection._linear_signal_decay(
            r_padded, R_water, cnr_slope, cnr_offset
        )
        cnr_signal_water = db2linear(
            cnr_signal_water_db
        )  # convert into dB scale, where the linear function becomes exponential

        # ------------------------- calculate pulse weighting ------------------------ #
        r_pulse = np.arange(
            -pulse_object.gate_length / 2, pulse_object.gate_length / 2 + dr, dr
        )
        pulse_weighting = pulse_object.get_weighting(r_pulse)

        # --------------------------- calculate convolution -------------------------- #
        modeled_function = np.convolve(
            cnr_signal_water, pulse_weighting, mode="same"
        ) / np.sum(pulse_weighting)

        # ------------------------------ remove padding ------------------------------ #
        modeled_function = modeled_function[n_values_pad:-n_values_pad]

        #  --------------------------- add noise floor level -------------------------- #
        modeled_function = np.where(
            modeled_function > db2linear(cnr_noise),
            modeled_function,
            db2linear(cnr_noise),
        )

        if return_dB:
            # return in decibel scale
            modeled_function = linear2db(modeled_function)

        return modeled_function

    @staticmethod
    def convolution_fit_error(
        params: tuple, data: xr.DataArray, pulse: PulseShape, use_linear_scale: bool
    ) -> float:
        """Calculate the fit error (sum of sqared errors) for convolution with fixed FWHM of pulse

        Args:
            params (tuple): Parameters for the fit error calculation (R_water, cnr_slope, cnr_offset, cnr_noise, FWHM)
            data (xr.DataArray): input data for fit error calculation, must have variables "range" and "cnr"
            pulse (PulseShape): pulse shape object
            use_linear_scale (bool): whether the input data is in linear scale

        Returns:
            float: Calculated fit error
        """

        R_water, cnr_slope, cnr_offset, cnr_noise = params

        r = data["range"].values
        cnr = data["cnr"].values  # [dB] if not use_linear_scale else [linear scale]

        # get the modeled function in the same scale as input data
        modeled_function = WaterRangeDetection.model_cnr_signal_convolution(
            r,
            R_water,
            cnr_slope,
            cnr_offset,
            pulse,
            cnr_noise=cnr_noise,
            return_dB=not use_linear_scale,
        )

        error = np.sum((modeled_function - cnr) ** 2)
        return error

    @staticmethod
    def convolution_fit_error_pulsevar(
        params: tuple, data: xr.DataArray, pulse: PulseShape, use_linear_scale: bool
    ) -> float:
        """Calculate the fit error betwenen input data and modeled data: model from pulse with variable FWHM

        Args:
            params (tuple): Parameters for the fit error calculation (R_water, cnr_slope, cnr_offset, cnr_noise, FWHM)
            data (xr.DataArray): input data for fit error calculation, must have variables "range" and "cnr"
            pulse (PulseShape): pulse shape object
            use_linear_scale (bool): whether the input data is in linear scale

        Returns:
            float: Calculated fit error
        """
        R_water, cnr_slope, cnr_offset, cnr_noise, FWHM = params
        pulse.FWHM = FWHM

        error = WaterRangeDetection.convolution_fit_error(
            (R_water, cnr_slope, cnr_offset, cnr_noise), data, pulse, use_linear_scale
        )
        return error

    def _convolution_fit_wrapper(self, r, *params):
        """Wrapper function to get modeled CNR signal for convolution fit with fixed pulse FWHM"""
        # get model function in linear scale
        cnr_noise = params[-1]
        modeled_cnr = WaterRangeDetection.model_cnr_signal_convolution(
            r, *params[:-1], self.pulse, cnr_noise=cnr_noise
        )

        if not self.use_linear_scale:
            modeled_cnr = linear2db(modeled_cnr)
        return modeled_cnr

    def _convolution_fit_wrapper_pulsevar(self, r, *params):
        """Wrapper function to get modeled CNR signal for convolution fit with variable pulse FWHM"""
        # get model function in linear scale
        *params, FWHM = params
        self.pulse.FWHM = FWHM  # update pulse FWHM
        return WaterRangeDetection._convolution_fit_wrapper(self, r, *params)

    def _fit_wrapper(self, *args):
        """Fit wrapper for CNR model with improved pulse shape (not used, deprecated)"""
        cnr = model_cnr_signal_CDF(*args, self.pulse)
        return cnr

    def get_water_range_from_cnr(
        self,
        min_cnr: float = -22,  # in dB
        cnr_hard_target: float = 0,  # in dB
        use_bounds: bool = True,
        func: Literal[
            "LinSig",
            "dBSig",
            "Gra24",
            "Rot22",
            "dBConvo",
            "Convo_pulse",
            "Convo",
            "LinConvo_pulse",
        ] = "LinSig",
        fit_method: Literal["curve_fit", "minimize", "LSQ"] = "LSQ",
        dist_guess: float = None,
        high_cnr_bounds=[-23, -3],
        low_cnr_bounds=[-40, -20],
        cnr_noise_cut=-40,
        growth_rate_bounds=[0.0005, 1],
        lin_factor_bounds=[-0.015, 0.03],
        lin_fac_guess: float = 0.01,
        cnr_noise_first_guess: float = -32,
        cnr_noise_bounds: list = [-40, -25],
        _first_guess_roll_window: float = 10,
        _first_guess_scale: Literal["dB", "Lin"] = "Lin",
        show_plot: bool = False,
        return_fit: bool = False,
    ) -> CNRFitResult:
        """Get the lidar-water range from the CNR signal of a pulsed lidar.

        All Inputs/Boundaries are taken in dB scale.

        Should be called like this:
            fit_result = WaterRangeDetection(data_db).get_water_range_from_cnr(...)
            distance = fit_result.r_water


        Follows:
             Meyer, P.J., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kühn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).


        Args:
            min_cnr (float, optional): minimum CNR to be considered as valid signal [dB]. If max(CNR) is below, the distance is NAN. Defaults to -22 dB.
            cnr_hard_target (float, optional): CNR value of hard target [dB]. If max(CNR) is larger, hard target is detected and distance is NAN. Defaults to 0 dB.
            use_bounds (bool, optional): Whether to use bounds for the fit parameters. Defaults to True.
            func (Literal["LinSig","dBSig", "Gra24", "Rot21","Convo", "Convo_pulse",'LinConvo','LinConvo_pulse'], optional): Method to use for the fitting. A full description is provided in the documentation. Defaults to "LinSig".
            fit_method (Literal["curve_fit", "minimize",'LSQ'], optional): Method to fit the least squares. Defaults to "LSQ".
            show_plot (bool, optional): Show plot of fit afterwards. Defaults to False.
            return_fit (bool, optional): Whether to return the fit result object. Defaults to False.
            dist_guess (float, optional): Initial guess for the distance to support fit [m]. If None, not considered. Defaults to None.
            high_cnr_bounds (list, optional): Bounds for high CNR values in fit [dB]. Defaults to [-23, -3].
            low_cnr_bounds (list, optional): Bounds for low CNR values in fit [dB]. Defaults to [-40, -20].
            cnr_noise_cut (int, optional): Threshold to cut off CNR noise in fit [dB]. Only used for LinSig. Should be adjusted to the actual data. Defaults to -40 dB.
            growth_rate_bounds (list, optional): Bounds for the growth rate parameter of the sigmoid. Defaults to [0.0005,1].
            lin_factor_bounds (list, optional): Bounds for the linear factor parameter of the CNR decay before the water [dB/m]. Defaults to [-0.015, 0.03].
            lin_fac_guess (float, optional): Initial guess for the linear factor [dB/m]. Defaults to 0.01.
            cnr_noise_first_guess (float, optional): Initial guess for the CNR noise level [dB]. Defaults to -32.
            cnr_noise_bounds (list, optional): Bounds for the CNR noise level [dB]. Defaults to [-40,-25].
            _first_guess_roll_window(float, optional): Number of values for rolling mean in CNR to find first guess for middle range, where change is the strongest. This parameter should be chosen accordinly to the spatial resoultion of the range gate centers and the probe volume length. In Principle, the value should be L_probevolume/ dr.
            _first_guess_scale:Literal['dB','Lin'] = 'Lin': Whether to use the CNR in dB scale or linear scale for the first guess estimation. Especially for far ranges (low elevation angles) LinSig can lead to a faulty first guess, as the CNR (in linear scale) might drop faster at shorter ranges (due to the atmospheric backscatter). The DB scale however can have strugles, when there is a lot of scatter in the CNR noise -> Should only be used with CNR noise removal parameter set. Defaults to 'Lin'.

        Returns:
            CNRFitResult: Fit Results dataclass with water range and fit success (and plot or fit data if requested)
        """

        data_db = self.data_db.copy()
        verbose = self.verbose

        # ---------------------------- preadjust cnr data ---------------------------- #
        valid = WaterRangeDetection._prepare_cnr_data(
            data_db, min_cnr, cnr_hard_target, verbose
        )
        if not valid:
            if verbose > 1:
                print("Not valid, will return NAN")
            return CNRFitResult(False, np.nan)

        # --------------------------- remove cnr noise data -------------------------- #
        if cnr_noise_cut is not None:
            mask = data_db["cnr"] < cnr_noise_cut
            if mask.any():
                if mask.all():
                    return CNRFitResult(False, np.nan)
                # first range with CNR > cnr_noise
                first_noise_range = mask.idxmax(dim="range").values
                # print(f"Remove data with range > {first_noise_range}")
                data_db["cnr"] = data_db["cnr"].where(
                    data_db["range"] < first_noise_range, other=cnr_noise_cut
                )

        self.cnr_noise = cnr_noise_first_guess

        # -------------- Convert into linear scale for fitting if required ------------- #
        if func in ["LinSig", "LinConvo", "LinConvo_pulse"]:
            self.use_linear_scale = True
            data_lin = data_db.copy()
            data_lin["cnr"] = db2linear(data_lin["cnr"])
            data_lin["cnr"].attrs["unit"] = "- (linear)"
            cnr_hard_target = db2linear(np.array(cnr_hard_target))
            min_cnr = db2linear(min_cnr)
            self.cnr_noise = db2linear(self.cnr_noise)
            data_use = data_lin
        else:
            self.use_linear_scale = False
            data_use = data_db

        # ------------------------------- set fit bounds ------------------------------ #
        distance_bounds = [data_db["range"].min(), data_db["range"].max()]

        general_cnr_bounds = [min_cnr, cnr_hard_target]
        _boundorder = [
            distance_bounds,
            low_cnr_bounds,
            high_cnr_bounds,
            growth_rate_bounds,
        ]

        if func in ["LinSig", "Gra24", "dBSig"]:
            ## adding bounds for linear decay factor
            _boundorder += [lin_factor_bounds]

        # ----------------------- initial guess for parameters ----------------------- #
        high_cnr = np.quantile(data_db["cnr"].values, 0.98)
        low_cnr = np.quantile(data_db["cnr"].values, 0.02)
        middle_cnr = (high_cnr + low_cnr) / 2

        if self.use_linear_scale:
            high_cnr = db2linear(high_cnr)
            low_cnr = db2linear(low_cnr)
            middle_cnr = db2linear(middle_cnr)

        # -------- find initial guess for range, where drop of CNR [linear scale] is steepest -------- #

        if _first_guess_scale == "dB":
            cnr_roll = (
                data_db["cnr"]
                .rolling(range=_first_guess_roll_window, center=True)
                .mean()
            )
        elif _first_guess_scale == "Lin":
            cnr_roll = (
                db2linear(data_db["cnr"])
                .rolling(range=_first_guess_roll_window, center=True)
                .mean()
            )
        else:
            raise ValueError(
                f"_first_guess_scale must be either 'dB' or 'Lin', but got {_first_guess_scale}"
            )
        middle_range = data_db["range"].values[
            cnr_roll.diff(dim="range").argmin(dim="range").values
        ]
        if verbose > 1:
            print(f"first guess for middle range: {middle_range}")

        #  old method: find initial guess where CNR is smaller than the middle cnr
        # last distance, where CNR > middle CNR
        # min_distance = data_use["range"].where(data_use["cnr"] > middle_cnr).max()
        # middle_range = (data["range"].where((data["range"] > min_distance) & (data["cnr"] <= middle_cnr)).min()).values
        # if middle_range != middle_range:  # nan
        #     middle_range = np.nanmax([min_distance, data["range"].mean()])

        # alternative:
        # middle_range = np.nanmax(data_db["range"].where(data_use["cnr"] > min_cnr))

        dist_guess = dist_guess if dist_guess is not None else middle_range
        first_guess = [dist_guess, low_cnr, high_cnr, 0.01]

        if func in ["LinSig", "Gra24", "dBSig"]:
            # add initial guess for linear factor
            first_guess += [lin_fac_guess]
        elif func in ["Convo", "LinConvo", "LinConvo_pulse", "Convo_pulse"]:
            diff_r = np.diff(data_use["range"].values)
            if not np.allclose(diff_r, diff_r[0]):
                raise ValueError(
                    "For convolution fit methods, the range gates must be equidistantly spaced!"
                )

            if self.use_linear_scale:
                # convert back into dB scale, which is used for parametrisation in convo
                general_cnr_bounds = [linear2db(d) for d in general_cnr_bounds]
                middle_cnr = linear2db(middle_cnr)
                high_cnr = linear2db(high_cnr)

            # for convolution, all input parameters are defined in dB and are converted into linear scale within the model function
            _boundorder = [
                distance_bounds,
                lin_factor_bounds,
                general_cnr_bounds,
                cnr_noise_bounds,
            ]
            first_guess = [dist_guess, lin_fac_guess, high_cnr, cnr_noise_first_guess]

            if verbose > 1 and fit_method != "minimize":
                print(
                    "Had to switch to minimize for convolution fit method, due to fitting problems with curve fit and ODR"
                )
            fit_method = "minimize"

            # add additional bound and guess for pulse FWHM if variable
            if func in ["Convo_pulse", "LinConvo_pulse"]:
                _boundorder += [[self.pulse.FWHM * 0.8, self.pulse.FWHM * 1.1]]
                first_guess += [self.pulse.FWHM]

            ## Upsample data for conolution. This is required to increase the resolution of the data. Otherwise, the convolution fit results have a resolution similar to the measurement datas resolution
            if verbose > 2:
                print("Upsampling data for convolution fit")
            data_use_original = data_use.copy()
            data_use = data_use.copy().interp(
                range=np.arange(
                    np.nanmin(data_use["range"].values),
                    np.nanmax(data_use["range"].values),
                    0.5,
                ),
                method="linear",
            )

        lower_bounds = [b[0] for b in _boundorder]
        upper_bounds = [b[1] for b in _boundorder]
        bounds = Bounds(lower_bounds, upper_bounds)

        if verbose > 1:
            print(f"first_guess: {first_guess}")
            if use_bounds:
                print(f"bounds: {bounds}")

        # ---------------------- check first_guess within bounds --------------------- #
        for i, (p, lb, up) in enumerate(zip(first_guess, bounds.lb, bounds.ub)):
            if not (p >= lb and p <= up) and use_bounds:
                if verbose > 1:
                    print(f"First guess {p} not in bounds {lb} - {up}, clipping")
                first_guess[i] = np.clip(first_guess[i], lb, up)

        # -------------------------- perform the actual fit -------------------------- #

        fitfunc = dict(
            dBSig=inverse_sigmoid_dbscale,
            LinSig=inverse_sigmoid_linscale,
            Gra24=inverse_sigmoid_Gra24,
            Rot21=inverse_sigmoid,
            improved_pulse=self._fit_wrapper,
            Convo=self._convolution_fit_wrapper,
            Convo_pulse=self._convolution_fit_wrapper_pulsevar,
            LinConvo=self._convolution_fit_wrapper,
            LinConvo_pulse=self._convolution_fit_wrapper_pulsevar,
        )[func]

        errorfunc = dict(
            Convo=WaterRangeDetection.convolution_fit_error,
            Convo_pulse=WaterRangeDetection.convolution_fit_error_pulsevar,
            LinConvo=WaterRangeDetection.convolution_fit_error,
            LinConvo_pulse=WaterRangeDetection.convolution_fit_error_pulsevar,
        ).get(func, None)

        def _calc_residuals(params):
            """wrapper to calculate the fit residuals for the least squares fitting routine"""
            residuals = (
                fitfunc(data_use["range"].values, *params) - data_use["cnr"].values
            )
            return residuals

        if verbose > 1:
            print(f"Using fit method: {fit_method}")

        if fit_method == "minimize":
            res = minimize(
                errorfunc,
                x0=first_guess,
                args=(data_use, self.pulse, self.use_linear_scale),
                method="Nelder-Mead",
                bounds=bounds if use_bounds else None,
            )
            param = res.x

        elif fit_method == "LSQ":
            res = least_squares(
                _calc_residuals,
                x0=first_guess,
                # args=(data_use, self.pulse, self.use_linear_scale),
                loss="linear",
                bounds=bounds if use_bounds else None,
            )
            param = res.x

        elif fit_method == "curve_fit":
            raise ValueError(
                "curve_fit not supported anymore for convolution fits, due to inconsistent fit results (probably too many parameters?)"
            )
            try:
                print(fit_method)
                param, covs = curve_fit(
                    fitfunc,
                    data_use["range"].values,
                    data_use["cnr"].values,
                    p0=first_guess,
                    bounds=bounds if use_bounds else (-np.inf, np.inf),
                )

            except RuntimeError:

                if verbose > 0:
                    print("Warning: Could not find fit result. Will continue")

                param = [np.nan] * len(first_guess)
            except Exception as e:
                print(e)
                param = [np.nan] * len(first_guess)

        elif fit_method == "ODR":
            raise ValueError(
                "ODR not supported anymore, due to non-consideration of bounds!"
            )
            from scipy.odr import ODR, Model, RealData

            model = Model(lambda param, x: fitfunc(x, *param))
            odr_data = RealData(data_use["range"].values, data_use["cnr"].values)
            odr = ODR(odr_data, model, beta0=first_guess)
            out = odr.run()
            if out.info >= 6:
                if verbose > 0:
                    print("Warning: Could not find fit result. Will continue")

                param = [np.nan] * len(first_guess)
            else:
                param = out.beta
        else:
            raise ValueError(f"Unknown fit method: {fit_method}")

        # --------------------------- get result parameter --------------------------- #
        if func in ["LinSig", "Gra24", "dBSig", "standard"]:
            distance, lower, upper, growth, linearfac = param[:5]
            result_dict = dict(
                distance=distance,
                lower=lower,
                upper=upper,
                growth=growth,
                linearfac=linearfac,
            )
        elif func == "Rot21":
            distance, lower, upper, growth = param[:4]
            result_dict = dict(
                distance=distance, lower=lower, upper=upper, growth=growth
            )
        elif func in ["Convo", "LinConvo"]:
            distance, linearfac, upper, lower = param[:4]
            result_dict = dict(
                distance=distance, linearfac=linearfac, upper=upper, noise=lower
            )
            data_use = data_use_original
        elif func in ["Convo_pulse", "LinConvo_pulse"]:
            distance, linearfac, upper, noise, FWHM = param[:5]
            result_dict = dict(
                distance=distance,
                linearfac=linearfac,
                upper=upper,
                noise=noise,
                FWHM=FWHM,
            )
            data_use = data_use_original

        else:
            raise ValueError("Unknown fitting function")

        # ---------------------- check results if within bounds ---------------------- #

        perform_checks = False
        if perform_checks:
            if upper < min_cnr:
                if verbose > 0:
                    print("CNR too low of min_cnr")
                    print(f"Min CNR: {min_cnr}, param: {param[0]}")
                distance = np.nan

        # print(result_dict)
        if "growth" in result_dict.keys():
            if np.isclose(result_dict["growth"], _boundorder[3][0]) or np.isclose(
                result_dict["growth"], _boundorder[3][1]
            ):
                if verbose > 1:
                    print(
                        "Growth rate close to bounds, probably no Sea hit or fit did not converge properly"
                    )
                    print(
                        f"Growth rate: {result_dict['growth']}, bounds: {_boundorder[3]}"
                    )
                distance = np.nan

        # --------------------------- return function call --------------------------- #
        residuals = np.sum(
            (fitfunc(data_use["range"].values, *param) - data_use["cnr"].values) ** 2
        )

        ret_obj = CNRFitResult(
            True if not np.isnan(distance) else False,
            distance,
            residuals,
            param,
            result_dict,
        )

        if return_fit or show_plot:
            fitdata = fitfunc(data_use["range"].values, *param)
        if show_plot:
            fig = self._plot_distance_retrieval(data_use, ret_obj, fitdata)
            ret_obj.fig = fig
            if isinstance(show_plot, str):
                fig.savefig(
                    f"{show_plot}/fit_{func}_{pd.to_datetime(self.data_db['time'].values).strftime('%Y%m%d_%H%M%S.%f')}.png"
                )
                plt.close(fig)

        if return_fit:
            fitdata = xr.Dataset(
                {
                    "fit_db": (
                        ["range"],
                        fitdata if not self.use_linear_scale else linear2db(fitdata),
                    ),
                    "fit_lin": (
                        ["range"],
                        fitdata if self.use_linear_scale else db2linear(fitdata),
                    ),
                    "data_db": (["range"], data_db["cnr"].values),
                    "data_lin": (["range"], db2linear(data_db["cnr"].values)),
                },
                coords={"range": data_use["range"]},
            )
            ret_obj.fit_data = fitdata
        return ret_obj

    @staticmethod
    def _prepare_cnr_data(
        data: xr.Dataset, min_cnr: float, cnr_hard_target: float, verbose: int = 0
    ) -> bool:
        """Preparation of CNR data for Sigmoid fitting: Check if data is valid for lidar-water range

        Args:
            data (xr.Dataset): CNR dataset with "range" as one dimensions
            min_cnr (float): minimum CNR to be considered as valid signal
            cnr_hard_target (float): maximum CNR to be considered as valid signal
            verbose (int, optional): verbosity level. Defaults to 0.

        Returns:
            bool: Boolean flag that indicates validity of the signal
        """
        # ----------------------------- Perform prechecks ---------------------------- #
        dataset_dimensions = data.squeeze().dims
        if "range" not in dataset_dimensions:
            raise ValueError("Range must be in the dataset dimensions")

        if len(dataset_dimensions) != 1:
            raise ValueError("Strange")

        # ------------------------------ clean cnr data ------------------------------ #
        if data["cnr"].max() < min_cnr:
            if verbose > 1:
                print(
                    f"CNR too low: min CNR:{min_cnr} dB, but data max is {data['cnr'].max().values:.2f}"
                )
            return False
        if data["cnr"].max() > cnr_hard_target:
            if verbose > 1:
                print(
                    f"Hard target found: {data['cnr'].max().values}, {cnr_hard_target}"
                )
            return False

        return True

    @staticmethod
    def _plot_distance_retrieval(
        data: xr.Dataset, ret_object: CNRFitResult, fitdata: xr.Dataset
    ) -> plt.Figure:
        """Plot the distance retrieval of the lidar-water range determination

        Args:
            data (xr.Dataset): original dataset with the cnr data
            ret_object (CNRFitResult): _description_
            fitdata (xr.Dataset): _description_

        Returns:
            plt.Figure: _description_
        """
        fig, ax = plt.subplots()
        data.plot.scatter(x="range", y="cnr", color="r", ax=ax)
        ax.set(xlabel=r"$r_\mathrm{lidar}$ [m]", ylabel="CNR [dB]")
        ax.plot(
            data["range"],
            fitdata,
            linewidth=2,
            color="k",
            # label="Fit with " + ", ".join([f"{p:.3f}" for p in param]),
            label=f"Fit with:"
            + "\n".join([f"{k}: {v:.4f}" for k, v in ret_object.params.items()]),
        )
        ax.axvline(ret_object.r_water)
        ax.legend(loc="lower left")

        # print(f"Estimated distance: {ret_object.r_water:.2f} m")
        ax.legend()
        return fig


@dataclass
class SSCFitResults:
    """Dataclass for SSC fit results.

    Args:
        success : bool
            Indicates if the fitting process was successful.
        x : NDArray[np.float64]
            Array of fitted parameter values.
        result_dict : dict, optional
            Dictionary containing additional fit results and metadata. Default is None.
        fig (optional): Matplotlib figure object containing the fit plot. Default is None.
    """

    success: bool
    x: NDArray[np.float64]
    result_dict: dict = None


class SSC:
    """Base class for the SeaSurfaceCalibration from surrounding lidar scans and derived lidar-water ranges.
    For a detailed description of the SSC see the documentation or:

    Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kühn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).

    Many parts are based on the Sea Surface Leveling by Rott et al. 2022:
    Rott, A., Schneemann, J., Theuer, F., Trujillo Quintero, J. J., & Kühn, M. (2022). Alignment of scanning lidars in offshore wind farms. Wind Energy Science, 7(1), 283–297. https://doi.org/10.5194/wes-7-283-2022


    """

    def __init__(self, ds: xr.Dataset, verbose: int = 1):
        """Initialize the SSC object. This requires the input dataset to have the dimension time (multiple LOS) and range (for lidar-water range determination from cnr signal)
        It wraps the WaterRangeDetection for all time steps in the dataset to obtain the lidar-water ranges.

        Args:
            ds (xr.Dataset): xarray dataset with the dimensions ['range', 'time'] and variables ['cnr', 'azimuth', 'elevation']
            verbose (int, optional): Verbosity level. Defaults to 1.
        """
        self._check_input(ds)
        self.ds = ds
        self.verbose = verbose

    def _check_input(self, ds):
        """check the input of the dataset for required variables and dimensions"""
        required_vars = ["cnr", "azimuth", "elevation"]
        for var in required_vars:
            if var not in ds.variables:
                raise ValueError(
                    f"Variable {var} not found in dataset, following vars are needed {','.join(required_vars)}"
                )
        dimensions = ["range", "time"]
        for dim in dimensions:
            if dim not in ds.dims:
                raise ValueError(
                    f"Dimension {dim} not found in dataset, following dims are needed {','.join(dimensions)}"
                )

    def get_all_water_ranges(
        self,
        n_processes: int = 1,
        pulse: PulseShape = None,
        saveplot: bool | str = False,
        **kwargs,
    ):
        """Wrapper for WaterRangeDetermination objects to get the lidar-water range for all measurements in the dataset.

        Args:
            n_processes (int, optional): number of processes to use for parallel processing. Defaults to 1.
            pulse (PulseShape, optional): pulse shape object. Only used for method LinSig. Defaults to None.
            saveplot (bool | str, optional): whether to save plots or the path to save them. Defaults to False.
            **kwargs: additional arguments to pass to WaterRangeDetection.get_water_range_from_cnr()

        Returns:
            self: SSC object with updated distance_ds attribute containing the lidar-water ranges
        """
        ds = self.ds  # .copy()

        if self.verbose > 1:
            _, ax = plt.subplots()
            ds["cnr"].plot.hist(ax=ax, bins=100)
            ax.set(ylabel="Frequency [-]")

        # initialize empty dataarrays for water range and residuals
        ds["water_range"] = xr.full_like(ds["azimuth"], np.nan)
        ds["residuals"] = xr.full_like(ds["azimuth"], np.nan)

        # get iterator to loop through all time steps
        iterator = (
            tqdm(ds["time"].values, desc="Obtaining lidar-water range")
            if (self.verbose > 0 and n_processes == 1)
            else ds["time"].values
        )

        def perform_detection(ti, return_dict: bool = False):
            """Wrapper function to call the WaterRangeDetection for a single time step"""
            dstime = ds.sel(time=ti)  # .copy()  # .squeeze()

            if len(dstime["cnr"].dropna(dim="range")) < 50:
                if self.verbose > 1:
                    print("Not enough data to fit CNR data")
                ret = CNRFitResult(False, np.nan)
            else:
                ret = WaterRangeDetection(
                    dstime, verbose=self.verbose, pulse=pulse
                ).get_water_range_from_cnr(show_plot=saveplot, **kwargs)
            return ret if return_dict else ret.r_water

        if n_processes == 1:
            params = []
            keys = []
            for ti in iterator:
                ret = perform_detection(ti, return_dict=True)
                ds["water_range"].loc[dict(time=ti)] = ret.r_water
                ds["residuals"].loc[dict(time=ti)] = ret.residuals
                params.append(ret.x)
                keys = ret.params.keys() if ret.success else keys

            # add all fit parameters to the dataset
            for i, k in enumerate(keys):
                data_k = [p[i] if p is not None else np.nan for p in params]
                ds[f"{k}"] = ("time", data_k)
        else:

            from joblib import Parallel, delayed

            # for the parallel processing, we only return the water range
            # results = Parallel(n_jobs=n_processes, verbose=5)(delayed(perform_detection)(ti) for ti in iterator)
            # ds["water_range"].loc[dict(time=ds["time"])] = results

            ## extending for all fit parameters: results is a list of dicts
            results = Parallel(n_jobs=n_processes, verbose=5)(
                delayed(lambda x: perform_detection(x, return_dict=True))(ti)
                for ti in iterator
            )
            ds["water_range"].loc[dict(time=ds["time"])] = [r.r_water for r in results]
            # assigning all values back, might not be optimal implementation but works
            for res, time in zip(results, ds["time"].values):
                if res.success:
                    for k, v in res.params.items():
                        if f"{k}" not in ds.variables:
                            ds[f"{k}"] = (
                                "time",
                                np.full_like(ds["time"], np.nan, dtype=float),
                            )
                        ds[f"{k}"].loc[dict(time=time)] = v

        ds["water_range"].attrs = dict(standard_name="Range to water", units="m")

        # ------------- assign the distance_ds dataset to the SSC object ------------- #
        self.distance_ds = ds  # .copy()

        return self

    @staticmethod
    def rotated_water_elevation(
        los_water_range: NDArray[np.float64],
        los_azimuth: NDArray[np.float64],
        lidar_roll: float,
        lidar_pitch: float,
        lidar_height: float,
        elevation_offset: float = 0,
    ) -> NDArray[np.float64]:
        """Calculate los_elevation with given water range, azimtuh, lidar roll, pitch and height for the simplified and generalized SSL.
        not used in SSC but in generalized SSL?

        Following Gramitzky et al. 2025:
        Gramitzky, K., Jäger, F., Callies, D., Hildebrand, T., Lundquist, J. K., & Pauscher, L. (2025). Alignment of Scanning Lidars in Offshore Campaigns – an Extension of the Sea Surface Levelling Method. Wind and the atmosphere/Wind and turbulence. https://doi.org/10.5194/wes-2025-191

        Args:
            los_water_range (NDArray[np.float]): Lidar-water range in line-of-sight direction [m]
            los_azimuth (NDArray[np.float]): Line-of-sight azimuth angle [deg]
            lidar_roll (float): Lidar roll angle [deg]
            lidar_pitch (float): Lidar pitch angle [deg]
            lidar_height (float): Lidar height above sea surface [m]
            elevation_offset (float, optional): Elevation offset in line-of-sight direction [deg]. Defaults to 0.

        Returns:
            NDArray[np.float]: Calculated los_elevation of line-of-sight [deg
        """
        azimuth_rad = np.deg2rad(los_azimuth)
        # Gramitzky 2025 Eq. (13)
        los_elevation = (
            np.deg2rad(lidar_roll) * np.cos(azimuth_rad)
            - np.deg2rad(lidar_pitch) * np.sin(azimuth_rad)
            - (lidar_height - (los_water_range) ** 2 / (2 * EarthCurvature._R_earth))
            / los_water_range
            + np.deg2rad(elevation_offset)
        )

        # return los_elevation in lidar coordinate system
        return np.rad2deg(los_elevation)

    @staticmethod
    def _misalignment_fit_elevation_error(
        params: tuple,
        measured_data: xr.Dataset,
        fit_method: Literal["LSQ", "lorentz"] = "LSQ",
        consider_earth_curvature: bool = True,
    ) -> float:
        """Calculate the LSQ fit error for the elevation residuals (Gramitzky 2025?)
        --> not used in SSC but in generalized SSL?

        Args:
            params (tuple): Parameteers
            measured_data (xr.Dataset): Dataset containing the variables "water_range", "azimuth", "elevation" and the dimensions "time"
            fit_method (Literal["LSQ", "lorentz"], optional): Fit method to use, can be LSQ or lorentz (less weighting of outliers). Defaults to "LSQ".
            consider_earth_curvature (bool, optional): Consider earth curvature or flat plate of earth (False). Defaults to True.

        Returns:
            float: Sum of squared fit residuals
        """

        lidar_roll, lidar_pitch, lidar_height = params[:3]

        # only if elevation offset is considered
        los_elevation_offset = params[3] if len(params) > 3 else 0

        misalignment_fit = SSC.rotated_water_elevation(
            measured_data["water_range"].values,
            measured_data["azimuth"].values,
            lidar_roll,
            lidar_pitch,
            lidar_height,
            elevation_offset=los_elevation_offset,
            consider_earth_curvature=consider_earth_curvature,
        )

        if fit_method == "lorentz":
            fit_error = np.log(
                1 + 0.5 * (measured_data["elevation"] - misalignment_fit) ** 2
            )

        elif fit_method == "LSQ":
            fit_error = (measured_data["elevation"] - misalignment_fit) ** 2

        else:
            raise ValueError('Unkwown fit method, use "lorentz" or "LSQ"')

        return np.sum(fit_error)

    @staticmethod
    def rotated_water_range(
        los_elevation: NDArray[np.float64],
        los_azimuth: NDArray[np.float64],
        lidar_roll: float,
        lidar_pitch: float,
        lidar_height: float,
        los_elevation_offset: float = 0,
        consider_earth_curvature: bool = True,
    ) -> NDArray[np.float64]:
        """Calculate the lidar-water range for given commanded LOS elevation and azimuth and external misalignment (roll, pitch, lidar_height) and internal misalignment (los_elevation_offset) for the Sea Surface Scans

        Following Rott et al. 2022 and Meyer et al. 2026 (their nomenclature is used here)

        Args:
            los_elevation (NDArray[np.float]): _line-of-sight elevation angles [deg]
            los_azimuth (NDArray[np.float]): _line-of-sight azimuth angles [deg]
            lidar_roll (float): roll of lidar [deg]. Right handed coordinate system. roll around y-axis (lidar north)
            lidar_pitch (float): pitch of lidar [deg]. Right handed coordinate system. pitch around x-axis (lidar east)
            lidar_height (float): height of lidar above sea surface [m]
            los_elevation_offset (float, optional): internal offset of line-of-sight elevation [deg]. Defaults to 0.
            consider_earth_curvature (bool, optional): Consider earth curvature or flat plate of earth (False). Defaults to True.

        Returns:
            NDArray[np.float]: lidar-water range [m]
        """

        actual_los_elevation = los_elevation - los_elevation_offset

        # azimuth is defined clockwise from north, rotation has to be negative
        # everything else is in right handed coordinate system
        # print(actual_los_elevation)
        # print(actual_los_elevation)

        los_rotation = R.from_euler(
            "xz",
            np.stack(
                (np.atleast_1d(actual_los_elevation), np.atleast_1d(-los_azimuth))
            ).T,
            degrees=True,
        ).as_matrix()

        lidar_rot = R.from_euler(
            "xy", [lidar_pitch, lidar_roll], degrees=True
        ).as_matrix()

        # zero azimuth is defined as in y-direction
        los_unit_vector = lidar_rot @ los_rotation @ np.array([0, 1, 0])

        # at which height does the unit vector hit the water
        vert_comp = los_unit_vector[:, 2]

        # remove, if becomes close to zero (for zero elevation in global coordinate system)
        water_range = -lidar_height / vert_comp

        if consider_earth_curvature:

            water_range = np.array(
                [
                    EarthCurvature.get_intercept_with_curvature(
                        lidar_height, np.rad2deg(np.arcsin(vc))
                    )
                    for vc in vert_comp
                ]
            )

        # only positive, if negative this means we never hit the water for positive ranges
        return np.where(water_range > 0, water_range, np.nan)

    @staticmethod
    def _misalignment_fit_range_error(
        params: tuple,
        measured_data: xr.Dataset,
        fit_method: str = "lorentz",
        error_normalized: bool = True,
        consider_earth_curvature: bool = True,
        return_residuals: bool = False,
    ) -> float:
        """Get the sum of squared errors for the range residuals during the SSC fit

        Args:
            params (tuple): parameters (roll, pitch, height, [elevation_offset])
            measured_data (xr.Dataset): Dataset containing the variables "water_range", "azimuth", "elevation" and the dimensions "time"
            fit_method (Literal["LSQ", "lorentz"], optional): Fit method to use, can be LSQ or lorentz (less weighting of outliers). Defaults to "lorentz".
            error_normalized (bool, optional): Normalize errors with range?. Defaults to True.
            consider_earth_curvature (bool, optional): Consider earth curvature or flat plate of earth (False). Defaults to True.
            return_residuals (bool, optional): Whether to return the residuals. Defaults to False.

        Returns:
            float: sum of squared errors
        """

        roll, pitch, height = params[:3]
        los_elevation_offset = params[3] if len(params) > 3 else 0

        misalignment_fit = SSC.rotated_water_range(
            measured_data["elevation"].values,  # - los_elevation_offset,
            measured_data["azimuth"].values,
            roll,
            pitch,
            height,
            los_elevation_offset=los_elevation_offset,
            consider_earth_curvature=consider_earth_curvature,
        )

        residual = misalignment_fit - measured_data["water_range"]

        if error_normalized:
            # normalization to remove dependence on larger ranges, where fluctuation is bigger
            residual = residual / measured_data["water_range"]

        if fit_method == "lorentz":
            # reduced outliers, following Rott et al. 2022
            fit_error = np.log(1 + 0.5 * (residual) ** 2)
        elif fit_method == "LSQ":
            fit_error = residual**2
        else:
            raise ValueError('Unkwown fit method, use "lorentz" or "LSQ"')

        if return_residuals:
            return fit_error
        return np.sum(fit_error)

    @staticmethod
    def get_misalignment(
        data_in: xr.Dataset,
        consider_elevation_offset: bool = True,
        plot: bool = False,
        print_help: bool = True,
        fit_method: Literal["lorentz", "LSQ"] = "lorentz",
        consider_earth_curvature: bool = True,
        error_normalized: bool = True,
        x0=[0, 0, 20],
        ax=None,
        fixed_height: float | None = None,
        reduce_errors: Literal["ranges", "elevation"] = "ranges",
        return_fit: bool = False,
    ) -> SSCFitResults:
        """Get the external misalignment (pitch, roll, height), and internal misalignment (elevation offset) from multi-elevation scans of the water

        Following:
        Meyer, P., Rott, A., Schneemann, J., Gramitzky, K., Pauscher, L., & Kühn, M. (2026). Experimental validation of the Sea-Surface-Calibration for scanning lidar static elevation offset determination (in preparation).


        Args:
            data_in (xr.Dataset): Dataset containing the variables "water_range", "azimuth", "elevation" and the dimensions "time"
            consider_elevation_offset (bool, optional): Consider elevation offset. Defaults to True.
            plot (bool, optional): Flag, whether to plot the results. Defaults to False.
            print_help (bool, optional): Flag, whether to print help information about necessary rotation of the legs. Defaults to True.
            fit_method (Literal["lorentz", "LSQ"], optional): Residual summation function to use. Defaults to 'lorentz'.
            consider_earth_curvature (bool, optional): Consider earth curvature for the sea surface. Defaults to True.
            error_normalized (bool, optional): Normalize errors with range. Defaults to True.
            x0 (list, optional): Initial guess for parameters. Defaults to [0, 0, 20].
            ax (_type_, optional): axes for plotting, if plot=True. Defaults to None.
            fixed_height (float | None, optional): Fixed height value. If this value is set, it will force the fit to use this value. Can be used, if the exact height is known. Otherwise, it is neglected. Defaults to None.
            reduce_errors (Literal['ranges','elevation'], optional): How to calculate the residuals. Defaults to 'ranges'. Possibly, in Gramitzky 2025 et al. only elevation residuals are used (to be confirmed).
            return_fit (bool, optional): Whether to return the fit data. Defaults to False.


        Returns:
            SSCFitResults: Results of the misalignment fit. Contains the attributes sucess (bool), x (NDArray[np.float64]) with the fitted parameters and result_dict (dict) with the named parameters.
        """

        bounds = [[-5, 5], [-5, 5], [0, 100]]

        # remove nan values from data
        data = data_in.dropna(
            dim="time", subset=["water_range", "azimuth", "elevation"]
        )
        # if fixed_height is defined, the fit will be forced to this value
        if fixed_height is not None:
            x0[2] = fixed_height
            bounds[2] = [fixed_height - 0.0001, fixed_height + 0.0001]

        if consider_elevation_offset:
            ## check if data allows for estimation of elevation error:
            if data["elevation"].max() - data["elevation"].min() < 0.5:
                print(
                    "Elevation error is not possible, because elevation variation is too small"
                )
                SSCFitResults(False, [np.nan], {})
            x0 = x0 + [0.01]
            bounds = bounds + [[-1, 1]]

        # get proper function to minimize
        func_use = dict(
            ranges=lambda x: SSC._misalignment_fit_range_error(
                x,
                data,
                fit_method=fit_method,
                consider_earth_curvature=consider_earth_curvature,
                error_normalized=error_normalized,
            ),
            elevation=lambda x: SSC._misalignment_fit_elevation_error(x, data),
        )[reduce_errors]

        res = minimize(
            func_use,
            x0=x0,
            bounds=bounds,
            method="Nelder-Mead",
        )
        if not res.success:
            print("Found a problem, see")
            print(res)
            return SSCFitResults(False, [np.nan], {})

        result = dict(
            roll=res.x[0],
            pitch=res.x[1],
            height=res.x[2],
            ele_offset=res.x[3] if len(res.x) > 3 else 0,
        )
        fit_results_obj = SSCFitResults(True, res.x, result)

        if print_help:
            SSC.interprete_results(result)

        if plot:
            if ax is None:
                fig, ax = plt.subplots()

            if reduce_errors == "ranges":
                data.plot.scatter(
                    x="azimuth",
                    y="water_range",
                    hue="elevation",
                    linewidth=0,
                    edgecolors="face",
                    marker=".",
                    # label = 'Measured data',
                    cbar_kwargs=dict(label=r"Elevation $\varphi$ [deg]"),
                )
                disti = SSC.rotated_water_range(
                    data["elevation"], data["azimuth"], *res.x
                )
                ax.scatter(
                    data["azimuth"],
                    disti,
                    c="k",
                    s=1,
                    label="Fit results:\n"
                    rf"$pitch$={result['pitch']:.3f}°, $roll$={result['roll']:.3f}°"
                    "\n"
                    rf"$h$={result['height']:.2f}m, $\varphi_0$={result['ele_offset']:.3f}°",
                )

            elif reduce_errors == "elevation":
                data.plot.scatter(
                    x="water_range",
                    y="elevation",
                    hue="azimuth",
                    linewidth=0,
                    edgecolors="face",
                    marker=".",
                    label="Measured data",
                    cbar_kwargs=dict(label=r"Lidar azimuth $\theta$ [deg]"),
                )

                model_elevations = SSC.rotated_water_elevation(
                    data["water_range"], data["azimuth"], *res.x
                )
                ax.scatter(
                    data["water_range"],
                    model_elevations,
                    c="k",
                    s=1,
                    label="SSC Fit results:\n"
                    rf"$pitch$={result['pitch']:.3f}°, $roll$={result['roll']:.3f}°"
                    "\n"
                    rf"$h$={result['height']:.2f}m, $\varphi_0$={result['ele_offset']:.3f}°",
                )

            ax.legend(loc="upper right")
            ax.grid(alpha=0.3, ls="--")
            ax.set(
                title=f"{pd.to_datetime(data['time'].values[0]).strftime('%Y-%m-%d %H:%M:%S')}"  #: Elevation {data["elevation"].mean().values:.2f}
            )
            fit_results_obj.fig = plt.gcf()
            fit_results_obj.ax = ax

        # currently only works with range fit, but could be easily extended to elevation fit as well
        if return_fit:
            fit_data = data.copy()
            fit_data["fit_water_range"] = (
                ["time"],
                SSC.rotated_water_range(data["elevation"], data["azimuth"], *res.x),
            )
            fit_data["residuals"] = SSC._misalignment_fit_range_error(
                res.x,
                data,
                fit_method=fit_method,
                consider_earth_curvature=consider_earth_curvature,
                error_normalized=error_normalized,
                return_residuals=True,
            )
            fit_results_obj.fit_data = fit_data

        return fit_results_obj

    @staticmethod
    def interprete_results(
        result,
        gewinde_steigung: float = 1.75 / 1000,
        distance_feet: float = 1,
    ) -> None:
        """Interprete the fit results so you can adjust your legs accordingly. Prints the number of rotations for each leg.

        Args:
            result (dict): Dictionary containing the fit results with keys "roll", "pitch"
            gewinde_steigung (float, optional): thread pitch of the lidar legs [m/rotation]. Typically 1.75 mm per rotation. Defaults to 1.75/1000 m/rotation.
            distance_feet (float, optional): distance between the legs of the lidar that contain the screws for lifting [m]. Defaults to 1,.
        """

        left_leg_rotations = (
            distance_feet * np.sin(np.deg2rad(result["roll"])) / gewinde_steigung
        )
        front_leg_rotations = (
            distance_feet * np.sin(np.deg2rad(result["pitch"])) / gewinde_steigung
        )
        print(f'Pitch: {result["pitch"]:.3f}°')
        print(f'Roll: {result["roll"]:.3f}°')
        print(
            f'This means: \n \t- the lidar west legs must go {["up","down"][int(left_leg_rotations>0)]} by {np.abs(left_leg_rotations):.2f} rotations'
            f'\n \t- the lidar north legs must go {["down","up"][int(front_leg_rotations>0)]} by {np.abs(front_leg_rotations):.2f} rotations'
        )


class EarthCurvature:
    _R_earth = 6_371_000  # mean earth radius [m]

    @staticmethod
    def get_height(distance: float) -> float:
        """Get the height difference due to earth curvature for a given distance

        Following
        Gramitzky, K., Jäger, F., Callies, D., Hildebrand, T., Lundquist, J. K., & Pauscher, L. (2025). Alignment of Scanning Lidars in Offshore Campaigns – an Extension of the Sea Surface Levelling Method. Wind and the atmosphere/Wind and turbulence. https://doi.org/10.5194/wes-2025-191

        Args:
            distance (float): distance from the observer [m]
        Returns:
            float: height difference due to earth curvature [m]
        """
        heightdifference = -(distance**2) / (2 * EarthCurvature._R_earth)
        return heightdifference

    @staticmethod
    def get_intercept_with_curvature(
        height: float, elevation: float, max_distance: float = 10_000
    ) -> float:
        """Get interception point between a line (beam of lidar) and a circle (earth curvature)
        This is required if you want to find out, where your line-of-sight would hit the water.

        Args:
            height (float): height of the observer (lidar lense) in m
            elevation (float): elevation angle of the view (line-of-sight), elevation is defined positive up. Must be large enough, that the water can be seen at this elevation angle at all.

        Returns:
            float: X-value where line of sight and circle have their first crossing (positive x only)
        """
        elevation_rad = np.deg2rad(elevation)

        def lineofsight(x):
            r_los = x / np.cos(elevation_rad)
            y = r_los * np.sin(elevation_rad) + height
            return y

        def circle(x):
            # func for circle: (x-x0)^2 + (y-y0)^2 = r^2
            y = np.sqrt(EarthCurvature._R_earth**2 - x**2) - EarthCurvature._R_earth
            return y

        def difference(x):
            return circle(x) - lineofsight(x)

        try:
            import scipy

            # maximum range of 10_000 --> only if lidar is VERY high, this should be changed
            res = brentq(difference, 10, max_distance, xtol=0.001)

            # res = scipy.optimize.fsolve(difference, 5_000, xtol = 1e-3)[0]
            return res  # keep only first intercept
        except ValueError as E:
            # print(E)
            print(
                f"No solution found, maybe you will never touch the water at this elevation {elevation:.3f}°"
            )
            return np.nan


if __name__ == "__main__":
    print(SSC.rotated_water_range(-0.5, 0, 0, 0, 20, 0, True))
    distance = np.arange(0, 20_000, 100)
    ele = -0.5
    h = 20
    height = EarthCurvature.get_height(distance)
    # fig, ax = plt.subplots()
    # ax.plot(distance, height)
    # ax.set(xlabel="Distance [m]", ylabel="Height due to Earth Curvature [m]")

    # fig, ax = plt.subplots()
    # ax.plot(distance, np.rad2deg(np.arctan(height / distance)))
    # ax.set(xlabel="Distance [m]", ylabel="Horizon angle due to Earth Curvature [deg]")

    # fig, ax = plt.subplots()
    # EarthCurvature.get_intercept_with_curvature(h, ele)
    # ax.plot(distance, height)
    # ax.plot(distance*np.cos(np.deg2rad(ele)), distance*np.sin(np.deg2rad(ele)) + h , label = 'line-of-sight')
    # ax.scatter(0,h, label = 'lidar position', marker = 's')
    # dist = EarthCurvature.get_intercept_with_curvature(h, ele)
    # ax.axvline(dist, label  =f'Intercect: {dist:.1f}m')
    # ax.legend()

    fig, [ax, axd] = plt.subplots(nrows=2, sharex=True)
    eles = np.arange(-1, -0.1, 0.05)
    distances = []
    dist_no_curv = []
    for ele in eles:
        distances.append(
            EarthCurvature.get_intercept_with_curvature(h, ele)
            / np.cos(np.deg2rad(ele))
        )
        dist_no_curv.append(-h / np.sin(np.deg2rad(ele)))

    ax.plot(eles, distances, label="$d$ (Curvature)")
    ax.plot(eles, dist_no_curv, label="$d$ (No Curvature)")
    ax.set(ylabel="Range [m]")
    axd.plot(
        eles,
        np.array(distances) - np.array(dist_no_curv),
        label="$d$ (Curv) - $d$ (no Curv)",
    )
    axd.set(yscale="log", xlabel="Elevation of LoS [deg]", ylabel="Error in Range [m]")
    axd.legend()
    axd.grid(alpha=0.3, ls="--")
    ax.grid(alpha=0.3, ls="--")
    # print(dist)


# %%
# ------------------------------------ xx ------------------------------------ #
# -------------------- Legacy scripts from Rott et al 2022 ------------------- #
# ------------------------------------ xx ------------------------------------ #
# following: Rott, A., Schneemann, J., Theuer, F., & Stone, S. (2021). Data supplement for “Alignment of scanning lidars in offshore wind farms”—Wind Energy Science Journal (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/ZENODO.5654866


def distance_to_water(
    data,
    min_cnr,
    show_plot=0,
    high_cnr_ub=0,
    high_cnr_lb=-25,
    low_cnr_ub=-15,
    low_cnr_lb=-30,
    distance_ub=3000,  # 550,
    distance_lb=200,  # 400,
    growth_ub=1,
    growth_lb=0,
):
    azis = data.azi.unique()
    distances = np.array([])
    for azi in azis:
        data_act = data[data.azi == azi]
        if data_act.cnr.max() < min_cnr:
            distance = np.nan
        else:
            fit_function = (
                lambda up, down, mid, growth: (up - down)
                / (1 + np.exp((data_act.range - mid) * growth))
                + down
            )
            cost_function = lambda param: np.sum(
                (data_act.cnr - fit_function(param[0], param[1], param[2], param[3]))
                ** 2
            )
            bounds = Bounds(
                [high_cnr_lb, low_cnr_lb, distance_lb, growth_lb],
                [high_cnr_ub, low_cnr_ub, distance_ub, growth_ub],
            )

            # initial guess for parameters of the inverse sigmoid function
            high_cnr = data_act.cnr.max()
            low_cnr = data_act.cnr.min()
            middle_cnr = (high_cnr + low_cnr) / 2
            min_distance = data_act.range[data_act.cnr > middle_cnr].min()
            middle_range = data_act.range[
                (data_act.range > min_distance) & (data_act.cnr <= middle_cnr)
            ].min()

            res = minimize(
                cost_function, [high_cnr, low_cnr, middle_range, 0.1], bounds=bounds
            )
            #             print(res)
            if res.x[0] < min_cnr - 3:
                distance = np.nan
            else:
                distance = res.x[2]
            if show_plot:
                ax = data_act.plot(
                    "range",
                    "cnr",
                    grid=True,
                    legend=False,
                    figsize=(12 / 2.54, 9 / 2.54),
                    linewidth=2,
                    style=".",
                    color="r",
                )
                ax.set_ylabel("CNR")
                fig = ax.get_figure()
                fig.tight_layout()
                ax.set_xlabel(r"$r_\mathrm{lidar}$ [m]")
                ax.set_ylabel("CNR [dB]")
                ax.plot(
                    data_act.range,
                    fit_function(res.x[0], res.x[1], res.x[2], res.x[3]),
                    linewidth=2,
                    color="k",
                )
                ax.vlines(
                    res.x[2], data_act["cnr"].min() - 1, data_act["cnr"].max() + 1
                )
                ax.autoscale(enable=True, axis="both", tight=True)
                plt.show()
                print(
                    f"high cnr: {res.x[0]} dB, low_cnr: {res.x[1]} dB, growth_rate: {res.x[3]}"
                )
                print(f"Azimut Angle: {azi}° and estimated distance: {distance} m")
        distances = np.append(distances, distance)
    return azis[~np.isnan(distances)], distances[~np.isnan(distances)]


def fit_function(pitch, roll, height, ele, yaw, s_x=-0.15, s_y=0.15):
    return (
        height
        + s_x * np.sin(yaw / 180 * np.pi) * np.sin(pitch / 180 * np.pi)
        + s_x
        * np.sin(roll / 180 * np.pi)
        * np.cos(yaw / 180 * np.pi)
        * np.cos(pitch / 180 * np.pi)
        + s_y
        * np.sin(yaw / 180 * np.pi)
        * np.sin(roll / 180 * np.pi)
        * np.cos(pitch / 180 * np.pi)
        - s_y * np.sin(pitch / 180 * np.pi) * np.cos(yaw / 180 * np.pi)
    ) / (
        np.cos(ele / 180 * np.pi)
        * (
            np.cos(yaw / 180 * np.pi) * np.sin(pitch / 180 * np.pi)
            - np.cos(pitch / 180 * np.pi)
            * np.sin(roll / 180 * np.pi)
            * np.sin(yaw / 180 * np.pi)
        )
        - np.cos(pitch / 180 * np.pi)
        * np.cos(roll / 180 * np.pi)
        * np.sin(ele / 180 * np.pi)
    )


def lidar_alignment(data, minimum_cnr, approx_height_above_nn, ele):
    Azi, Distance = distance_to_water(data, minimum_cnr)
    start_time = data.index.min()
    # if all(data.ele > -3.01) and all(data.ele < -2.99) and len(Azi) > 20:
    #     ele = -3
    # else:
    #     return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    x0 = [0, 0, approx_height_above_nn]
    #     method = 'least-squares'
    method = "lorentz"
    if method == "lorentz":
        Cost = lambda x: np.sum(
            np.log(
                1
                + 0.5
                * (
                    Distance
                    - fit_function(pitch=x[0], roll=x[1], height=x[2], ele=ele, yaw=Azi)
                )
                ** 2
            )
        )
    elif method == "least-squares":
        Cost = lambda x: np.sum(
            (
                Distance
                - fit_function(pitch=x[0], roll=x[1], height=x[2], ele=ele, yaw=Azi)
            )
            ** 2
        )
    res = minimize(
        Cost, x0, method="nelder-mead", options={"xatol": 1e-8, "disp": False}
    )
    p, r, h = res.x
    return p, r, h, Azi, Distance, start_time
