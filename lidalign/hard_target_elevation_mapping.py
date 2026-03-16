# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.transform import Rotation as R
from tqdm import trange
from typing import Literal


def cosine_curve(azi:float|np.ndarray, phase_offset:float, amplitude:float, offset:float) -> float|np.ndarray:
    """Cosine curve

    Args:
        azi (float | np.ndarray): Azimuth angle [degree]
        phase_offset (float): Phase Offset of cosine curve [degree]
        amplitude (float): Amplitude of cosine curve
        offset (float): Y-offset of cosine curve

    Returns:
        float|np.ndarray: Y values of cosine curve
    """
    
    
    return np.cos(np.deg2rad(azi + phase_offset)) * amplitude + offset


def uncertain_df(
    mean_values: np.ndarray,
    uncertainty_params: np.ndarray,
    uncertainty_distribution: str = "normal",
) -> pd.DataFrame :
    
    """Define a Uncertainty dataframe, that contains the required statistics

    Args:
        mean_values (np.ndarray): mean values of a variable
        uncertainty_params (np.ndarray): uncertainty parameters, here often standard uncertainty of a variable
        uncertainty_distribution (str, optional): How is the uncertainty distributed?. Defaults to "normal"/gaussian.

    Returns:
        pd.DataFrame: Dataframe containing the Uncertainty informations
    """
    df = pd.DataFrame(columns=["mean", "std", "form"])
    df.loc[:, "mean"] = mean_values
    df.loc[:, "std"] = uncertainty_params
    df.loc[:, "form"] = uncertainty_distribution
    return df


class MonteCarloFunc:
    def __init__(self, parameters: list[pd.DataFrame], n: int = 1_000):
        """Initalize MC Object

        Args:
            parameters (list[pd.DataFrame]): Parameters for the uncertainty propagation
            n (int, optional): Number of iterations. Defaults to 1_000.
        """
        self.parameters = parameters
        self.n = n

    def apply_func(self, func):
        """Apply monte carlo iterations for a given function with the uncertainty dataframes

        Args:
            func: Function to perform the iterations for

        Returns:
            tuple of results, means and standard deviations of the iterations
        """
        all_vals = []
        for n in trange(self.n, desc="Performing Monte Carlo Iterations"):
            fit_vals = []
            for i, param in enumerate(self.parameters):
                # if param["form"].iloc[0] == "normal":
                fit_vals.append(np.random.normal(param["mean"], param["std"]))

            retvals = func(*fit_vals)

            all_vals.append(retvals)

        mean = np.mean(all_vals, axis=0)
        std = np.std(all_vals, axis=0)

        return all_vals, mean, std


class HardTargetMappingElevation:
    def __init__(
        self,
        azimuth: np.ndarray,
        delta_elevation: np.ndarray,
        unc_azimuth: np.ndarray|None = None,
        unc_elevation: np.ndarray|None = None,
    ):
        """Initialize the Elevation hard target mapping. Needs a dataset of measured elevation and azimuths from the lidar and the thedilite (as reference).


        Args:
            azimuth (np.array): azimuth values of the hard targets
            delta_elevation (np.array): elevation delta (lidar-reference) of the hard targest
            unc_azimuth (np.array, optional): uncertainty in the azimuth readings. Has to have the size of the azimuth readings. If none, no uncertainty is considered. Defaults to None.
            unc_elevation (np.array, optional): uncertainty in the azimuth readings. Has to have the size of the azimuth readings. If none, no uncertainty is considered. Defaults to None.
        """
        self.azimuth = azimuth
        self.delta_elevation = delta_elevation
        self.uncertainties_considered = False
        self.unc_azimuth = unc_azimuth
        self.unc_elevation = unc_elevation
        if unc_azimuth is not None and unc_elevation is not None:
            self.uncertainties_considered = True
            self._prepare_uncertainty_df()
            print(self.azi_uncertain_df)
            print(self.ele_uncertain_df)

    def _prepare_uncertainty_df(self):

        self.azi_uncertain_df = uncertain_df(self.azimuth, self.unc_azimuth, "normal")
        self.ele_uncertain_df = uncertain_df(self.delta_elevation, self.unc_elevation, "normal")

    @staticmethod
    def pitchroll_fit_func(azimuth, delta_elevation):
        params, coeffs = scipy.optimize.curve_fit(
            HardTargetMappingElevation._pitch_roll_func,
            azimuth,
            delta_elevation,
            bounds=((-1, -1, -1), (1, 1, 1)),
            p0=(0.1, 0.1, 0.1),
        )
        params[0] = params[0]
        return params

    @staticmethod
    def cosine_fit_func(azimuth, delta_elevation):

        params, coeffs = scipy.optimize.curve_fit(
            HardTargetMappingElevation._cosine_curve,
            azimuth,
            delta_elevation,
            bounds=((-180, 0, -np.inf), (360 + 180, np.inf, np.inf)),
            p0=(0, 0, 0.1),
        )
        params[0] = params[0] % 360  # ensure phase is in [0,360]
        return params

    def TaitBryan_rotation(roll, pitch, yaw, azimuth, elevation):
        LOS_matrix_measured = R.from_euler(
            "xz",
            np.column_stack([elevation, -azimuth]),
            degrees=True,
        ).as_matrix()

        Lidar_matrix = R.from_euler("ZYX", [yaw, roll, pitch], degrees=True).as_matrix()
        LOS_vector = Lidar_matrix @ LOS_matrix_measured @ np.array([0, 1, 0])
        return LOS_vector

    @staticmethod
    def fit_TaitBryanAngles(azimuth_reference, elevation_reference, azimuth_measured, elevation_measured):
        def errorfunc(param):
            roll, pitch, yaw, elevation_offset = param
            # vectors = R.from_euler('xz', [-elevation, -azimuth], degrees = True).as_matrix()
            # matrix= R.from_euler('xyz',[pitch, roll, yaw], degrees = True).as_matrix()
            # direction_vector = matrix@vectors@np.array([1,0,0])
            LOS_reference_vector = HardTargetMappingElevation.TaitBryan_rotation(
                0, 0, 0, azimuth_reference, elevation_reference
            )

            LOS_vector_measured = HardTargetMappingElevation.TaitBryan_rotation(
                roll,
                pitch,
                yaw,
                azimuth_measured,
                elevation_measured + elevation_offset,
            )

            # measured_azimuth = (np.rad2deg(np.arctan2(vector[1], vector[0])) - 90) % 360
            # measured_elevation = np.rad2deg(np.arcsin(vector[2]))

            error = np.sum(((LOS_reference_vector - LOS_vector_measured) ** 2))

            return error

        bounds = [[-90, 90], [-90, 90], [-360, 360], [-1, 1]]
        res = scipy.optimize.minimize(
            errorfunc,
            bounds=bounds,
            x0=[0.0, 0.00, 40, 0.0],
            method="Nelder-Mead",
        )
        return res

    def fit(self, n_mc=1000, typ:Literal['cosine','pitchroll']="pitchroll"):
        """Fit the data to the cosine curve."""
        self.fit_typ = typ
        if typ == "cosine":
            results = self.cosine_fit_func(self.azimuth, self.delta_elevation)
            self.params = results

            if self.uncertainties_considered:
                results, mean, std = MonteCarloFunc([self.azi_uncertain_df, self.ele_uncertain_df], n=n_mc).apply_func(
                    HardTargetMappingElevation.cosine_fit_func
                )
                self.mc_results = results, mean, std

        else:
            print(typ)
            results = self.pitchroll_fit_func(self.azimuth, self.delta_elevation)
            self.params = results

            if self.uncertainties_considered:
                results, mean, std = MonteCarloFunc([self.azi_uncertain_df, self.ele_uncertain_df], n=n_mc).apply_func(
                    HardTargetMappingElevation.pitchroll_fit_func
                )
                self.mc_results = results, mean, std

        return self

    def plot(self, ax: plt.Axes|None = None, show_offset=False, **kwargs) -> plt.Axes:
        """Plot the data and the fit on a matplotlib plot.

        Args:
            ax (plt.Axes, optional): Axes to plot onto, if None a new one will be created. Defaults to None.


        Returns:
            plt.Axes: axes with plot
        """
        if not "params" in self.__dict__.keys():
            raise ValueError("Fit the model before plotting, using .fit()")

        if ax is None:
            fig, ax = plt.subplots()

        azi = np.arange(0, 360, 1)

        labels = []
        legend_names = []
        if self.uncertainties_considered:
            eb = ax.errorbar(
                self.azimuth,
                self.delta_elevation,
                yerr=self.unc_elevation,
                xerr=self.unc_azimuth,
                label="Measurements",  # $\mu\pm \sigma$",
                capsize=2,
                c="tab:orange",
                linestyle="",
                marker = '.'
            )

            values = []
            for vals in self.mc_results[0]:
                if self.fit_typ == "cosine":
                    values.append(cosine_curve(azi, *vals))
                else:
                    values.append(HardTargetMappingElevation._pitch_roll_func(azi, *vals))

            print(np.array(values).shape)
            std = np.array(values).std(axis=0)
            mean = np.array(values).mean(axis=0)

            stdvals = self.mc_results[2]

            
            shade = ax.fill_between(
                azi,
                mean - std,
                mean + std,
                alpha=0.2,
                # label=label,
            )

        else:
            eb = ax.scatter(self.azimuth, self.delta_elevation, label="Measurements", c="k")

        labels.append(eb)
        legend_names.append("Measurements")

        params = self.params
        if self.fit_typ == "cosine":
            (line,) = ax.plot(
                azi,
                cosine_curve(azi, *params),
                label=rf"Curve Fit with $Phase$ = {params[0]:.1f}$\degree$, $Amp$ = {params[1]:.3f}$\degree$, $Offs$ = {params[2]:.3f}$\degree$",
                **kwargs,
            )
        else:
            
            label = (
                    rf"$pitch$= {params[0]:.3f}$\degree$"
                    + "\n"
                    + rf"$roll$ = {params[1]:.3f}$\degree$"
                    + "\n"
                    + rf"$\Delta\varphi$ = {params[2]:.3f}$\degree$"
                )
            
            if self.uncertainties_considered:
                label = (
                    rf"$pitch$= {params[0]:.3f}$\degree \pm {stdvals[0]:.3f}\degree$,"
                    + "\n"
                    + rf"$roll$ = {params[1]:.3f}$\degree \pm {stdvals[1]:.3f}\degree$"
                    + "\n"
                    + rf"$\Delta\varphi$ = {params[2]:.3f}$\degree \pm {stdvals[2]:.3f}\degree$"
                )
            (line,) = ax.plot(
                azi,
                HardTargetMappingElevation._pitch_roll_func(azi, *params),
                label=label,
                **kwargs,
            )

        if self.uncertainties_considered:
            labels.append((shade, line))
        else:
            labels.append(line)
        legend_names.append(label)
        ax.set(
            xlabel=r"$\Theta_{\rm reference}$ [deg]",
            ylabel=r"$\varphi_e$ = ($\varphi_{\rm lidar}-\varphi_{\rm reference}$) [deg]",
            title=f"Hard Target Elevation Mapping",
        )
        if show_offset:
            off = ax.axhline(
                params[2],
                ls="--",
                c="tab:red",
                alpha=0.5,
                label=rf"Static elevation offset",  # $\Delta \varphi$= {params[2]:.3f}$\degree$",
            )
            labels.append(off)
            legend_names.append(rf"Static elevation offset")
            ax.axhline(0, ls="--", c="k", alpha=0.5)
        ax.grid(alpha=0.3, ls="--")
        ax.legend(labels, legend_names, loc="lower left", bbox_to_anchor=(0, 1.1))
        plt.tight_layout()
        return ax

    @staticmethod
    def _pitch_roll_func(azi, pitch, roll, offset):
        azi = np.atleast_1d(azi)

        # has to be negative for positive with right handed system
        Rrp = R.from_euler(
            "xy",
            np.column_stack([np.atleast_1d(-pitch), np.atleast_1d(-roll)]),
            degrees=True,
        ).as_matrix()

        R_los = R.from_euler("z", -azi[:, None], degrees=True).as_matrix()

        # print(np.dot(R_los, np.array([0, 1, 0])))

        pos = Rrp @ R_los @ np.array([0, 1, 0])
        ele = np.rad2deg(np.arcsin(pos[:, 2]))
        # azi_out = np.arctan2(pos[0], pos[1])
        return ele + offset

    @staticmethod
    def _cosine_curve(azi, phase_offset, amplitude, offset):
        return np.cos(np.deg2rad(azi + phase_offset)) * amplitude + offset


if __name__ == "__main__":
    print(HardTargetMappingElevation._pitch_roll_func(np.array([50, 150, 250]), 0, 0, 0))

    ## low variation due to high azimuth distribution
    # df = pd.DataFrame(dict(HA=[0, 90, 180], Delta_Elevation=[-0.1, -0.15, -0.2]))
    df = pd.DataFrame(dict(HA=[90, 180, 270], Delta_Elevation=[0.05, 0, -0.05]))

    ## high variation due to small azimuth distribution
    # df = pd.DataFrame(dict(HA=[90, 120, 140], Delta_Elevation=[-0.1, -0.15, -0.2]))

    df["Unc_azi"] = 0.1
    df["Unc_ele"] = 0.02

    HTE = HardTargetMappingElevation(
        df["HA"].values,
        df["Delta_Elevation"].values,
        df["Unc_azi"].values,
        df["Unc_ele"].values,
    )

    # HTE.fit()
    # ax = HTE.plot()

    HTE = HTE.fit(typ="pitchroll", n_mc=100)
    HTE.plot(ls="--")
