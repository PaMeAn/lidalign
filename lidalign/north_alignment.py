# %%
"""The scripts here are based on the scripts by Rott et al 2022:
Rott, A., Schneemann, J., Theuer, F., Trujillo Quintero, J. J., & Kühn, M. (2022). Alignment of scanning lidars in offshore wind farms. Wind Energy Science, 7(1), 283–297. https://doi.org/10.5194/wes-7-283-2022

They are also available at:
Andreas Rott, Jörge Schneemann, & Frauke Theuer. (2021). AndreasRott/Alignment_of_scanning_lidars_in_offshore_wind_farms: Version1.0 (Release1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5654919
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from scipy.optimize import curve_fit
from typing import Literal


# %%
def gaussian(x, A, mu, sigma):
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


class Northalignment:
    """Class for the north alignment of a scanning lidar based on surrounding hard target measurements."""

    def __init__(self, HardTargets_coordinates: pd.DataFrame):
        """Initialize Northalignment object

        Args:
            HardTargets_coordinates (pd.DataFrame): Coordinates of the surrounding hard targets [in meters/UTM], must contain columns x and y
        """
        self.HardTargets = HardTargets_coordinates

    def _prepare_data(
        self, ds: xr.Dataset, CNR_Hardtarget: float, method_use: str = "V2"
    ) -> pd.DataFrame:
        """internal function to prepare the CNR data

        Args:
            ds (xr.Dataset): xarray dataset with the columns cnr [db], azimuth [°] and range [m]
            CNR_Hardtarget (float): CNR threshold to consider a measurement as a hard target measurement, in dB
            method_use (str, optional): Method that is used, V1 is Rott et al. 2022, V2 is the updated version. Defaults to "V2".


        Returns:
            pd.DataFrame: Prepared lidar data
        """

        if method_use == "V1":
            """keep all measurements above the CNR threshold, original method by Rott"""
            hard_target_criterium = ds["cnr"] > CNR_Hardtarget
            lidar_data_df = (
                ds.where(hard_target_criterium)[["cnr", "azimuth", "range"]]
                .to_dataframe()
                .reset_index()
                .dropna()
            )

        elif method_use == "V2":
            """only keep a single range (obtained from a guassian fit of CNR in linear scale) for each LOS"""
            hard_target_criterium = ds["cnr"].max(dim="range") > CNR_Hardtarget
            lidar_data_ds = ds.where(hard_target_criterium)[
                ["cnr", "azimuth", "range"]
            ].dropna(dim="time", how="all")

            ## lets fit a gaussian to the cnr profiles
            lidar_data_ds["cnr_lin"] = 10 ** (lidar_data_ds["cnr"] / 10)

            results = []
            for time in lidar_data_ds.time:

                lidar_data_dst = lidar_data_ds.sel(time=time)

                # limit to area around actual maximum
                r_max = lidar_data_dst["cnr_lin"].idxmax().values
                max_data = lidar_data_dst.sel(range=slice(r_max - 200, r_max + 200))

                y_data = max_data["cnr_lin"].values
                x_data = max_data["range"].values

                x_data = x_data[~np.isnan(y_data)]
                y_data = y_data[~np.isnan(y_data)]

                p0 = [np.max(y_data), x_data[np.argmax(y_data)], 100]

                try:

                    params, cov = curve_fit(gaussian, x_data, y_data, p0=p0)
                    A_fit, mu_fit, sigma_fit = params
                    results.append(
                        (time.values, lidar_data_dst["azimuth"].item(), mu_fit)
                    )
                except Exception as e:
                    print("had problem fitting gaussian, adding nan for this time step")
                    print(e)

            lidar_data_df = pd.DataFrame(results, columns=["time", "azimuth", "range"])

        else:
            raise ValueError('Invalid method_use. Please choose either "V1" or "V2".')
        return lidar_data_df

    def fit(
        self,
        lidardata: xr.Dataset,
        initial_guess: tuple[float, float, float],
        CNR_hardtarget: float = 0,
        method_use: Literal["V1", "V2"] = "V2",
        R_add_hardtargets: float = 0,
        bounds_dist: float = 300,
        bounds_angle: float = 30,
        redo_iterations: int = 2,
        max_distance: float = 100,
        print_result: bool = False,
        plot: bool = False,
    ):
        """Fit the measurements with the coordinate list

        Args:
            lidardata (xr.Dataset): xarray dataset with the columns cnr [db], azimuth [°] and range [m], containing the measurements of the hard targets. Here, no elevation is considered.
            initial_guess (tuple[float, float, float]): Initial guess [position x, position y, initial lidar orientation] for the fit.
            CNR_hardtarget (float, optional): CNR threshold for a measurement to be considered a hard target measurement, in dB. Defaults to 0.
            method_use (Literal[&quot;V1&quot;, &quot;V2&quot;], optional): Method to use for the fit, V1 is Rott et al. 2022, V2 is the updated version. Defaults to "V2".
            R_add_hardtargets (float, optional): Range to add to the detected hard target positions, as the turbine coordinate is for the center of the tower, while the hard target results in the tower shell position. Defaults to 5.
            bounds_dist (float, optional): bounds for the posotion to deviate from the initial guess. Defaults to 300.
            bounds_angle (float, optional): bounds for the posotion to deviate from the initial guess. Defaults to 30.
            redo_iterations (int, optional): number of iterations to redo the fit with outliers removed. Defaults to 2.
            max_distance (float, optional): maximum distance for a measurement to be considered an outlier. Defaults to 100.
            print_result (bool, optional): whether to print the fit result. Defaults to False.
            plot (bool, optional): whether to plot the fit result. Defaults to False.

            Returns:
                result: result of the fit, including the optimized parameters and the cost function value. Object of scipy fit results
        """

        lidar_data_df = self._prepare_data(
            lidardata, CNR_hardtarget, method_use=method_use
        )

        if method_use == "V2":
            # add half of the distance between the turbines to get the position of the turbine center instead of the turbine front
            lidar_data_df["range"] = lidar_data_df["range"] + R_add_hardtargets

        # Initial guess: x,y,initial lidar orientation
        bnds = (
            (initial_guess[0] - bounds_dist, initial_guess[0] + bounds_dist),
            (initial_guess[1] - bounds_dist, initial_guess[1] + bounds_dist),
            (initial_guess[2] - bounds_angle, initial_guess[2] + bounds_angle),
        )

        # get first fit results
        result = minimize(
            Northalignment.cost_function,
            initial_guess,
            method="Nelder-Mead",
            bounds=bnds,
            args=(self.HardTargets, lidar_data_df),
        )
        all_hard_targets_lidar = lidar_data_df.copy()

        ## if method v2 is used, perform outlier removal and redo fit with removed outliers
        if method_use == "V2":
            for i in range(redo_iterations):
                distances = Northalignment.cost_function(
                    result.x, self.HardTargets, lidar_data_df, summed_cost=False
                )
                if np.any(distances > max_distance):
                    print(
                        f"\t --> Outliers detected, performing second fit with {len(distances[distances>=max_distance])} removed outliers, will keep ({(distances<max_distance).sum()})..."
                    )
                    # remove outliers
                    lidar_data_df = lidar_data_df.loc[distances < max_distance]

                    # perform second fit with removed outliers
                    result = minimize(
                        Northalignment.cost_function,
                        initial_guess,
                        method="Nelder-Mead",
                        bounds=bnds,
                        args=(self.HardTargets, lidar_data_df),
                    )
        self.fit_result = result

        if print_result:
            print("x_offset = {:2.2}".format(result.x[0]))
            print("y_offset = {:2.2}".format(result.x[1]))
            print("North_offset = {:3.2f}\N{DEGREE SIGN}".format(result.x[2]))

        self.lidar_data_df = lidar_data_df
        self.all_hard_targets_lidar = all_hard_targets_lidar
        if plot:
            self.plot(
                result.x,
                lidar_data_df,
                self.HardTargets,
                all_hard_targets_lidar=all_hard_targets_lidar,
            )
        return result

    @staticmethod
    def plot(
        params: tuple[float, float, float],
        lidar_data_df: pd.DataFrame,
        HardTargets: pd.DataFrame,
        interactive: bool = False,
        all_hard_targets_lidar: pd.DataFrame | None = None,
    ):
        """Plot the alignment and results

        Args:
            params (tuple[float, float, float]): params of the fit, including x_offset, y_offset and azi_offset (each float) of the lidar in the global coordinate system.
            lidar_data_df (pd.DataFrame): lidar data, must contain columns range and azimuth. Here, no elevation is considered.
            HardTargets (pd.DataFrame): Hard target positions, must contain columns x and y
            interactive (bool, optional): whether to create an interactive plot (with plotly). Defaults to False.
            all_hard_targets_lidar (pd.DataFrame | None, optional): all detected hard targets. Defaults to None.

    
        """
        x_offset, y_offset, azi_offset = params

        cost = Northalignment.cost_function(params, HardTargets, lidar_data_df)

        los_x = (
            np.sin(np.deg2rad(lidar_data_df["azimuth"] - azi_offset))
            * lidar_data_df["range"]
        )  # + x_offset
        los_y = (
            np.cos(np.deg2rad(lidar_data_df["azimuth"] - azi_offset))
            * lidar_data_df["range"]
        )  # + y_offset

        if all_hard_targets_lidar is not None:
            los_x_all = (
                np.sin(np.deg2rad(all_hard_targets_lidar["azimuth"] - azi_offset))
                * all_hard_targets_lidar["range"]
            )  # + x_offset
            los_y_all = (
                np.cos(np.deg2rad(all_hard_targets_lidar["azimuth"] - azi_offset))
                * all_hard_targets_lidar["range"]
            )  # + y_offset

        if not interactive:
            fig, ax = plt.subplots()
            if all_hard_targets_lidar is not None:
                ax.scatter(
                    los_x_all, los_y_all, label="All detected hard targets", c="tab:red"
                )
            ax.scatter(los_x, los_y, label="Considered Measurements")
            ax.scatter(
                HardTargets["x"] - x_offset,
                HardTargets["y"] - y_offset,
                marker="1",
                label="Hard Targets",
            )
            ax.set_aspect("equal")
            ax.set(
                xlabel=r"Easting from Lidar pos, $x\ [\mathrm{m}]$",
                ylabel=r"Northing from Lidar pos, $y\ [\mathrm{m}]$",
                title=f'Lidar positions at {lidar_data_df.time.min().strftime("%Y-%m-%d %H:%M")}'
                + "\n"
                + f"({x_offset:.1f}m, {y_offset:.1f}m), {azi_offset:.2f}°, $={cost:.2e}",
            )
            ax.tick_params(axis="both", which="major")
            ax.scatter(0, 0, marker="+", s=80, c="tab:red", label="lidar position")
            ax.grid(True, alpha=0.3, ls="--")
            ax.legend(loc="lower right")
            return ax
        else:
            fig = go.Figure()

            fig.add_trace(
                go.Scattergl(
                    x=HardTargets["x"] - x_offset,
                    y=HardTargets["y"] - y_offset,
                    mode="markers",
                    name="Hard Targets",
                    marker_symbol="cross",
                )
            )

            if all_hard_targets_lidar is not None:
                fig.add_trace(
                    go.Scattergl(
                        x=los_x_all,
                        y=los_y_all,
                        mode="markers",
                        name="All detected hard targets",
                    )
                )
            fig.add_trace(
                go.Scattergl(
                    x=los_x, y=los_y, mode="markers", name="Considered measurements"
                )
            )
            fig.add_trace(
                go.Scattergl(
                    x=[0],
                    y=[0],
                    mode="markers",
                    name="lidar position",
                    marker_symbol="x",
                    marker_size=10,
                    marker_color="red",
                )
            )
            fig.update_layout(
                title=f'Lidar positions at {lidar_data_df.time.min().strftime("%Y-%m-%d %H:%M")}'
                + "\n"
                + f"({x_offset:.1f}m, {y_offset:.1f}m), {azi_offset:.2f}°, $={cost:.2e}",
                xaxis_title="Easting from Lidar pos, <br>x [m]",
                yaxis_title="Northing from Lidar pos, <br>y [m]",
            )
            return fig

    @staticmethod
    def cost_function(
        params: tuple[float, float, float],
        HardTargets: pd.DataFrame,
        lidardata: pd.DataFrame,
        summed_cost: bool = True,
    ):
        """Cost function to optimize the lidar positioning and alignment.

        Reference:
            Rott, A., Schneemann, J., Theuer, F., Trujillo Quintero, J. J., & Kühn, M. (2022). Alignment of scanning lidars in offshore wind farms. Wind Energy Science, 7(1), 283–297. https://doi.org/10.5194/wes-7-283-2022

        Args:
            params (list[float, float, float]): Includes x_offset, y_offset and azi_offset (each float) of the lidar in the global coordinate system.
            HardTargets (pd.DataFrame): DataFrame containing the positions of the Hard Targets. Columns must be x and y
            lidardata (pd.DataFrame): lidar data from the hard target measurements. Must contain columns range and azimuth. Here, no elevation is considered.

        Returns:
            float:  cost of the function with given parameters
        """
        x_offset, y_offset, azi_offset = params
        # ---- get positions of hard targets and expand the array for calculation ---- #
        t_x = np.expand_dims(HardTargets["x"].values, axis=1)
        t_y = np.expand_dims(HardTargets["y"].values, axis=1)
        # ---------------- calc lidar positions with given parameters ---------------- #
        ht_x = (
            np.sin(np.deg2rad(lidardata["azimuth"] - azi_offset)) * lidardata["range"]
            + x_offset
        ).values
        ht_y = (
            np.cos(np.deg2rad(lidardata["azimuth"] - azi_offset)) * lidardata["range"]
            + y_offset
        ).values
        # ----------------- calculate the cost for the given parameters -------------- #
        cost = np.sqrt((ht_x - t_x) ** 2 + (ht_y - t_y) ** 2)
        cost = np.min(cost, axis=0)
        if summed_cost:
            return np.linalg.norm(cost)
        else:
            return cost
