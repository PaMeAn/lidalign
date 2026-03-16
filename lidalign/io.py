# %%
"""
Scripts related to reading of WindCubeScan data from Vaisala.
Authors: Paul Meyer, Ignace Ransquin (ForWind WESys)
Created: 2025-08-25

V0 2025-08-25: Initial push to Git + minor changes
"""
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
import gzip
from tqdm import tqdm
from typing import Literal
import datetime
import re
from pathlib import Path
import io
import matplotlib.pyplot as plt
import netCDF4 as nc


datatypes = {
    "wind_and_aerosols_data": {"file_split_columns": ["lidar", "date", "time", "type", "ScanID", "ext"]},
    "environmental_data": {"file_split_columns": ["lidar", "f1", "f2", "date", "time"]},
    "scans": {"file_split_columns": ["lidar", "ScanID", "name"]},
}


class FileDB:
    def __init__(self, path: str, regex: str = "*.*"):
        self.path = pathlib.Path(path)
        if not self.path.exists():
            raise ValueError(f"Path does not exist, double check: \n{str(self.path)}")
        self.all_files = list(self.path.glob(regex))

    def _get_file_information(self, filepattern: str | list, names=None):

        groups = []
        if isinstance(filepattern, list):
            for f in tqdm(self.all_files, desc="Evaluating patterns"):
                for filepatt in filepattern:
                    match = re.match(filepatt, f.name)
                    if match is not None:
                        groups.append(match.groups())
                        continue

        else:
            for f in self.all_files:

                match = re.match(filepattern, f.name)

                if match is not None:
                    groups.append(match.groups())
                else:
                    print(f.name)
            groups = [re.match(filepattern, f.name).groups() for f in self.all_files]

        self.info_df = pd.DataFrame(groups, columns=names.keys())  # , dtype = names)
        self.info_df = self.info_df.astype(names)
        self.info_df["filename"] = [f.name for f in self.all_files]
        self.info_df["full_path"] = [str(f) for f in self.all_files]  # tqdm(self.all_files, desc="Resolving paths")

    def filter_file_names(
        self,
        start: str = None,
        end: str = None,
        timedelta_back: str = None,
        closest_to_time: str = None,
        max_timedist="3h",
    ):
        if closest_to_time is not None:
            self.closest_time(closest_to_time, max_timedist)
            return self

        end = datetime.datetime.now() if end is None else end
        if start is None:
            if timedelta_back is None:
                start = self.info_df["file_start_date"].min()
            else:
                start = end - pd.to_timedelta(timedelta_back)

        # self.info_df["UTC"].max()
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.start = self.start.tz_localize("UTC") if self.start.tzinfo is None else self.start
        self.end = self.end.tz_localize("UTC") if self.end.tzinfo is None else self.end

        self.filtered_files = self.info_df.where(self.info_df["file_start_date"].between(self.start, self.end)).dropna()

        return self

    def get_filtered_filelist(
        self,
        start: str = None,
        end: str = None,
        timedelta_back: str = None,
        filename_regex: str = None,
    ):
        """Get the reduced filelist, depending on start and end time, as well as a regex to search for in the filenames

        Args:
            start (str, optional): Start date+time of the lidar data to read. If None, start is set to first date. Defaults to None.
            end (str, optional): End date+time of lidar data to read. If None, end is set to NOW. Defaults to None.
            filename_regex (str, optional): regex to filter the filenames. Can be e.g., the scan type and number: ppi_93 or similar. Defaults to None.

        Returns:
            self: Database Object with filtered_files_list attribute, which contains the filtered files list
        """
        end = datetime.datetime.now() if end is None else end
        if start is None:
            if timedelta_back is None:
                start = self.info_df["UTC"].min()
            else:
                start = end - pd.to_timedelta(timedelta_back)

        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.start = self.start.tz_localize("UTC") if self.start.tzinfo is None else self.start
        self.end = self.end.tz_localize("UTC") if self.end.tzinfo is None else self.end

        print(f"\t Filtering for {start} to {end}, {filename_regex} ")
        self.filtered_files = self.info_df.where(self.info_df["UTC"].between(self.start, self.end)).dropna()


        if filename_regex is not None:


            self.filtered_files = self.filtered_files.loc[self.filtered_files["full_path"].str.contains(filename_regex)]

        self.filtered_files_list = self.filtered_files["pathlib"].dropna().values
        print(f"\t --> {len(self.filtered_files)} files found for given regex and time range")

        return self

    def closest_time(self, times, max_time_dist: str = "1h"):
        """
        Get items closest to given time"""

        info_df = self.info_df.copy().reset_index()
        idx = [np.abs((info_df["UTC"] - t).values).argsort()[0] for t in times]
        if not list(np.unique(idx)) == idx:
            print("Non-unique return, will return only uniqe")
            idx = np.unique(idx)
        # check max dist

        dt = pd.to_timedelta(max_time_dist)
        idx = [i for i, t in zip(idx, times) if np.abs(info_df["UTC"].loc[i] - t) < dt]
        #     idx = np.unique(idx)
        self.filtered_files = info_df.loc[idx]

        # self.filesdf = [f for f in self.all_files if re.match(filepattern, f.name)]


class RawEnvironmentalDB(FileDB):

    filepattern = r".*(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}).csv"
    names = {"Time": str}

    def __init__(self, path: str):
        super().__init__(path, regex="*.csv")
        self._get_file_information(self.filepattern, self.names)
        self.info_df["file_start_date"] = pd.to_datetime(self.info_df["Time"], format="%Y%m%d_%H%M%S", utc=True)

    def read_period(self, **kwargs):
        self.filter_file_names(**kwargs)
        files = self.filtered_files["full_path"].values
        if len(files) == 0:
            print("No files found for the given time period.")
            return pd.DataFrame()
        df = pd.concat([self.read_file(f) for f in tqdm(files, desc="Reading files")], axis=0)
        return df

    def read_file(self, f):
        return pd.read_csv(f, parse_dates=True, index_col=0, sep=";")


class WindCubeScanDB(FileDB):
    """
    Setup of "database" to read measurements from Vaisala WindCube Scan devices.
    Author: Paul Meyer, Ignace Ransquin (WESys)
    Created: 2025-08-25
    """

    def __init__(
        self,
        path_to_campaign: str,
        # lidar_name: str = "WLS200S-24",
        datatype: Literal["wind_and_aerosols_data", "environmental_data", "scans"] = "wind_and_aerosols_data",
        file_structure: Literal["native_vaisala", "flat"] = "native_vaisala",
        verbose=0,
        prefilter_dates: dict | None = None,
        position: list[float, float, float] = None,
    ):
        """Set up an instance of a "database", depending on the raw path and the lidar name

        Args:
            path_to_campaign (str): filepath to the campaign data. Should be the level before the lidar name
            lidar_name (str, optional): Lidar name, also the directory directly under the path_to_campaign. Defaults to 'WLS200S-24'.
        """
        self.verbose = verbose
        self.path = pathlib.Path(path_to_campaign)  # generate pathlib object
        if not self.path.exists():
            raise ValueError(f"Path does not exist, double check: \n{str(self.path)}")
        # self.lidar_name = lidar_name
        self.datatype = datatype
        self.file_structure = file_structure
        self.position = position
        self._get_all_files(file_structure=file_structure, prefilter_dates=prefilter_dates)
        self._get_file_datetime()

    # def _get_all_files(self, file_structure: Literal["native_vaisala", "flat"] = "native_vaisala",
    #                    prefilter_dates: dict = None):
    #     self._get_all_files(file_structure=file_structure, prefilter_dates = prefilter_dates)

    def _get_all_files(
        self, file_structure: Literal["native_vaisala", "flat"] = "native_vaisala", prefilter_dates: dict = None
    ):
        """Get ALL filenames that are in the "database"

        Args:
            file_structure (str, optional): file_structure, can be native_vaisala or . Defaults to 'native_vaisala'.

        Raises:
            ValueError: _description_
        """
        print(f"Finding all files for {self.datatype}")
        if file_structure == "native_vaisala":
            if self.datatype == "wind_and_aerosols_data":
                # self.all_files = list(self.path.glob("*/wind_and_aerosols_data/*/*.nc*"))
                glob_path = "*/wind_and_aerosols_data/*/*.nc*"

            elif self.datatype == "environmental_data":
                glob_path = "*/environmental_data/*.csv*"
            elif self.datatype == "scans":
                glob_path = "*/scans/*.xscan*"
            else:
                raise ValueError(f'Undefined datatype "{self.datatype}"')

        elif file_structure == "flat":
            if self.datatype == "wind_and_aerosols_data":
                glob_path = "*.nc*"
            elif self.datatype == "environmental_data":
                glob_path = "*EnvironmentalData*.csv*"
            elif self.datatype == "scans":
                glob_path = "*.xscan*"
            else:
                raise ValueError(f'Undefined datatype "{self.datatype}"')

        else:
            raise ValueError(f'Format "{file_structure}" not supported')

        if prefilter_dates is not None:
            dates = pd.date_range(prefilter_dates["start"], prefilter_dates["end"], freq="1d").date

            # glob_path = '^(' + '|'.join([d.strftime('%Y-%m-%d') for d in dates]) +')' + glob_path[1:]
            all_files = []
            for d in tqdm(dates, desc="Prefiltering dates to get all files"):
                all_files += list(self.path.glob(d.strftime("%Y-%m-%d") + glob_path[1:]))
            self.all_files = all_files
        else:
            self.all_files = list(self.path.glob(glob_path))

        print(f"\t --> {len(self.all_files)} files found")

    def _get_file_datetime(self):
        """Get more information from the filenames"""
        splitfilename = [file.stem.replace(".csv", "").split("_") for file in self.all_files]

        if self.datatype != "scans":
            self.info_df = pd.DataFrame(splitfilename, columns=datatypes[self.datatype]["file_split_columns"])
            self.info_df["UTC"] = pd.to_datetime(
                self.info_df["date"] + " " + self.info_df["time"].str.replace("-", ":"),
                utc=True,
            )
        else:
            splitfilename = [[f[0], f[1], "_".join(f[2:])] for f in splitfilename]
            self.info_df = pd.DataFrame(splitfilename, columns=datatypes[self.datatype]["file_split_columns"])

        self.info_df["pathlib"] = self.all_files
        self.info_df["filename"] = [f.name for f in self.all_files]
        self.info_df["full_path"] = [str(f) for f in self.all_files]
        if self.datatype == "scans":
            if self.file_structure != "flat":
                self.info_df["UTC"] = pd.to_datetime([f.parts[-3] for f in self.info_df["pathlib"].values], utc=True)
            else:
                self.info_df["UTC"] = pd.to_datetime("2025-01-01", utc=True)

    def get_extended_file_information(self, df=None):
        import os

        df = self.info_df if df is None else df
        size = []
        for f in tqdm(df["full_path"], desc="Getting file size"):
            size.append(os.stat(f).st_size)

        df["size"] = size
        df = df.set_index("UTC").sort_index()
        return df

    def get_data(
        self,
        start: str = None,
        end: str = None,
        concatenated=False,
        timedelta_back: str = None,
        filename_regex: str = None,
        **read_kwargs,
    ):
        """Wrapper to get the data after filtering the total files list

        Args:
            concatenated (bool, optional): If true, retrieved datasets are concatenated to a single one with time as main dimension. If False, a list of datasets is returned. Defaults to False.

        kwargs:
            keywords for WindCubeScanDB.get_filtered_filelist()

        Returns:
            xr.dataset or List: List or Xarray Dataset, depending on concatenated keyword
        """
        self.get_filtered_filelist(start, end, timedelta_back, filename_regex)
        if self.datatype == "wind_and_aerosols_data":
            ds = self.read_wind_files(self.filtered_files_list, concatenated=concatenated, **read_kwargs)

        elif self.datatype == "environmental_data":
            dflist = [
                pd.read_csv(f, sep=";", header=0)
                for f in tqdm(self.filtered_files_list, desc="Reading environmental files")
            ]
            if len(dflist) == 0:
                print("No environment data")
                return []
            df = pd.concat(dflist)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            ds = df.set_index(["Timestamp", "Name"]).to_xarray()
            ds = ds.drop_vars("Unit")
            # need to create datetime index again
            ds["Timestamp"] = pd.to_datetime(ds["Timestamp"], utc=True)
    
        return ds

   

    def read_wind_files(
        self,
        files: list,
        concatenated=True,
        generate_utc_time=True,
        sel_kwargs: {} = None,
        query_kwargs: {} = None,
        max_n: int = None,
        processes: int = 1,
        VAD_ranges: list = None,
        **read_kwargs,
    ):
        """Read datasets from filenames

        Args:
            files (list): list of files to read
            concatenated (bool, optional): If true, retrieved datasets are concatenated to a single one with time as main dimension. If False, a list of datasets is returned. Defaults to False.
            generate_utc_time (bool, optional): If true, an additional variable with pandas datetime is created. Defaults to True.

        Returns:
            data: depending on concatenated, list or concatenated dataset of lidar measuremnts
        """

        def read_single(file):
            dsf = self._read_wind_file(file, generate_utc_time, position=self.position)
            if sel_kwargs is not None:
                dsf = dsf.sel(**sel_kwargs)

            return dsf

        data = []
        if max_n is not None:
            files = files[:max_n]
        if processes == 1:
            for file in tqdm(files, desc="Reading files"):
                dsf = self._read_wind_file(file, generate_utc_time, position=self.position, **read_kwargs)
                
                if sel_kwargs is not None:

                    dsf = dsf.sel(**sel_kwargs)
                if query_kwargs is not None:
                    dsf = dsf.query(**query_kwargs).dropna(dim="range", how="all")
                    for dim in dsf.dims:
                        dsf = dsf.dropna(dim=dim, how="all")
                if VAD_ranges is not None:
                    self.get_VAD_reconstruction(dsf, VAD_ranges)
                data.append(dsf)

        else:
            # is not really faster
            from joblib import Parallel, delayed

            data = Parallel(n_jobs=processes, verbose=10)(delayed(read_single)(f) for f in files)

        if concatenated:
            all_ds = xr.concat(data, dim="timestamp")
            return all_ds
        else:
            return data

    @staticmethod
    def _read_wind_file(
        filename: str,
        filter: bool = True,
        remove_azimuth_offset: bool = False,
        get_middle_azimuth: bool = True,  ## middle of azimuth bin instead of end
        returntype: Literal["full", "simplified"] = "simplified",
        returnformat: Literal["xarray", "dict"] = "dict",
        include_attrs: bool = True,
        position: list[float] = None,
        **kwargs,
    ):
        """Read single netcdf file in Vaisala format, with additional functionalities

        Args:
            filename (str): name of the netcdf file to read
            generate_utc_time (bool, optional): generate a utc time for the xarray dataset. Defaults to True.
            filter (bool, optional): filter the data (status == 1). Defaults to True.
            remove_azimuth_offset (bool, optional): removes the azimuth north offset that can be found in the files. Defaults to False.
            get_middle_azimuth (bool, optional): gets middle azimuth. Defaults to True.
            returnformat (Literal["xarray", "dict"], optional): return data as array in a dict or as xarray dataset. Defaults to "dict".
            include_attrs (bool, optional): include attributes in the returned dataset. Defaults to True.

        Returns:
            _type_: _description_
        """
        # TODO: check functionality for dict return and volumetric scans (multiple sweeps in one file)

        if isinstance(filename, str):
            filename = Path(filename)

        # inner function to read the data
        def _extract_ncdata(filedata):
            return_data = []
            keys = filedata.groups.keys()
            sweeps = [g for g in keys if "sweep" in g.lower()]
            azimuth_correction = filedata.groups["georeference_correction"]["azimuth_correction"][:]

            # TODO: numpy return with volume stuff
            for sweep in sweeps:
                sweep_no = sweep.split("_")[-1]
                variables = list(filedata.groups[sweep].variables.keys())
                toplevel_variables = ["latitude", "longitude", "altitude", "sweep_fixed_angle"]
                if returntype == "simplified":
                    variables = [
                        "azimuth",
                        "elevation",
                        "range",
                        "radial_wind_speed",
                        "radial_wind_speed_status",
                        "cnr",
                        "rotation_direction",
                        "ray_angle_resolution",
                        "sweep_mode",
                        "sweep_index",
                        "timestamp",
                        "range_gate_length",
                        "rotation_direction"
                    ]
                data = {var: np.atleast_1d(filedata.groups[sweep][var][:]) for var in variables}
                toplevel_data = {var: np.atleast_1d(filedata[var][:]) for var in toplevel_variables}
                data = data | toplevel_data

                data["time"] = pd.to_datetime(data["timestamp"], utc=True, format="ISO8601")
                
                variables += ["time"]
                # data["mean_elevation"] = np.full_like(data["elevation"], np.mean(data["elevation"]))
                # variables += ["mean_elevation"]

                if get_middle_azimuth and data["sweep_mode"][0] != "fixed":
                   
                    # direct: clockwise, indirect: anticlockwise
                    sign = np.where(data["rotation_direction"] == "direct", 1, -1)
                    azi_diffs = np.diff(
                        data["azimuth"], prepend=data["azimuth"][0] - sign[0] * data["ray_angle_resolution"]
                    )
                    # correct for 360/0 jumps
                    azi_diffs = (azi_diffs + 180) % 360 - 180
                    azimuth_middle = (data["azimuth"] - sign * azi_diffs / 2) % 360
                    data["azimuth"] = azimuth_middle
                   

                if returnformat == "dict":
                    data["sweep"] = np.array([sweep_no])
                    # data["time"] = np.atleast_1d(filedata.groups[sweep]["timestamp"][:])

                    return_data.append(data)
                elif returnformat == "xarray":
                    dsi = xr.Dataset()

                    for var in variables:
                        dim = filedata.groups[sweep][var].dimensions
                        if len(dim) != 0:
                            dsi[var] = (dim, data[var])
                        else:
                            dsi[var] = (("other"), data[var])  # no dimension

                        if include_attrs:
                            attrs = dict()
                            nc_attrs = filedata.groups[sweep][var].ncattrs()
                            for key in nc_attrs:
                                attrs[key] = getattr(filedata.groups[sweep][var], key)
                            dsi[var].attrs = attrs

                    dsi["sweep"] = sweep_no
                    dsi["time"] = pd.to_datetime(dsi["time"])
                    # adjust the time, gets lost when converting into xarray
                    # dsi = dsi.assign_coords(
                    #     time_utc=(
                    #         "time",
                    #         pd.to_datetime(dsi["timestamp"].values, utc=True, format="ISO8601"),
                    #     )
                    # )
                    # # dsi = dsi.set_index(time="time_utc")  # replace index
                    # dsi["time"] = pd.to_datetime(dsi.coords["time"].astype(int), utc=True)

                    return_data.append(dsi.squeeze())

                # dss.attrs = dss.attrs | ds["/"].attrs
            if returnformat == "dict":
                return_dict = dict()
                if len(sweeps) > 1:
                    for var in variables:
                        return_dict[var] = np.concatenate([return_data[i][var] for i, v in enumerate(sweeps)], axis=0)
                else:
                    return_dict = {var: return_data[0][var] for var in variables}
                return_dict["time"] = return_dict["timestamp"]
                return return_dict
            elif returnformat == "xarray":

                all_ds = xr.concat(return_data, dim="time")
                all_ds["azimuth_correction"] = azimuth_correction

                if returntype == "full":
                    metadata_variables = [
                        var
                        for var in list(all_ds.keys())
                        if not (("Sweep" in var) or ("georeference_correction" in var))
                    ]

                    sweepdims = []
                    for var in metadata_variables:
                        if all_ds[var].dims == ("sweep",):
                            ## the variables with sweep as dimension (sweep_fixed_angle,...)
                            if len(all_ds[var]) > 1:
                                sweepdims.append(all_ds[var].to_dataset())
                            else:
                                all_ds[var] = all_ds[var].squeeze()
                        else:
                            all_ds[var] = all_ds[var]

                    if len(sweepdims) > 0:
                        sweepdims_ds = xr.merge(sweepdims)
                        sweepdims_ds = sweepdims_ds.assign_coords(sweep=np.arange(0, len(sweepdims_ds.sweep)) + 1)

                        all_ds[list(sweepdims_ds.keys())] = sweepdims_ds.sel(sweep=all_ds["sweep_index"]).drop_vars(
                            "sweep"
                        )

                return all_ds

        ## Read and postprocess wind files
        if filename.suffix == ".gz":
            with gzip.open(filename, "rb") as f:
                with nc.Dataset("dummy", mode="r", memory=f.read()) as da:
                    ds = _extract_ncdata(da, **kwargs)
        else:
            with nc.Dataset(filename, mode="r") as da:
                ds = _extract_ncdata(da, **kwargs)


        if filter:
            conditions = (ds["radial_wind_speed_status"] == 1) & (ds["cnr"] < 5)
            if returnformat == "xarray":
                ds["radial_wind_speed"] = ds["radial_wind_speed"].where(conditions, np.nan)
            else:
                ds["radial_wind_speed"] = np.where(conditions, ds["radial_wind_speed"], np.nan)

        if remove_azimuth_offset:
            if returnformat != "xarray":
                raise ValueError("remove_azimuth_offset only works for xarray returnformat")
            attrs = ds["azimuth"].attrs
            ds["azimuth"] = (ds["azimuth"] + ds["azimuth_correction"]) % 360
            ds["azimuth_correction"] = 0
            ds["azimuth"].attrs = {
                **attrs,
                "long_name": "azimuth_angle_from_lidar_north",
                "comments": "Scanning heads azimuth angle relative to lidar north when each measurement finished",
            }

        if get_middle_azimuth and ds["sweep_mode"][0] != "fixed":


            sign = xr.where(ds["rotation_direction"] == "direct", 1, -1)
            azimuth_middle = (ds["azimuth"] - ds["ray_angle_resolution"] * sign / 2) % 360


            if returnformat == "dict":
                ds["azimuth"] = azimuth_middle
            else:
                attrs = ds["azimuth"].attrs
                ds["azimuth"] = azimuth_middle
                ds["azimuth"].attrs = attrs

        if returnformat == "xarray":
            ds["hor_distance"] = np.cos(np.deg2rad(ds["elevation"])) * ds["range"]
            ds["dz"] = np.sin(np.deg2rad(ds["elevation"])) * ds["range"]
        else:
            ds["hor_distance"] = np.cos(np.deg2rad(ds["elevation"])) * ds["range"][:, np.newaxis]
            ds["dz"] = np.sin(np.deg2rad(ds["elevation"])) * ds["range"][:, np.newaxis]

        ds["dx"] = np.sin(np.deg2rad(ds["azimuth"])) * ds["hor_distance"]
        ds["dy"] = np.cos(np.deg2rad(ds["azimuth"])) * ds["hor_distance"]

        if isinstance(position, list):
            ds["lidar_x"] = position[0]
            ds["lidar_y"] = position[1]
            ds["lidar_z"] = position[2]

            ds["x"] = ds["lidar_x"] + ds["dx"]
            ds["y"] = ds["lidar_y"] + ds["dy"]
            ds["z"] = ds["lidar_z"] + ds["dz"]

        if returnformat == "dict":
            return ds
        return ds


