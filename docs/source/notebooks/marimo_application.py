import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium", app_title="Sea Surface Leveling Application")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🌊🔦 Sea Surface Calibration

    Welcome to the marimo Notebook to illustrate the SeaSurfaceCalibration (SSC)! 
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):

    mo.image(r"figures\ScanningLidar_atTP.jpg", height = 300)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What is the SSC
    The SSC is based on the Sea Surface Leveling by [Rott et al. 2022](https://wes.copernicus.org/articles/7/283/2022/), utilizing lidar scans of the surrounding water to align scanning lidars in an offshore environment. Additionally to the SSL, the **SSC** can be used to calibrate the elevation offset in the lidar scanner, which is crucial for accurate measurements. It is based on the generalization of the SSL by [Gramitzky et al. 2025](https://wes.copernicus.org/preprints/wes-2025-191/) and brings an additional correction for the lidar-water range determination from the CNR signal. 

    ### How are SSC and SSL used in praxis?
    """
    )
    return


@app.cell
def _(mo):
    mo.image('figures/Procedure.png', height = 400)
    return


@app.cell
def _():
    import marimo
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from lidalign.SSL import SSL, WaterRangeDetection, db2linear, linear2db
    from pylito.io.WindCubeScan import WindCubeScanDB
    import pathlib 
    return (
        SSL,
        WaterRangeDetection,
        WindCubeScanDB,
        db2linear,
        linear2db,
        marimo,
        mo,
        np,
        pathlib,
        plt,
    )


@app.cell
def _(mo):
    ## selecting files

    file_browser = mo.ui.file_browser(
        initial_path=r'\Harbour Test\prepared_data\OffshoreAlignment\\', multiple=True,
        filetypes = None ,# ['.nc','.nc.gz'],
        limit = 200
    )
    file_browser

    return (file_browser,)


@app.cell
def _(file_browser):
    files = [file_browser.path(index = i) for i in range(len(file_browser.value))]
    print(files)
    print(file_browser.path(index = 0))
    return (files,)


@app.cell(hide_code=True)
def _(mo):
    load_button = mo.ui.run_button(label="Load new data")
    load_button
    return (load_button,)


@app.cell
def _(marimo):
    # Linke Einstellungen: drei Checkboxen
    bool_linearscale = marimo.ui.checkbox(label="Use linear Scale for fit", value=True)
    bool_elevationerror = marimo.ui.checkbox(label="Consider Elevation error", value=True)

    return bool_elevationerror, bool_linearscale


@app.cell
def _(mo):
    mo.md(
        """
    ---
    # Settings
    """
    )
    return


@app.cell
def _(bool_linearscale, mo):
    mo.md(f"""{bool_linearscale}""")
    return


@app.cell
def _(bool_elevationerror, mo):
    mo.md(f"""{bool_elevationerror}""")
    return


@app.cell
def _(mo):
    # Slider für Wertebereich
    cnr_ht = mo.ui.text(value="-5", label = 'CNR hard target [dB]')
    cnr_min = mo.ui.text(value="-22", label = 'Minimum CNR for signal definition [dB]')
    cnr_cutoff = mo.ui.text(value="-38", label = 'Cutoff CNR (Noise) [dB]')

    return cnr_cutoff, cnr_ht, cnr_min


@app.cell
def _():
    return


@app.cell
def _(cnr_cutoff, cnr_ht, cnr_min, mo):
    mo.md('# Select settings')
    cnr_ht_val = float(cnr_ht.value)
    cnr_min_val = float(cnr_min.value)
    cnr_cutoff_val = float(cnr_cutoff.value)

    return cnr_cutoff_val, cnr_ht_val, cnr_min_val


@app.cell
def _(cnr_ht):
    cnr_ht 
    return


@app.cell
def _():
    return


@app.cell
def _(cnr_min):
    cnr_min
    return


@app.cell
def _(cnr_cutoff):
    cnr_cutoff
    return


@app.cell
def _(WindCubeScanDB, files, load_button, pathlib):

    import xarray as xr
    from tqdm import tqdm

    if load_button.value:
        ds  = xr.concat([WindCubeScanDB._read_wind_file(pathlib.Path(file), returnformat = 'xarray') for file in tqdm(files, desc = 'Reading files...')], dim = 'time')
    else:
        _files = [r'\Harbour Test\WLS200s-17_2010-01-02_17-37-32_ppi_926_50m.nc',
            r'\Harbour Test\WLS200s-17_2010-01-02_17-38-33_ppi_927_50m.nc', 
            r'\Harbour Test\WLS200s-17_2010-01-02_17-39-34_ppi_928_50m.nc']

        ds  = xr.concat([WindCubeScanDB._read_wind_file(pathlib.Path(file), returnformat = 'xarray') for file in tqdm(_files, desc = 'Reading files...')], dim = 'time')


    return (ds,)


@app.cell
def _(ds):
    # Filter data 
    # combdata = ds.where((ds['cnr'].max(dim = 'range')<-8) & 
    #                           (ds['cnr'].max(dim = 'range')>-20) & 
    #                           (ds['cnr'].idxmax(dim = 'range')!= ds.range[0]) &
    #                           (ds['azimuth']>140) & 
    #                           (ds['azimuth']<240)  &
    #                           (~((ds['time'] > pd.to_datetime('2025-05-03 09:55:00')) & (ds['time'] < pd.to_datetime('2025-05-03 10:15:00'))))
    #                        )

    # rollmean = ds['cnr'].max(dim = 'range').rolling(time=10, center=True).mean().interpolate_na(dim = 'time').bfill(dim = 'time').ffill(dim= 'time')

    # rolling_condition = np.abs(ds['cnr'].max(dim = 'range') - rollmean) > 1 ## remove outliers 
    # combdata = combdata.where(~rolling_condition)
    # combdata_nona = combdata.dropna(dim = 'time', how = 'all')

    combdata = ds.copy().dropna(dim = 'time', how = 'all')
    # combdata_nona = combdata
    return (combdata,)


@app.cell(hide_code=True)
def _(mo):
    fitmethod = mo.ui.dropdown(
        options=["SSC", "Gra24", "standard"],
        value="SSC",
        label="Choose fit method:",
        searchable=False,
    )
    fitmethod
    return (fitmethod,)


@app.cell(hide_code=True)
def _(marimo):
    bool_showscale_linear = marimo.ui.checkbox(label="Show in linear Scale", value=True)
    bool_showscale_linear
    return (bool_showscale_linear,)


@app.cell(hide_code=True)
def _(combdata, mo):
    timeslider = mo.ui.slider(steps = range(len(combdata["time"])), label = 'Choose timestamp to plot', value = 100)
    timeslider

    return (timeslider,)


@app.cell
def _(
    WaterRangeDetection,
    bool_linearscale,
    bool_showscale_linear,
    cnr_cutoff_val,
    cnr_ht_val,
    cnr_min_val,
    combdata,
    db2linear,
    fitmethod,
    linear2db,
    mo,
    plt,
    timeslider,
):
    # seldata = combdata.sel(time = '2025-08-12 06:21:15.7', method = 'nearest'
    seldata = combdata.isel(time = timeslider.value).copy()
    returni = WaterRangeDetection(seldata, verbose =3).get_water_range_from_cnr(use_linear_scale =  bool_linearscale.value, 
                                         min_cnr = cnr_min_val, 
                                         cnr_hard_target = cnr_ht_val, 
                                        cnr_noise = cnr_cutoff_val, return_fit= True, func = fitmethod.value, return_all_params=True)

    resi, fitdata, params = returni
    print(params)

    seldataplot = seldata.copy()
    figc, axc = plt.subplots(figsize = (8,5))
    fit, raw = fitdata['fit'], fitdata['data']
    if bool_linearscale.value:
        fit = linear2db(fit) 
        raw = linear2db(raw) 

    if bool_showscale_linear.value:
        fit = db2linear(fit)
        raw = db2linear(raw)
        seldataplot['cnr'] = db2linear(seldataplot['cnr'])
        ylabel = '$CNR$ [linear]'
    else:
        ylabel = '$CNR$ [dB]'
    # fit.plot(x = 'range', ax = axc, c = 'k')
    axc.plot(fit['range'], fit, c= 'k', label = 'Fitted function')
    axc.plot(seldataplot['range'], seldataplot['cnr'], label = 'Measurement data')
    axc.plot(raw['range'], raw, c= 'tab:orange', label = 'Processed Data for fit',ls = ':')
    axc.set(ylabel = ylabel, xlabel = 'Range [m]')
    axc.axvline(resi, c = 'k', ls = '--', label = f'Distance: {resi:.2f}m')
    axc.legend()

    mo.mpl.interactive(plt.gcf())


    return


@app.cell(hide_code=True)
def _(mo):
    start_button = mo.ui.run_button(label="Run on all data")
    start_button

    return (start_button,)


@app.cell
def _(
    SSL,
    bool_linearscale,
    cnr_cutoff_val,
    cnr_ht_val,
    cnr_min_val,
    combdata,
    fitmethod,
    mo,
    start_button,
):
    if start_button.value:
        with mo.redirect_stdout():
            dist_all = SSL(combdata, verbose=1
                      ).get_all_water_ranges(
                                             verbose = 0, 
                                             use_linear_scale =  bool_linearscale.value, 
                                             n_processes = 8, 
                                             min_cnr = cnr_min_val, 
                                             cnr_hard_target = cnr_ht_val, 
                                            cnr_noise = cnr_cutoff_val,
                                             func = fitmethod.value
                                            ).distance_ds

    return (dist_all,)


@app.cell
def _(combdata, dist_all, ds, mo, np, plt):
    #filter a bit 

    d = dist_all['water_range']
    ro = d.rolling(time = 10)
    dist_all['water_range'] = d.where(np.abs(d-ro.mean()) < ro.std()*2)



    fig, ax = plt.subplots()
    kwargsp = dict(vmin=-25, vmax=-5)
    ds["cnr"].plot(ax=ax, x="time", zorder=1, alpha = 0.3, add_colorbar=False,**kwargsp)
    # combdata["cnr"].plot(ax=ax, x="time", zorder=1, **kwargsp)
    combdata['cnr'].where(~np.isnan(dist_all['water_range'])).plot(ax = ax, x = 'time', **kwargsp)
    dist_all['water_range'].plot(ax = ax, x = 'time' ,c = 'k', marker = '.')
    mo.mpl.interactive(plt.gcf())
    return


@app.cell
def _(SSL, bool_elevationerror, dist_all, mo):
    # print(dist_all.dropna(dim = 'time'))
    with mo.redirect_stdout():
        res, figm = SSL.get_misalignment(
                    dist_all, elevation_error=bool_elevationerror.value, plot=True, print_help=True,
                    # x0 = [0.1,0.1,6]
                )

    return (figm,)


@app.cell
def _(figm, mo):
    mo.mpl.interactive(figm)
    return


if __name__ == "__main__":
    app.run()
