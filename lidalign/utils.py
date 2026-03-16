import pathlib
import matplotlib.pyplot as plt


def load_template():
    path = pathlib.Path(
        pathlib.Path(__file__).parent, "../application/SSCValidationHeligoland/publication_style.mplstyle"
    )
    plt.style.use(str(path))
    print("Using lidar_monitoring.mplstyle as matplotlib style template")


load_template()


def publication_figure(
    relative_width: float = 1 / 2, width_paper: float = 6.30045, height=3, fig_only: bool = False, **kwargs
):  ## width of IOP template
    """
    width_paper = 6.3.. -> width of IOP template, find more here: https://tex.stackexchange.com/questions/39383/determine-text-width
    """
    figure_width = width_paper * relative_width
    preset_kw = dict(dpi=300, figsize=(figure_width, height))
    if fig_only:
        fig = plt.figure(**{**preset_kw, **kwargs})
        return fig
    fig, ax = plt.subplots(**{**preset_kw, **kwargs})
    return fig, ax
