# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 08:46:13 2021

@author: yr3g17
"""
from matplotlib import pyplot as plt
import numpy as np
from uncertainties import unumpy


def create_ufloatmesh(xintv: tuple, xstdev: float, yintv: tuple, ystdev: float,
                      zfunction):
    """
    Create colourmap plot data with uncertainty propagation from the inputs

    Parameters
    ----------
    xintv : tuple
        Two-element tuple of upper and lower bounds for the nominal value of x.
    xstdev : float
        The absolute standard deviation expected of a measurement along x.
    yintv : tuple
        Two-element tuple of upper and lower bounds for the nominal value of y.
    ystdev : float
        The absolute standard deviation expected of a measurement along y.
    zfunction : f(x, y)
        A function of x and y that is compatible with unumpy and ufloat objects
        from the uncertainties package: https://pythonhosted.org/uncertainties/

    Returns
    -------
    x : np.ndarray of unumpy.uarray objects
        unumpy array objects for x, as returned by np.meshgrid for plotting.
    y : np.ndarray of unumpy.uarray objects
        unumpy array objects for y, as returned by np.meshgrid for plotting.
    z : np.ndarray of unumpy.uarray objects
        unumpy array objects for z plotting, created using zfunction argument.

    """
    # Number of data points along an axis
    axres = 200

    # From the given interval and stddev, produce uncertainty arrays in x and y
    x_nominal = np.linspace(*xintv, axres)
    x_stddev = np.array([xstdev] * axres)
    x_uarray = unumpy.uarray(x_nominal, x_stddev)
    y_nominal = np.linspace(*yintv, axres)
    y_stddev = np.array([ystdev] * axres)
    y_uarray = unumpy.uarray(y_nominal, y_stddev)

    # Prepare x, y, and z data in a format that pyplot plots are happy with
    y, x = np.meshgrid(y_uarray, x_uarray)
    z = zfunction(x, y)

    return x, y, z


def plot_ufloatmesh(x, y, z, title: str, xlabel: str, ylabel: str):
    """
    Plots the x y z colour map data produced by create_uplotmesh

    Parameters
    ----------
    x : np.ndarray of unumpy.uarray objects
        unumpy array objects for x, as returned by np.meshgrid for plotting.
    y : np.ndarray of unumpy.uarray objects
        unumpy array objects for y, as returned by np.meshgrid for plotting.
    z : np.ndarray of unumpy.uarray objects
        unumpy array objects for z plotting, created using zfunction argument.
    title : str
        Super title of the resulting plot.
    xlabel : str
        x-label of the resulting plot.
    ylabel : str
        y-label of the resulting plot.

    Returns
    -------
    None.

    """
    # Construct a figure with 2 rows, 3 columns, and origin at the top right
    fig, axs = plt.subplots(*(2, 3), sharex=True, sharey=True,
                            figsize=(10, 6), dpi=100)
    fig.suptitle(title, fontsize=14)

    # We only need the nominal values from the x and y data
    x_nom = unumpy.nominal_values(x)
    y_nom = unumpy.nominal_values(y)

    # Unpack the z data into nominal values, std. deviation, and % uncertainty
    z_nom, z_std = unumpy.nominal_values(z), unumpy.std_devs(z)
    z_rel = z_std / z_nom * 100

    # It's easy for z_rel to become out of hand, so clip the data first
    clipper = np.ceil(np.nanpercentile(z_rel, 80) / 0.07) / 10
    clipper = (-clipper, clipper)
    z_rel = np.clip(z_rel, *clipper)

    # The x, y, and z data is shaped into an array the same shape of the figure
    xs = np.repeat(np.array([x_nom] * 3)[np.newaxis, :], 2, axis=0)
    ys = np.repeat(np.array([y_nom] * 3)[np.newaxis, :], 2, axis=0)
    zs = np.repeat(np.array([z_nom, z_std, z_rel])[np.newaxis, :], 2, axis=0)
    subtitles = ["Nominal Value", "Standard Deviation", "Uncertainty [%]"]

    # Iterate through each of the plots on the figure
    for index, ax in enumerate(axs.flat):
        # Row indexer, Column indexer
        j, i = index // 3, index % 3

        # The shaped data from before is accessed, 1 set of data per plot
        x, y, z = xs[j][i], ys[j][i], zs[j][i]
        ax.axis([x.min(), x.max(), y.min(), y.max()])

        # Plot colourmaps in the top row
        if j == 0:
            c = ax.pcolormesh(x, y, z[:-1, :-1], cmap='viridis')
        # Plot contourmaps in the bottom row
        elif j == 1:
            # If there is more than one unique z val, contour plots can be made
            if len(set(z.flatten())) > 1:
                c = ax.contour(x, y, z)
                ax.clabel(c, inline=True, fontsize=10)
            # Else you can't contour plot if there are no values to contour!
            else:
                c = ax.pcolormesh(x, y, z[:-1, :-1], cmap='twilight')
                ax.plot((x.max(), x.min()), (y.max(), y.min()), "r")
                warningstr = "Runtime Warning: No change in z_stddev, "
                warningstr += "unable to plot contours"
                print(warningstr)
        # Regardless of what graph was just plotted, add a colourbar to the ax
        fig.colorbar(c, ax=ax)

        # Labelling
        if i == 0:
            ax.set(ylabel=ylabel)
        if j == 0:
            ax.set_title(subtitles[i])
        elif j == 1:
            ax.set(xlabel=xlabel)

    # Save to disk
    # plt.savefig(f"{title} uncertain.png".replace("/", "_").replace(" ", "-"))
    plt.show()


if __name__ == "__main__":

    pass
