###### Plotting Utils #######
# Refer to: https://github.com/imagingofthings/DeepWave/blob/master/datasets/Pyramic/color_plot.py
import collections.abc as abc

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import mpl_toolkits.basemap as basemap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans


def wrapped_rad2deg(lat_r, lon_r):
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


def cart2pol(x, y, z):
    """
    Cartesian coordinates to Polar coordinates.
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


def cart2eq(x, y, z):
    """
    Cartesian coordinates to Equatorial coordinates.
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


def is_scalar(x):
    """
    Return :py:obj:`True` if `x` is a scalar object.
    """
    if not isinstance(x, abc.Container):
        return True

    return False


def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.
    """
    r = np.array([r]) if is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
            .to_cartesian()
            .xyz.to_value(u.dimensionless_unscaled)
    )
    return XYZ


def cmap_from_list(name, colors, N=256, gamma=1.0):
    """
    Parameters
    ----------
    name : str
    colors :
        * a list of (value, color) tuples; or
        * list of color strings
    N : int
        Number of RGB quantization levels.
    gamma : float
        Something?

    Returns
    -------
    cmap : :py:class:`matplotlib.colors.LinearSegmentedColormap`
    """
    from collections.abc import Sized
    import matplotlib.colors

    if not isinstance(colors, abc.Iterable):
        raise ValueError('colors must be iterable')

    if (isinstance(colors[0], Sized) and
            (len(colors[0]) == 2) and
            (not isinstance(colors[0], str))):  # List of value, color pairs
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))

    return matplotlib.colors.LinearSegmentedColormap(name, cdict, N, gamma)


def draw_map(I, R, lon_ticks, catalog=None, show_labels=False, show_axis=False):
    """
    Parameters
    ==========
    I : :py:class:`~numpy.ndarray`
        (3, N_px)
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    """

    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bm = basemap.Basemap(projection='mill',
                         llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                         llcrnrlon=R_az_min, urcrnrlon=R_az_max,
                         resolution='c',
                         ax=ax)

    if show_axis:
        bm_labels = [1, 0, 0, 1]
    else:
        bm_labels = [0, 0, 0, 0]
    bm.drawparallels(np.linspace(R_el_min, R_el_max, 5),
                     color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                     textcolor='#565656', zorder=0, linewidth=2)
    bm.drawmeridians(lon_ticks,
                     color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                     textcolor='#565656', zorder=0, linewidth=2)

    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)

    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)

    N_px = I.shape[1]
    mycmap = cmap_from_list('mycmap', I.T, N=N_px)
    colors_cmap = np.arange(N_px)
    ax.tripcolor(triangulation, colors_cmap, cmap=mycmap,
                 shading='gouraud', alpha=0.9, edgecolors='w', linewidth=0.1)

    Npts = 6  # find N maximum points
    I_s = np.square(I).sum(axis=0)
    max_idx = I_s.argsort()[-Npts:][::-1]
    x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))  # stack N max energy points
    km_res = KMeans(n_clusters=1).fit(x_y)  # apply k-means to max points
    clusters = km_res.cluster_centers_  # get center of the cluster of N points
    ax.scatter(R_x[max_idx], R_y[max_idx], c='b', s=5)  # plot all N points
    ax.scatter(clusters[:, 0], clusters[:, 1], s=500, alpha=0.3)  # plot the center as a large point
    cluster_center = bm(clusters[:, 0][0], clusters[:, 1][0], inverse=True)

    return fig, ax, cluster_center


def comp_plot(x, y, x_g, y_g, timestamp, azimuth, elevation, ir_times, out_folder, main_title):
    err_az = [a_i - b_i for a_i, b_i in zip(x, x_g)]
    err_el = [a_i - b_i for a_i, b_i in zip(y, y_g)]
    df = {}
    df['azimuth_gt'] = x_g
    df['elevation_gt'] = y_g
    df['azimuth_est'] = x
    df['elevation_est'] = y
    df['azimuth_error'] = err_az
    df['elevation_error'] = err_el
    df['timestamp'] = timestamp
    df = pd.DataFrame(df)

    # plot groundtruth and estimated trajectory
    plt.close("all")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Trajectory", "Localization Error"))
    fig.add_trace(go.Scatter(x=x, y=y, name='estimated', mode='markers', marker_size=20), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_g, y=y_g, name='ground truth', mode='markers', marker_size=20), row=1, col=1)

    # plot localization error box plot
    fig.add_trace(go.Box(y=df['azimuth_error'].values, name='azimuth error'), row=1, col=2)
    fig.add_trace(go.Box(y=df['elevation_error'].values, name='elevation error'), row=1, col=2)
    fig.update_xaxes(range=[-100, 100], title_text='azimuth', row=1, col=1)
    fig.update_yaxes(range=[-40, 40], title_text='elevation', row=1, col=1)
    fig.update_yaxes(range=[-15, 20], title_text='degree', row=1, col=2)
    fig.update_layout(title_text=main_title, title_x=0.5, title_font_size=40)
    fig.write_html(out_folder + "boxplot.html")
    fig.show()

    # plot azimuth and elevation change over time
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Azimuth over time", "Elevation over time"))
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.azimuth_est, mode='markers',
                             marker_size=abs(df['azimuth_error']) / abs(df['azimuth_error']).max() * 50,
                             name='estimated'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.azimuth_gt, mode='markers+lines', name='ground truth'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.elevation_est, mode='markers',
                             marker_size=abs(df['elevation_error']) / abs(df['azimuth_error']).max() * 50,
                             name='estimated'),
                  row=1, col=2)
    fig.add_trace(go.Line(x=df.timestamp, y=df.elevation_gt, mode='markers+lines', name='ground truth'), row=1, col=2)
    for n, i in enumerate(ir_times):
        fig.add_vline(x=i * 1000, line_dash='dash', line_color='blue', row=1, col=1)  # at what frame
        fig.add_scatter(x=[i * 1000],
                        y=[azimuth[n]],
                        marker=dict(
                            color='green',
                            size=20
                        ),
                        name='actual gt', row=1, col=1)

        fig.add_vline(x=i * 1000, line_dash='dash', line_color='blue', row=1, col=2)  # at what frame
        fig.add_scatter(x=[i * 1000],
                        y=[elevation[n]],
                        marker=dict(
                            color='green',
                            size=20
                        ),
                        name='actual gt', row=1, col=2)
    fig.update_yaxes(range=[-100, 100], title_text='azimuth', row=1, col=1)
    fig.update_yaxes(range=[-40, 40], title_text='elevation', row=1, col=2)
    fig.update_xaxes(title_text='time', row=1, col=1)
    fig.update_xaxes(title_text='time', row=1, col=2)
    fig.update_layout(title_text=main_title, title_x=0.5, title_font_size=40)
    fig.write_html(out_folder + "time.html")
    fig.show()
