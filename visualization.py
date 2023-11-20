import math
import os

import librosa
import scipy.constants as constants
import scipy.special as special
import scipy.signal.windows as windows
import skimage.util as skutil

from apgd import *
from plot_utils import *
from utils import *

ambeovr_raw = {
    # colatitude (deg), azimuth (deg), radius (m)
    "Ch1:FLU": [55, 45, 0.01],
    "Ch2:FRD": [125, -45, 0.01],
    "Ch3:BLD": [125, 135, 0.01],
    "Ch4:BRU": [55, -135, 0.01],
}

tetra_raw = {
    # colatitude (deg), azimuth (deg), radius (m)
    "Ch1:FLU": [55, 45, 0.042],
    "Ch2:FRD": [125, -45, 0.042],
    "Ch3:BLD": [125, 135, 0.042],
    "Ch4:BRU": [55, -135, 0.042],
}

eigenmike_raw = {
    # colatitude, azimuth, radius
    # (degrees, degrees, meters)
    "1": [69, 0, 0.042],
    "2": [90, 32, 0.042],
    "3": [111, 0, 0.042],
    "4": [90, 328, 0.042],
    "5": [32, 0, 0.042],
    "6": [55, 45, 0.042],
    "7": [90, 69, 0.042],
    "8": [125, 45, 0.042],
    "9": [148, 0, 0.042],
    "10": [125, 315, 0.042],
    "11": [90, 291, 0.042],
    "12": [55, 315, 0.042],
    "13": [21, 91, 0.042],
    "14": [58, 90, 0.042],
    "15": [121, 90, 0.042],
    "16": [159, 89, 0.042],
    "17": [69, 180, 0.042],
    "18": [90, 212, 0.042],
    "19": [111, 180, 0.042],
    "20": [90, 148, 0.042],
    "21": [32, 180, 0.042],
    "22": [55, 225, 0.042],
    "23": [90, 249, 0.042],
    "24": [125, 225, 0.042],
    "25": [148, 180, 0.042],
    "26": [125, 135, 0.042],
    "27": [90, 111, 0.042],
    "28": [55, 135, 0.042],
    "29": [21, 269, 0.042],
    "30": [58, 270, 0.042],
    "31": [122, 270, 0.042],
    "32": [159, 271, 0.042],
}

def spherical_jn_series_threshold(x, table_lookup=True, epsilon=1e-2):
    r"""
    Convergence threshold of series :math:`f_{n}(x) = \sum_{q = 0}^{n} (2 q + 1) j_{q}^{2}(x)`.

    Parameters
    ----------
    x : float
    table_lookup : bool
        Use pre-computed table (with `epsilon=1e-2`) to accelerate the search.
    epsilon : float
        Only used when `table_lookup` is :py:obj:`False`.

    Returns
    -------
    n : int
        Value of `n` in :math:`f_{n}(x)` past which :math:`f_{n}(x) \ge 1 - \epsilon`.
    """
    if not (0 < epsilon < 1):
        raise ValueError("Parameter[epsilon] must lie in (0, 1).")

    if table_lookup is True:
        rel_path = pathlib.Path("data", "math", "special", "spherical_jn_series_threshold.csv")
        abs_path = pkg.resource_filename("imot_tools", str(rel_path))

        data = pd.read_csv(abs_path).sort_values(by="x")
        x = np.abs(x)
        idx = int(np.digitize(x, bins=data["x"].values))
        if idx == 0:  # Below smallest known x.
            n = data["n_threshold"].iloc[0]
        else:
            if idx == len(data):  # Above largest known x.
                ratio = data["n_threshold"].iloc[-1] / data["x"].iloc[-1]
            else:
                ratio = data["n_threshold"].iloc[idx - 1] / data["x"].iloc[idx - 1]
            n = int(np.ceil(ratio * x))

        return n
    else:

        def series(n, x):
            q = np.arange(n)
            _2q1 = 2 * q + 1
            _sph = special.spherical_jn(q, x) ** 2

            return np.sum(_2q1 * _sph)

        n_opt = int(0.95 * x)
        while True:
            n_opt += 1
            if 1 - series(n_opt, x) < epsilon:
                return n_opt

def pol2cart(r, colat, lon):
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)

    XYZ = np.array([x, y, z])

    return XYZ

def fibonacci(N, direction=None, FoV=None):
    r"""
    fibonnaci near-uniform sampling on the sphere.

    Parameters
    ----------
    N : int
        Order of the grid, i.e. there will be :math:`4 (N + 1)^{2}` points on the sphere.
    direction : :py:class:`~numpy.ndarray`
        (3,) vector around which the grid is centered.
        If :py:obj:`None`, then the grid covers the entire sphere.
    FoV : float
        Span of the grid [rad] centered at `direction`.
        This parameter is ignored if `direction` left unspecified.

    Returns
    -------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_px) sample points.
        `N_px == 4*(N+1)**2` if `direction` left unspecified.

    Examples
    --------
    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on the
    sphere:

    .. testsetup::

       import numpy as np

       from imot_tools.math.sphere.grid import fibonacci

    .. doctest::

       >>> N = 2
       >>> XYZ = fibonacci(N)

       >>> np.around(XYZ, 2)
       array([[ 0.23, -0.29,  0.04,  0.36, -0.65,  0.61, -0.2 , -0.37,  0.8 ,
               -0.81,  0.39,  0.28, -0.82,  0.95, -0.56, -0.13,  0.76, -1.  ,
                0.71, -0.05, -0.63,  0.97, -0.79,  0.21,  0.46, -0.87,  0.8 ,
               -0.33, -0.27,  0.68, -0.7 ,  0.36,  0.1 , -0.4 ,  0.4 , -0.16],
              [ 0.  , -0.27,  0.51, -0.47,  0.12,  0.39, -0.74,  0.72, -0.29,
               -0.34,  0.82, -0.89,  0.48,  0.21, -0.8 ,  0.98, -0.64, -0.04,
                0.71, -1.  ,  0.76, -0.13, -0.55,  0.93, -0.81,  0.28,  0.37,
               -0.78,  0.76, -0.36, -0.18,  0.56, -0.58,  0.31,  0.03, -0.17],
              [ 0.97,  0.92,  0.86,  0.81,  0.75,  0.69,  0.64,  0.58,  0.53,
                0.47,  0.42,  0.36,  0.31,  0.25,  0.19,  0.14,  0.08,  0.03,
               -0.03, -0.08, -0.14, -0.19, -0.25, -0.31, -0.36, -0.42, -0.47,
               -0.53, -0.58, -0.64, -0.69, -0.75, -0.81, -0.86, -0.92, -0.97]])

    Sampling a zonal function :math:`f(r): \mathbb{S}^{2} \to \mathbb{C}` of order :math:`N` on
    *part* of the sphere:

    .. doctest::

       >>> N = 2
       >>> direction = np.r_[1, 0, 0]
       >>> FoV = np.deg2rad(90)
       >>> XYZ = fibonacci(N, direction, FoV)

       >>> np.around(XYZ, 2)
       array([[ 0.8 ,  0.95,  0.76,  0.71,  0.97,  0.8 ],
              [-0.29,  0.21, -0.64,  0.71, -0.13,  0.37],
              [ 0.53,  0.25,  0.08, -0.03, -0.19, -0.47]])

    Notes
    -----
    The sample positions on the unit sphere are given (in radians) by [2]_:

    .. math::

       \cos(\theta_{q}) & = 1 - \frac{2 q + 1}{4 (N + 1)^{2}}, \qquad & q \in \{ 0, \ldots, 4 (N + 1)^{2} - 1 \},

       \phi_{q} & = \frac{4 \pi}{1 + \sqrt{5}} q, \qquad & q \in \{ 0, \ldots, 4 (N + 1)^{2} - 1 \}.


    .. [2] B. Rafaely, "Fundamentals of Spherical Array Processing", Springer 2015
    """
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if FoV is not None:
            if not (0 < np.rad2deg(FoV) < 360):
                raise ValueError("Parameter[FoV] must be in (0, 360) degrees.")
        else:
            raise ValueError("Parameter[FoV] must be specified if Parameter[direction] provided.")

    if N < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)

    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5))
    XYZ = np.stack(pol2cart(1, colat, lon), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]

    return XYZ

def nyquist_rate(XYZ, wl):
    """
    Order of imageable complex plane-waves by an instrument.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    wl : float
        Wavelength [m]

    Returns
    -------
    N : int
        Maximum order of complex plane waves that can be imaged by the instrument.
    """
    baseline = linalg.norm(XYZ[:, np.newaxis, :] - XYZ[:, :, np.newaxis], axis=0)

    N = spherical_jn_series_threshold((2 * np.pi / wl) * baseline.max())
    return N

def visualizer(file_path, output_dir = "viz_output", time_step=10.0e-3):
    """
    each frame is 100ms
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_signal, rate = librosa.load(file_path, mono=False)
    audio_signal = audio_signal.T
    N_antenna = audio_signal.shape[1]
    print("Number of mics (antennas):", N_antenna)
    assert N_antenna == 32, "For optimal visualization the test signal should contain 32 channels"

    # Generated tesselation for Robinson projection
    arg_lonticks = np.linspace(-180, 180, 5)

    # Filter field to lie in specified interval
    apgd_data, R = get_frame(audio_signal, rate, time_step=time_step)  # TODO: what is R?
    _, R_lat, R_lon = cart2eq(*R)
    R_lat_d, R_lon_d = wrapped_rad2deg(R_lat, R_lon)
    min_lon, max_lon = arg_lonticks.min(), arg_lonticks.max()
    mask_lon = (min_lon <= R_lon_d) & (R_lon_d <= max_lon)
    R_field = eq2cart(1, R_lat[mask_lon], R_lon[mask_lon])

    plt.rcParams['figure.figsize'] = [10, 5]

    apgd_T = np.transpose(apgd_data, (1, 0, 2))  # frame, bin, 242? TODO: what is 242

    centers_x = []
    centers_y = []
    for i, I_frame in enumerate(apgd_T):  # I_frame in bin * 242
        N_px = I_frame.shape[1]
        I_rgb = I_frame.reshape((3, 3, N_px)).sum(axis=1)
        I_rgb /= I_rgb.max()

        fig, ax, cluster_center = draw_map(I_rgb, R_field,
                                           lon_ticks=arg_lonticks,
                                           catalog=None,
                                           show_labels=True,
                                           show_axis=True)
        centers_x.append(cluster_center[0])
        centers_y.append(cluster_center[1])


        plt.savefig("{}/{}.jpg".format(output_dir, i))

    return centers_x, centers_y


def get_center(img_rgb, lat_degree, lon_degree):
    # convert to greyscale
    img_grey = 0.299 * img_rgb[0, :] + 0.587 * img_rgb[1, :] + 0.114 * img_rgb[2, :]
    min = img_grey.mean() * 0.1
    pos = np.zeros_like(lat_degree)
    for index, intensity in enumerate(img_grey):
        if intensity <= min:
            pos[index] += 1
    center_lat = pos @ lat_degree.T / sum(pos)  # y axis
    center_lon = pos @ lon_degree.T / sum(pos)  # x axis

    return center_lon, center_lat


def get_frame(audio_signal, rate, time_step=1):
    freq, bw = (skutil  # Center frequencies to form images
                .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
                .mean(axis=-1)), 50.0  # [Hz]

    xyz = get_xyz("eigenmike")  # get xyz coordinates of mic channels
    dev_xyz = np.array(xyz).T
    T_sti = 10.0e-3
    T_stationarity = 10 * T_sti  # Choose to have frame_rate = 10.
    N_freq = len(freq)

    wl_min = constants.speed_of_sound / (freq.max() + 500)
    sh_order = nyquist_rate(dev_xyz, wl_min) # Maximum order of complex plane waves that can be imaged by the instrument.
    R = fibonacci(sh_order)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(50))
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    N_px = R.shape[1]

    apgd_data = []  # np.zeros((N_freq, N_sample, N_px))
    for idx_freq in range(N_freq):
        wl = constants.speed_of_sound / freq[idx_freq]
        A = steering_operator(dev_xyz, R, wl)
        S = form_visibility(audio_signal, rate, freq[idx_freq], bw, T_sti, T_stationarity)
        N_sample = S.shape[0]

        apgd_gamma = 0.5
        apgd_per_band = np.zeros((N_sample, N_px))
        I_prev = np.zeros((N_px,))
        for idx_s in range(N_sample):

            # Normalize visibilities
            S_D, S_V = linalg.eigh(S[idx_s])
            if S_D.max() <= 0:
                S_D[:] = 0
            else:
                S_D = np.clip(S_D / S_D.max(), 0, None)
            S_norm = (S_V * S_D) @ S_V.conj().T

            I_apgd = solve(S_norm, A, gamma=apgd_gamma, x0=I_prev.copy(), verbosity='NONE')
            apgd_per_band[idx_s] = I_apgd['sol']
        apgd_data.append(apgd_per_band)
    apgd_data = np.stack(apgd_data, axis=0)

    return apgd_data, R


def extract_visibilities(_data, _rate, T, fc, bw, alpha):
    """
    Transform time-series to visibility matrices.

    Parameters
    ----------
    T : float
        Integration time [s].
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    alpha : float
        Shape parameter of the Tukey window, representing the fraction of
        the window inside the cosine tapered region. If zero, the Tukey
        window is equivalent to a rectangular window. If one, the Tukey
        window is equivalent to a Hann window.

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices (complex-valued).
    """
    N_stft_sample = int(_rate * T)
    if N_stft_sample == 0:
        raise ValueError('Not enough samples per time frame.')
    # print(f'Samples per STFT: {N_stft_sample}')

    N_sample = (_data.shape[0] // N_stft_sample) * N_stft_sample
    N_channel = _data.shape[1]
    stf_data = (skutil.view_as_blocks(_data[:N_sample], (N_stft_sample, N_channel))
                .squeeze(axis=1))  # (N_stf, N_stft_sample, N_channel)

    window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (N_stf, N_stft_sample, N_channel)
    N_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (N_stf, N_stft_sample, N_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * N_stft_sample / _rate)
    idx_end = int((fc + 0.5 * bw) * N_stft_sample / _rate)
    collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    # collapsed_spectrum = collapsed_spectrum[0,:]
    S = (collapsed_spectrum.reshape(N_stf, -1, 1).conj() *
         collapsed_spectrum.reshape(N_stf, 1, -1))
    return S


def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    '''
    Parameter
    ---------
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) antenna samples. (float)
    rate : int
        Sample rate [Hz]
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    T_sti : float
        Integration time [s]. (time-series)
    T_stationarity : float
        Integration time [s]. (visibility)

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices.

        # N_slot == number of audio frames in track

    Note
    ----
    Visibilities computed directly in the frequency domain.
    For some reason visibilities are computed correctly using
    `x.reshape(-1, 1).conj() @ x.reshape(1, -1)` and not the converse.
    Don't know why at the moment.
    '''
    S_sti = (extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0))

    N_sample, N_channel = data.shape
    N_sti_per_stationary_block = int(T_stationarity / T_sti)
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
         .squeeze(axis=(1, 2))
         .sum(axis=1))
    return S


def _deg2rad(coords_dict):
    """
    Take a dictionary with microphone array
    capsules and 3D polar coordinates to
    convert them from degrees to radians
    colatitude, azimuth, and radius (radius
    is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }


def _polar2cart(coords_dict, units=None):
    """
    Take a dictionary with microphone array
    capsules and polar coordinates and convert
    to cartesian
    Parameters:
        units: (str) indicating 'degrees' or 'radians'
    """
    if units == None or units != "degrees" and units != "radians":
        raise ValueError("you must specify units of 'degrees' or 'radians'")
    elif units == "degrees":
        coords_dict = _deg2rad(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }


def get_xyz(mic='ambeo'):
    mic_coords = None
    if mic == 'ambeo':
        mic_coords = _polar2cart(ambeovr_raw, units='degrees')
    elif mic == 'tetra':
        mic_coords = _polar2cart(tetra_raw, units='degrees')
    elif mic == 'eigenmike':
        mic_coords = _polar2cart(eigenmike_raw, units='degrees')

    if mic_coords == None:
        raise ValueError("you must specify a valid microphone: 'ambeo', 'tetra', 'eigenmike'")

    xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords]

    return xyz


def save_gif(pic_path):
    with Image() as wand:
        # Add new frames into sequance
        for file in os.listdir(pic_path):
            if file.endswith('.jpg'):
                wand.sequence.append(Image(filename=file))
        # Create progressive delay for each frame
        for cursor, frame in enumerate(wand.sequence):
            frame.delay = 10 * (cursor + 1)
        # Set layer type
        wand.type = 'optimize'
        wand.save(filename='animated.gif')
