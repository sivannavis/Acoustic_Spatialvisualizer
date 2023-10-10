import math
import os

import scipy.constants as constants
import scipy.io.wavfile as wavfile
import scipy.signal.windows as windows
import skimage.util as skutil
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import librosa

from apgd import *
from plot_utils import *
from utils import *


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
    N_sti_per_stationary_block = int(T_stationarity / T_sti) + 1
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
         .squeeze(axis=(1, 2))
         .sum(axis=1))
    return S


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


def generate_frames(frame):
    I_frame = apgd_T[frame]
    N_px = I_frame.shape[1]
    I_rgb = I_frame.reshape((3, 3, N_px)).sum(axis=1)
    I_rgb /= I_rgb.max()
    fig, ax = draw_map(I_rgb, R_field,
                       lon_ticks=arg_lonticks,
                       catalog=None,
                       show_labels=True,
                       show_axis=True)
    return fig,


if __name__ == "__main__":
    os.chdir("/Users/sivanding/Codebase/DeepWaveTorch/")

    file_path = "/Users/sivanding/Codebase/seld-dcase2023/spatialization/violin_metu_1000.wav"  # spatialized track
    audio_signal, rate = librosa.load(file_path, sr=None, mono=False)
    audio_signal = librosa.resample(audio_signal, orig_sr=rate, target_sr=24000)
    audio_signal = audio_signal.T
    rate = 24000
    # audio_signal = np.pad(audio_signal, ((0, audio_signal.shape[0]),(0,0)), 'constant')
    N_antenna = audio_signal.shape[1]
    print("Number of mics (antennas):", N_antenna)
    assert N_antenna == 32, "For optimal visualization the test signal should contain 32 channels"

    freq, bw = (skutil  # Center frequencies to form images
                .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
                .mean(axis=-1)), 50.0  # [Hz]

    idx_s = 10  # For the sake of an example, we will choose the 10th audio frame (you can choose whichever frame you want)

    idx_freq = 0  # choose 0th frequency
    T_sti = 10e-3
    T_stationarity = 10 * T_sti  # Choose to have frame_rate = 10
    S = form_visibility(audio_signal, rate, freq[idx_freq], bw, T_sti, T_stationarity)
    print(S.shape)

    xyz = get_xyz("eigenmike")  # get xyz coordinates of mic channels
    dev_xyz = np.array(xyz).T
    T_sti = 10.0e-3
    rate = 24000
    T_stationarity = 10 * T_sti  # Choose to have frame_rate = 10.
    N_freq = len(freq)

    wl_min = constants.speed_of_sound / (freq.max() + 500)
    R = np.load("tracks/eigenmike_grid.npy")
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(50))
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    N_px = R.shape[1]
    N_sample = S.shape[0]

    apgd_data = np.zeros((N_freq, N_sample, 242))
    for idx_freq in range(N_freq):
        wl = constants.speed_of_sound / freq[idx_freq]
        A = steering_operator(dev_xyz, R, wl)

        apgd_gamma = 0.5
        apgd_lambda_ = np.zeros((N_sample,))
        apgd_N_iter = np.zeros((N_sample,), dtype=int)
        apgd_tts = np.zeros((N_sample,))
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
            apgd_data[idx_freq][idx_s] = I_apgd['sol']

    # Generated tesselation for Robinson projection
    arg_lonticks = np.linspace(-180, 180, 5)
    # Filter field to lie in specified interval
    _, R_lat, R_lon = cart2eq(*R)
    _, R_lon_d = wrapped_rad2deg(R_lat, R_lon)
    min_lon, max_lon = arg_lonticks.min(), arg_lonticks.max()
    mask_lon = (min_lon <= R_lon_d) & (R_lon_d <= max_lon)
    R_field = eq2cart(1, R_lat[mask_lon], R_lon[mask_lon])

    plt.rcParams['figure.figsize'] = [10, 5]
    apgd_T = np.transpose(apgd_data, (1, 0, 2)) # frame, bin, 242? TODO: what is 242
    N_max_frames = 50  # maximum number of frames to display (each frame is 100ms)
    for i, I_frame in enumerate(apgd_T[:N_max_frames]): # I_frame in bin * 242
        N_px = I_frame.shape[1]
        I_rgb = I_frame.reshape((3, 3, N_px)).sum(axis=1)
        # print(I_rgb, R_field)
        I_rgb /= I_rgb.max()
        fig, ax = draw_map(I_rgb, R_field,
                           lon_ticks=arg_lonticks,
                           catalog=None,
                           show_labels=True,
                           show_axis=True)

        # get the ground truth for chosen time frame

        # plt.show()
        plt.savefig("/Users/sivanding/Codebase/seld-dcase2023/spatialization/{}.jpg".format(i))

    apgd_T = np.transpose(apgd_data, (1, 0, 2))
    animation = FuncAnimation(plt.figure(), generate_frames, frames=len(apgd_T), interval=200)
    HTML(animation.to_jshtml())
