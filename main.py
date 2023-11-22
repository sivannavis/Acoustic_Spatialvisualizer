"""
This script is the main script of acoustic spatial visualizer.
"""
import os

import librosa
import pandas as pd
import soundfile as sf


from plot_utils import *
from spatialization import spatializer
from utils import load_rir_pos
from visualization import visualizer

FS = 24000  # Target sample rate for spatialization
WS = 512  # window size for spatialization
TS = 256 * 21  # trim padding applied during the convolution process (constant independent of win_size or dur)


def get_IR(path_to_irs, IRS, win_size=WS, fs=FS):
    """
    Stack IRs of selected locations from spargair dataset.
    ---
    Params
        path_to_irs
        IRS
        win_size
        fs

    Return
    """
    if 'spargair' in path_to_irs:
        # to make the last RIR play for the same duration we need to append a dummy one
        IRS.append(IRS[-1])
        irs = []

        for ir in IRS:
            path_to_files = path_to_irs + ir + '/'
            chans = []
            for m in range(1, 33):
                # default sample rate for spargair is 48kHz
                x, sr = librosa.load(path_to_files + f'IR{m:05d}.wav', sr=48000, mono=True)
                x = librosa.resample(x, orig_sr=sr, target_sr=fs)
                chans.append(x)
            irs.append(chans)
        irs = np.transpose(np.array(irs), (2, 1, 0))  # samples * channel * locations
        x = [ (3 - int(i[0])) * 0.5 for i in IRS]
        y = [(3 - int(i[1])) * 0.5 for i in IRS]
        z = [(-2 + int(i[2])) * 0.3 for i in IRS]
        r, el, az = zip(*[cart2eq(*i) for i in list(zip(x, y, z))])
        el, az = wrapped_rad2deg(el, az)
    elif 'tau' in path_to_irs:  # tau boom shelter has 6480 four channel RIR with shape 7200 / 24kHz
        rirs, pos = load_rir_pos(path_to_irs)
        # take first 10 points
        irs = np.transpose(np.array(rirs[IRS, ...]), (2, 1, 0))
        r, el, az = cart2eq(*pos[IRS, ...].T)

    else:
        raise NotImplementedError

    return irs, el, az


def get_mono(audio_path, duration=None, fs=FS, trim_samps=TS):
    """
    Params
        duration: mixture duration in seconds

    Return
        signal: padded, trimmed signal
    """

    trim_dur = (trim_samps) / fs  # get duration in seconds for the padding section
    signal, sr = librosa.load(audio_path, mono=True)
    signal = librosa.resample(signal, orig_sr=sr, target_sr=fs)
    if duration is None:
        length = len(signal)
    else:
        length = duration * fs
    signal = signal[: length + trim_samps]  # account for removed samples from trim_samps
    # IR times: how you want to move the sound source over its event span as if a discrete estimation
    return signal


def get_gt(azimuth, elevation, ir_times, fs):
    timestamps = (FS * ir_times).astype(int)
    timestamps[-1] -= 1
    high_el = np.full(signal.shape, np.nan)
    high_az = np.full(signal.shape, np.nan)
    for n, t in enumerate(timestamps):
        high_el[t] = elevation[n]
        high_az[t] = azimuth[n]
    high_az = pd.Series(high_az).interpolate()
    high_el = pd.Series(high_el).interpolate()
    timestamps = np.linspace(100 / 2, len(signal) / FS * 1000 - 100 / 2, 50) * FS / 1000
    gt_el = []
    gt_az = []
    for index in timestamps.astype(int):
        gt_el.append(high_el[index])
        gt_az.append(high_az[index])

    return gt_az, gt_el, timestamps * 1000 / FS


if __name__ == "__main__":
    spatialize = False
    visualize_map = False
    visualize_plot = True
    '''
    Trajectory configurations
    '''
    trajectories = ['left_to_right_mid',
                    'left_to_right_down',
                    'left_to_right_over',
                    'up_to_down_left',
                    'up_to_down_mid',
                    'up_to_down_right',
                    'left_up_to_right_down']
    IRS = {}
    trajectory = trajectories[6]
    # left to right on middle plane
    IRS[trajectories[0]] = ['302', '202', '212', '112', '122', '022', '032', '042','142','152', '252','262', '362']
    # left to right down
    IRS[trajectories[1]] = ['300', '200', '210', '110', '120', '020', '030', '040','140','150', '250','260', '360']
    # left to right over
    IRS[trajectories[2]] = ['304', '204', '214', '114', '124', '024', '034', '044','144','154', '254','264', '364']
    # up to down left
    IRS[trajectories[3]] = ['304', '313','303', '302', '301', '311','300']
    # up to down middle
    IRS[trajectories[4]] = ['034', '133','033', '032', '031', '131','030']
    # up to down right
    IRS[trajectories[5]] = ['364', '353','363', '362', '361', '351','360']
    # left up to right down
    IRS[trajectories[6]] = ['304', '204', '213', '113', '123', '022', '032', '042','141','151', '251','260', '360']

    # Specify customization
    path_to_irs = '/Users/sivanding/database/spargair/em32/'
    # path_to_irs = './tau_srir/bomb_shelter.sofa'
    audio_name = "white"
    audio_path = f'./monosound/{audio_name}.wav'
    output_path = f'./trajectories/{trajectory}/{audio_name}/'
    os.makedirs(output_path, exist_ok=True)
    spatial_path = f'{audio_name}_spatial.wav'

    '''
    Spatializer
    '''

    # Prepare IRs
    irs, elevation, azimuth = get_IR(path_to_irs, IRS[trajectory])
    signal = get_mono(audio_path, duration=5)
    ir_times = np.linspace(0, len(signal) / FS, irs.shape[-1])  # uniform interpolation in time
    gt_az, gt_el, timestamp = get_gt(azimuth, elevation, ir_times, FS)

    # The real thing
    if spatialize:
        spatialized_sig = spatializer(signal, irs, ir_times, target_sample_rate=FS)
        sf.write(output_path + spatial_path, spatialized_sig, samplerate=FS)
        print("Spatialization completed.")

    '''
    Visualizer
    '''
    # Specify custominzation
    file_path = output_path + spatial_path  # spatialized track
    x, y = visualizer(file_path, visualize_map, output_dir=output_path + "viz_output", time_step=ir_times[1])

    '''
    Compare groundtruth and imager estimation in plots
    '''
    if visualize_plot:
        comp_plot(x, y, gt_az, gt_el, timestamp, azimuth, elevation, ir_times, output_path, trajectory)

    print("Visualization completed.")
