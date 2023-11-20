"""
This script is the main script of acoustic spatial visualizer.
"""
import os

import librosa
import pandas as pd
import plotly.graph_objects as go
import soundfile as sf
from plotly.subplots import make_subplots

from plot_utils import *
from spatialization import spatializer
from utils import load_rir_pos
from visualization import visualizer

FS = 24000  # Target sample rate for spatialization
WS = 512  # window size for spatialization
TS = 256 * 21  # trim padding applied during the convolution process (constant independent of win_size or dur)


def IR_spargair(path_to_irs, IRS, win_size=WS, fs=FS):
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
        az = [90, 90 - 26.6, 90 - 63.4, 0, -26.6, -63.4, -90, -90]
        el = [0, 0, 0, 0, 0, 0, 0, 0]
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


def comp_plot(x, y, x_g, y_g, timestamp, azimuth, elevation, ir_times, out_folder):
    err_az = [a_i - b_i for a_i, b_i in zip(x, gt_az)]
    err_el = [a_i - b_i for a_i, b_i in zip(y, gt_el)]
    df = {}
    df['azimuth_gt'] = gt_az
    df['elevation_gt'] = gt_el
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
    fig.update_xaxes(title_text='azimuth', row=1, col=1)
    fig.update_yaxes(title_text='elevation', row=1, col=1)
    fig.update_yaxes(title_text='degree', row=1, col=2)
    fig.write_html(out_folder + "boxplot.html")
    fig.show()

    # plot azimuth and elevation change over time
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Azimuth over time", "Elevation over time"))
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.azimuth_est, mode='markers',
                             marker_size=abs(df['azimuth_error']) / abs(df['azimuth_error']).max() * 50, name='estimated'), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.azimuth_gt, mode='markers+lines',name='ground truth'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.timestamp, y=df.elevation_est, mode='markers',
                             marker_size=abs(df['elevation_error'])/ abs(df['azimuth_error']).max() * 50, name='estimated'),
                             row=1,col=2)
    fig.add_trace(go.Line(x=df.timestamp, y=df.elevation_gt, mode='markers+lines',name='ground truth'), row=1, col=2)
    for n, i in enumerate(ir_times):
        fig.add_vline(x= i * 1000, line_dash='dash', line_color='blue', row=1, col=1) # at what frame
        fig.add_scatter(x= [i * 1000],
                        y= [azimuth[n]],
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


    fig.write_html(out_folder + "time.html")
    fig.show()


if __name__ == "__main__":
    '''
    Spatializer
    '''
    # # Specify customization
    path_to_irs = '/Users/sivanding/database/spargair/em32/'
    # path_to_irs = './tau_srir/bomb_shelter.sofa'
    audio_name = "violin"
    audio_path = './monosound/{}.wav'.format(audio_name)
    output_path = './trajectories/left-to-right/{}/'.format(audio_name)
    os.makedirs(output_path, exist_ok=True)
    spatial_path = '{}_spatial.wav'.format(audio_name)
    # IRS = ['302', '212', '122', '032', '142', '252', '362']  # az = [90, 90 - 26.6, 90 - 63.4, 0, -26.6, -63.4, -90, -90]
    IRS = ['302', '212', '122', '032', '142', '252',
           '362']  # az = [90, 90 - 26.6, 90 - 63.4, 0, -26.6, -63.4, -90, -90]

    # Prepare IRs
    irs, elevation, azimuth = IR_spargair(path_to_irs, IRS)
    signal = get_mono(audio_path, duration=5)
    ir_times = np.linspace(0, len(signal) / FS, irs.shape[-1])  # uniform interpolation in time
    gt_az, gt_el, timestamp = get_gt(azimuth, elevation, ir_times, FS)

    # The real thing
    # spatialized_sig = spatializer(signal, irs, ir_times, target_sample_rate=FS)
    # sf.write(output_path + spatial_path, spatialized_sig, samplerate=FS)

    print("Spatialization completed.")

    '''
    Visualizer
    '''
    # Specify custominzation
    file_path = output_path + spatial_path  # spatialized track
    x, y = visualizer(file_path, output_dir=output_path + "viz_output", time_step=ir_times[1])

    '''
    Compare groundtruth and imager estimation in plots
    '''
    comp_plot(x, y, gt_az, gt_el, timestamp, azimuth, elevation, ir_times, output_path)

    print("Visualization completed.")
