"""
This script is the main script of acoustic spatial visualizer.
"""
import librosa
import soundfile as sf

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
            path_to_files = path_to_irs + str(ir) + '/'
            chans = []
            for m in range(1, 33):
                # default sample rate for spargair is 48kHz
                x, sr = librosa.load(path_to_files + f'IR{m:05d}.wav', sr=48000, mono=True)
                x = librosa.resample(x, orig_sr=sr, target_sr=fs)
                chans.append(x)
            irs.append(chans)
        irs = np.transpose(np.array(irs), (2, 1, 0))  # samples * channel * locations
        az = [90, 90 + 26.6, 90 + 63.4, 180, -90 - 63.4, -90 - 26.6, -90]
        el = [0, 0, 0, 0, 0, 0, 0]
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


if __name__ == "__main__":
    '''
    Spatializer
    '''
    # Specify customization
    # path_to_irs = '/Users/sivanding/database/spargair/em32/'
    path_to_irs = './tau_srir/bomb_shelter.sofa'
    audio_path = 'violin.wav'
    output_path = '{}_spatial_3.wav'.format(audio_path[:-4])
    IRS = [302, 412, 522, 632, 542, 452, 362]  # azimuth: 90, 90+26.6, 90+63.4, 180, -90-63.4, -90-26.6, -90

    # Prepare IRs
    irs, elevation, azimuth = IR_spargair(path_to_irs, IRS)
    signal = get_mono(audio_path, duration=5)
    ir_times = np.linspace(0, len(signal) / FS, irs.shape[-1])  # uniform interpolation in time

    # The real thing
    spatialized_sig = spatializer(signal, irs, ir_times, target_sample_rate=FS)
    sf.write(output_path, spatialized_sig, samplerate=FS)

    print("Spatialization completed.")

    '''
    Visualizer
    '''
    # Specify custominzation
    file_path = output_path  # spatialized track
    x, y = visualizer(file_path, output_dir="viz_output_2")

    # plot groundtruth and estimated trajectory
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-', label='estimated')
    # show groundtruth
    x_g = azimuth
    y_g = elevation
    ax.plot(x_g, y_g, 'ro-', label='ground truth')
    plt.title("Trajectory of spatialized audio")
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig('trajectory.jpg')
    plt.show()
    print("Visualization completed.")
