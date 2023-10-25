"""
This script is the main script of acoustic spatial visualizer.
"""
import librosa
import numpy as np
import soundfile as sf

from spatialization import spatializer
from visualization import visualizer


FS = 24000 # Target sample rate for spatialization
WS = 512 # window size for spatialization
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

    return irs


def get_mono(audio_path, duration=5, fs=FS, trim_samps=TS):
    """
    Params
        duration: mixture duration in seconds

    Return
        signal: padded, trimmed signal
    """


    trim_dur = (trim_samps) / fs  # get duration in seconds for the padding section
    signal, sr = librosa.load(audio_path, mono=True)
    signal = librosa.resample(signal, orig_sr=sr, target_sr=fs)
    signal = signal[:FS * duration + trim_samps]  # account for removed samples from trim_samps
    # IR times: how you want to move the sound source over its event span as if a discrete estimation
    duration += trim_dur
    return signal



if __name__ == "__main__":
    # '''
    # Spatializer
    # '''
    # # Specify customization
    # path_to_irs = '/Users/sivanding/database/spargair/em32/'
    # audio_path = 'violin.wav'
    # IRS = [302, 412, 522, 632, 542, 452, 362]  # azimuth: 90, 90+26.6, 90+63.4, 180, -90-63.4, -90-26.6, -90
    #
    # # Prepare IRs
    # irs = IR_spargair(path_to_irs, IRS)
    # signal = get_mono(audio_path)
    # ir_times = np.linspace(0, len(signal)/FS, irs.shape[-1]) # uniform interpolation in time
    #
    # # The real thing
    # spatialized_sig = spatializer(signal, irs, ir_times, target_sample_rate=FS)
    # sf.write('violin_metu_test.wav', spatialized_sig, samplerate=FS)
    #
    # print("Spatialization completed.")

    '''
    Visualizer
    '''
    # Specify custominzation
    file_path = "violin_metu_test.wav"  # spatialized track
    visualizer(file_path)
    print("Visualization completed.")