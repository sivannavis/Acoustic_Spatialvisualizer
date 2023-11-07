import numpy as np
import scipy


def spatializer(sig, irs, ir_times, win_size=512, target_sample_rate=24000, trim_samps=256 * 21):
    """
    This funciton uses ctf_ltv_direct to create spatialized signal from mono with some normalization and trimming.
    ---------
    Params:
        sig
        irs
        ir_times
        win_size
        target_sample_rate
        trim_samps
    Returns:
        output_signal
    """
    output_signal = ctf_ltv_direct(sig, irs, ir_times, target_sample_rate, win_size)
    output_signal /= output_signal.max()  # apply max normalization (prevents clipping/distortion)
    output_signal = output_signal[trim_samps:, :]  # trim front padding segment
    print("Length of output_signal:", output_signal.shape)
    print("Sampling rate:", target_sample_rate)

    return output_signal


def stft_ham(insig, winsize=256, fftsize=512, hopsize=128):
    nb_dim = len(np.shape(insig))
    lSig = int(np.shape(insig)[0])
    nCHin = int(np.shape(insig)[1]) if nb_dim > 1 else 1
    x = np.arange(0, winsize)
    nBins = int(fftsize / 2 + 1)
    nWindows = int(np.ceil(lSig / (2. * hopsize)))
    nFrames = int(2 * nWindows + 1)

    winvec = np.zeros((len(x), nCHin))
    for i in range(nCHin):
        winvec[:, i] = np.sin(x * (np.pi / winsize)) ** 2

    frontpad = winsize - hopsize
    backpad = nFrames * hopsize - lSig

    if nb_dim > 1:
        insig_pad = np.pad(insig, ((frontpad, backpad), (0, 0)), 'constant')
        spectrum = np.zeros((nBins, nFrames, nCHin), dtype='complex')
    else:
        insig_pad = np.pad(insig, ((frontpad, backpad)), 'constant')
        spectrum = np.zeros((nBins, nFrames), dtype='complex')

    idx = 0
    nf = 0
    if nb_dim > 1:
        # adding spectrum frame by frame
        while nf <= nFrames - 1:
            insig_win = np.multiply(winvec, insig_pad[idx + np.arange(0, winsize), :])
            inspec = scipy.fft.fft(insig_win, n=fftsize, norm='backward', axis=0)
            # inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
            inspec = inspec[:nBins, :]
            spectrum[:, nf, :] = inspec
            idx += hopsize
            nf += 1
    else:
        while nf <= nFrames - 1:
            insig_win = np.multiply(winvec[:, 0], insig_pad[idx + np.arange(0, winsize)])
            inspec = scipy.fft.fft(insig_win, n=fftsize, norm='backward', axis=0)
            # inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
            inspec = inspec[:nBins]
            spectrum[:, nf] = inspec
            idx += hopsize
            nf += 1

    return spectrum


def ctf_ltv_direct(sig, irs, ir_times, fs, win_size):
    convsig = []
    win_size = int(win_size)
    hop_size = int(win_size / 2)
    fft_size = win_size * 2
    nBins = int(fft_size / 2) + 1

    # IRs
    ir_shape = np.shape(irs)
    sig_shape = np.shape(sig)

    lIr = ir_shape[0]

    if len(ir_shape) == 2:
        nIrs = ir_shape[1]
        nCHir = 1
    elif len(ir_shape) == 3:
        nIrs = ir_shape[2]
        nCHir = ir_shape[1]

    if nIrs != len(ir_times):
        return ValueError('Bad ir times')

    # number of STFT frames for the IRs (half-window hopsize)

    nIrWindows = int(np.ceil(lIr / win_size))
    nIrFrames = 2 * nIrWindows + 1
    # number of STFT frames for the signal (half-window hopsize)
    lSig = sig_shape[0]
    nSigWindows = np.ceil(lSig / win_size)
    nSigFrames = 2 * nSigWindows + 1

    # quantize the timestamps of each IR to multiples of STFT frames (hopsizes)
    tStamps = np.round((ir_times * fs + hop_size) / hop_size)

    # create the two linear interpolator tracks, for the pairs of IRs between timestamps
    nIntFrames = int(tStamps[-1])
    Gint = np.zeros((nIntFrames, nIrs))
    for ni in range(nIrs - 1):
        tpts = np.arange(tStamps[ni], tStamps[ni + 1] + 1, dtype=int) - 1
        ntpts = len(tpts)
        ntpts_ratio = np.arange(0, ntpts) / (ntpts - 1)
        Gint[tpts, ni] = 1 - ntpts_ratio
        Gint[tpts, ni + 1] = ntpts_ratio

    # compute spectra of irs
    if nCHir == 1:
        irspec = np.zeros((nBins, nIrFrames, nIrs), dtype=complex)
    else:
        # IR stft of the first location
        temp_spec = stft_ham(irs[:, :, 0], winsize=win_size, fftsize=2 * win_size, hopsize=win_size // 2)
        irspec = np.zeros((nBins, np.shape(temp_spec)[1], nCHir, nIrs), dtype=complex)

    for ni in range(nIrs):
        if nCHir == 1:
            irspec[:, :, ni] = stft_ham(irs[:, ni], winsize=win_size, fftsize=2 * win_size, hopsize=win_size // 2)
        else:
            spec = stft_ham(irs[:, :, ni], winsize=win_size, fftsize=2 * win_size, hopsize=win_size // 2)
            irspec[:, :, :, ni] = spec  # np.transpose(spec, (0, 2, 1))

    # compute input signal spectra
    sigspec = stft_ham(sig, winsize=win_size, fftsize=2 * win_size, hopsize=win_size // 2)
    # initialize interpolated time-variant ctf
    Gbuf = np.zeros((nIrFrames, nIrs))
    if nCHir == 1:
        ctf_ltv = np.zeros((nBins, nIrFrames), dtype=complex)
    else:
        ctf_ltv = np.zeros((nBins, nIrFrames, nCHir), dtype=complex)

    S = np.zeros((nBins, nIrFrames), dtype=complex)

    # processing loop
    idx = 0
    nf = 0
    inspec_pad = sigspec
    nFrames = int(np.min([np.shape(inspec_pad)[1], nIntFrames]))

    convsig = np.zeros((win_size // 2 + nFrames * win_size // 2 + fft_size - win_size, nCHir))

    while nf <= nFrames - 1:
        # compute interpolated ctf
        Gbuf[1:, :] = Gbuf[:-1, :]  # TODO
        Gbuf[0, :] = Gint[nf, :]
        if nCHir == 1:
            for nif in range(nIrFrames):
                ctf_ltv[:, nif] = np.matmul(irspec[:, nif, :], Gbuf[nif, :].astype(complex))
        else:
            for nch in range(nCHir):
                for nif in range(nIrFrames):
                    ctf_ltv[:, nif, nch] = np.matmul(irspec[:, nif, nch, :], Gbuf[nif, :].astype(complex))
        inspec_nf = inspec_pad[:, nf]
        S[:, 1:nIrFrames] = S[:, :nIrFrames - 1]
        S[:, 0] = inspec_nf

        repS = np.tile(np.expand_dims(S, axis=2), [1, 1, nCHir])
        convspec_nf = np.squeeze(np.sum(repS * ctf_ltv, axis=1))
        first_dim = np.shape(convspec_nf)[0]
        convspec_nf = np.vstack((convspec_nf, np.conj(convspec_nf[np.arange(first_dim - 1, 1, -1) - 1, :])))
        convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, norm='forward',
                                            axis=0))  ## get rid of the imaginary numerical error remain
        # convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, axis=0))
        # overlap-add synthesis
        convsig[idx + np.arange(0, fft_size), :] += convsig_nf  # TODO
        # advance sample pointer
        idx += hop_size
        nf += 1
    print("convolved signal shape", convsig.shape)
    convsig = convsig[(win_size):(nFrames * win_size) // 2, :]

    return convsig
