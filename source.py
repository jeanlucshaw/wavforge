import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wv
from scipy.interpolate import interp1d

# Parameters
SAMPLERATE = 44100
AMPLITUDE = np.iinfo(np.int16).max


class Wave:
    """
    Base class of the wavforge module.

    Attributes
    ----------
    time : 1D array
        Waveform cycle time coordinate.
    data : 1D array
        Waveform shape.
    freq : float
        Frequency at which to render waveform.

    Methods
    -------
    blend_with(other, mix)
        Blend Wave with other Wave in `mix` proportions.
    phase(shift)
        Move Wave phase by `shift` degrees.
    noise(mix)
        Blend Wave with digital noise in `mix` proportions.
    show()
        Plot Wave shape.
    to_audio()
        Standardize for conversion to wav file.
    to_math()
        Standardize for manipulation.
    to_wav(fname)
        Save Wave to wav file.
    """
    def __init__(self, time, data, freq):
        self.freq = freq
        self.time = time
        self.data = data

    def blend_with(self, other, mix):
        """
        Return self averaged with other waveform.

        Parameters
        ----------
        other : Wave object
            Wave with which to blend.
        mix : float
            Fraction of self to return.

        Returns
        -------
        Wave object
            New Wave object mixing the input shapes.
        """
        return blend(self, other, mix)

    def phase(self, shift):
        """
        Return Wave with phase moved by `shift`.

        Parameters
        ----------
        shift : float
            Phase shift amount (-360, 360 degrees).

        Returns
        -------
        Wave object
            New Wave object with shifted phase.
        """
        other = self._copy()
        steps = int(self.time.size * shift / 360)
        other.data = np.roll(other.data, steps)
        return other

    def noise(self, mix):
        """
        Return wave blended with digital noise.
        """
        other = self._copy()
        other.data = np.random.uniform(-1, 1, other.time.size)
        return blend(self, other, mix)

    def show(self):
        """
        Plot Wave shape and show.
        """
        plt.plot(self.time, self.data, 'ko-')
        plt.show()

    def to_audio(self):
        """
        Set data numerical format for writing to audio.
        """
        self.time, self.data = standardize(self.time,
                                           self.data,
                                           freq = self.freq)
        return self

    def to_math(self):
        """
        Set data numerical format for mathematical operations.
        """
        self.data = self.data.astype(np.float64) / abs(self.data).max()
        return self

    def to_wav(self, fname):
        """
        Write waveform to file.

        Putting the call here locks the samplerate to everything else.

        Parameters
        ----------
        fname : str
            Output file name.
        """
        wv.write(fname, SAMPLERATE, self.data)

    def _cat(self, other):
        data = np.hstack((self.data, other.data))
        time = np.hstack((self.time, self.time[-1] + other.time))
        return Wave(time, data, freq=self.freq)

    def _compatible(self, other):
        return self.freq == other.freq and self.time.size == other.time.size

    def _copy(self):
        return Wave(**self.__dict__)


class Saw(Wave):
    def __init__(self,
                 freq,
                 reverse=False,
                 super_saw_n=None,
                 super_saw_shift=None):
        self.freq = freq
        self.time = np.arange(int(SAMPLERATE / freq))
        self.data = np.linspace(-1, 1, time.size)[::-1]

        # Super saw if required
        if super_saw_n and super_saw_shift:
            print('supersawing')
            shift = int(super_saw_shift * time.size / super_saw_n)
            for i in range(super_saw_n):
                self.data = self.data + np.roll(self.data, -shift)
                self.data = self.to_math().data
        
        # Reverse if required
        if reverse:
            self.data = self.data[::-1]


class Sine(Wave):
    def __init__(self, freq):
        self.freq = freq
        self.time = np.arange(int(SAMPLERATE / freq))
        self.data = np.sin(2 * np.pi * time / time.max())


class Square(Wave):
    def __init__(self, freq, width=0.5):
        self.freq = freq
        self.time = np.arange(int(SAMPLERATE / freq))
        self.data = np.ones(time.size)
        self.data[int(width * time.size):] = -1


class Triangle(Wave):
    def __init__(self, freq, mix=0.5):
        self.freq = freq
        self.time = np.arange(int(SAMPLERATE / freq))

        # Make ramp and saw halves
        half_idx = int(time.size/2)
        ramp = 2 * np.linspace(-1, 1, time.size) + 1
        ramp[half_idx:] = 0
        saw = ramp[::-1]

        # Blend the two and normalize
        self.data = (1 - mix) * ramp + mix * saw
        self.data /= abs(self.data).max()

        # Shift 90 degrees for near zero beginning and end
        self.data = self.phase(90).data


# Functions
def blend(Wave_1, Wave_2, mix=0.5):
    """
    Return a blend of two waveforms.

    Parameters
    ----------
    Wave_1, Wave_2 : Wave objects
        The two objects to blend.
    mix : float
        Mixing proportion. (mix=0 means 100% Wave_1).

    Returns
    -------
    Wave object
        Result of the blend.
    """
    if Wave_1._compatible(Wave_2):
        data = (1 - mix) * Wave_1.to_math().data + mix * Wave_2.to_math().data
    else:
        raise ValueError('Wave not compatible with other.')

    return Wave(Wave_1.time, data, Wave_1.freq)


def morph(Wave_1, Wave_2, steps):
    """
    Produce wavetable going from Wave_1 to Wave_2.

    Parameters
    ----------
    Wave_1, Wave_2 : Wave object
        Extremes of the wavetable.
    steps : int
        Number of total steps in the wavetable.

    Returns
    -------
    list
        List of Wave objects.
    """
    wave_list = []
    mix_amounts = np.linspace(0, 1, steps)
    for mix in mix_amounts:
        wave_list.append(blend(Wave_1, Wave_2, mix=mix))
    return wave_list


def wave_list_to_slice_file(wave_list, fname):
    """
    Export wavetable to wav file.

    Slice files are wav files containing single cycle waveforms
    of same frequency back to back. They are usefull for example
    when used with the Elektron Octatrack sampler, as it can slice
    audio at regular intervals. When loaded into a sampler, slice
    files can be used as wavetables.

    Parameters
    ----------
    wave_list : list
        List of Wave objects to write to file.
    fname : str
        Path and name of output file.
    """
    # Concatenate waves
    output = wave_list[0]._copy().to_audio()
    for wave in wave_list[1:]:
        if wave._compatible(wave_list[0]):
            wave.to_audio()
            output = output._cat(wave)
        else:
            raise ValueError('Incompatible Wave in list.')

    # Write to file
    output.to_wav(fname)


def standardize(time, data, freq=220, fades=True):
    """
    Prepare data for writing to wav file.

    In order to be readable as a wav file, data must be represented
    as 16 bit integers (-32768, 32767). In order to be read at the
    right frequency, it must be gridded in time and represented by
    the correct number of samples. For example, a Wave object with
    frequency = 441 Hz will need,

    .. math::

       N = \\frac{\\text{Sample rate}}{\\text{frequency}}
         = \\frac{44100}{441} = 100

    one hundred samples to represent on waveform cycle. This function
    ensures these conditions are met and optionally adds fades to zero
    at the start and end of the cycle.

    Parameters
    ----------
    time, data : 1D array
        Array to convert to audio and its coordinate.
    freq : float
         Frequency of the output tone.
    fades : bool
        Set first and last values to zero.

    Returns
    -------
    grid : 1D array
        Gridded time coordinate.
    data : 1D array
        Normalized and gridded data.
    """
    NSAMPLES = int(SAMPLERATE / freq)
    grid = np.linspace(0, time.max(), NSAMPLES)
    data = interp1d(time, data, bounds_error=False).__call__(grid)

    # Set amplitude
    data = AMPLITUDE * data / data.max()

    # Fades
    if fades:
        data[0], data[-1] = 0, 0

    # Convert to integer
    data = data.astype(np.dtype('i2'))

    return grid, data


# Tests
time = np.arange(100)
freq = 440
sq = Square(time, freq)
saw = Saw(time, freq)

mlist = morph(saw, sq, 64)

for wav in mlist:
    plt.plot(wav.time, wav.data)
plt.show()

wave_table = list_to_slice_file(mlist, 'saw_square.wav')
