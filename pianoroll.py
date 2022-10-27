from pathlib import Path

from midi2audio import FluidSynth
import pypianoroll
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


__all__ = ('pianoroll_to_audio', )


def pianoroll_to_audio(
        pianoroll: pypianoroll.Multitrack,
        name: str = 'output',
        filetype: str = 'wav',
        font: str = None,
) -> None:
    """
    Note that FluidSynth (https://github.com/FluidSynth/fluidsynth/wiki/Download) must be installed.

    I did this on Ubuntu, so it should be easy to install in a Google Colab, though I don't know how easy it is to
    install on Windows or macOS.
    """

    pianoroll.write(f'{name}.midi')
    fs = FluidSynth(font) if font else FluidSynth()
    fs.midi_to_audio(f'{name}.midi', f'{name}.{filetype}')


def plot(pianoroll: pypianoroll.Multitrack | pypianoroll.StandardTrack) -> None:
    """To plot correctly the line `matplotlib.use('TkAgg')` must be included."""
    pianoroll.plot()
    plt.show()


if __name__ == '__main__':
    cleansed_directory = Path('lpd/lpd_5/lpd_5_cleansed')
    track_files = tuple(cleansed_directory.rglob('*.npz'))

    print('Loading piano roll...')
    pianoroll = pypianoroll.load(track_files[0])

    pianoroll_to_audio(pianoroll, font='/usr/share/sounds/sf3/default-GM.sf3')
