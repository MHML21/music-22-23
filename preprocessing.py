from dataset import *

import numpy as np
import pandas as pd
import pypianoroll

# pandas printing settings
pd.set_option('max_colwidth', None)
pd.options.display.max_columns = None
pd.options.display.width = 0


__all__ = (
    'get_file_paths',
    'clean_multitrack',
    'load_tracks_from_genre',
)


def get_file_paths(genre: str) -> tuple[str]:
    ids = load_cleansed_ids()
    files = load_file_names()
    rock = load_lastfm(genre)
    df = pd.merge(ids, files, on='file_id', how='outer')
    df = pd.merge(df, rock, on='msd_id', how='inner')
    df = df.drop_duplicates(subset='file_id', keep='first')
    return tuple(df['file_path'])


def clean_multitrack(multitrack: pypianoroll.Multitrack, track: str) -> np.ndarray:
    track_ids = {
        'Drums': 0,
        'Piano': 1,
        'Guitar': 2,
        'Bass': 3,
        'Strings': 4,
    }
    multitrack.binarize()
    try:
        multitrack.trim()
    except ValueError:
        pass
    return multitrack.tracks[track_ids[track]].pianoroll[:, 24:-20]


def load_tracks_from_genre(genre: str, track_type: str, verbose: bool = True) -> np.ndarray:
    paths = get_file_paths(genre)

    multitracks = []
    for i, path in enumerate(paths):
        if verbose and (i % 100 == 0):
            print(f'Loading track {i}...')
        multitracks.append(clean_multitrack(pypianoroll.load(path), track_type))
    return np.concatenate(multitracks, axis=0)


if __name__ == '__main__':
    combined = load_tracks_from_genre('rock', 'Strings')
