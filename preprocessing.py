from dataset import *

import numpy as np
import pandas as pd
import pypianoroll
import sys

# pandas printing settings
pd.set_option('max_colwidth', None)
pd.options.display.max_columns = None
pd.options.display.width = 0
np.set_printoptions(threshold=sys.maxsize)
#np.set_printoptions(linewidth=np.inf)


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
    try:
        multitrack.trim()
    except ValueError:
        pass
    return multitrack.tracks[track_ids[track]].pianoroll[:, 24:-20]


def load_tracks_from_genre(genre: str, track_type: str, verbose: bool = True) -> np.ndarray:
    paths = get_file_paths(genre)

    multitracks = []
    for i, path in enumerate(paths):
        if(i == 129):
            break
        if verbose and (i % 10 == 0):
            print(f'Loading track {i}...')
        multitracks.append(one_multitrack_parse(path, genre, track_type))
    return multitracks

def one_multitrack_parse(path: str, genre: str, track_type: str):
    multi = pypianoroll.load(path)
    cleansed = clean_multitrack(multi, track_type).T
    mask = cleansed != 0
    axis = 0
    not_played_val = -1
    note_vals = np.where(mask.any(axis=axis), mask.argmax(axis=axis), not_played_val)
    return np.trim_zeros(note_vals + 1)
if __name__ == '__main__':
    one_dim_tracks = load_tracks_from_genre('rock', 'Guitar')
