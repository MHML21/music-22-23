from pathlib import Path

import pandas as pd


__all__ = (
    'load_amg',
    'load_cleansed_ids',
    'load_masd_labels_cleansed',
    'load_midi_info_v2',
    'load_lastfm',
    'load_file_names',
)


def load_amg(dataset_path: str = 'lpd') -> pd.DataFrame:
    dataset_path = Path(dataset_path) / 'amg'
    labels = (
        ('Blues', 0),
        ('Country', 1),
        ('Electronic', 2),
        ('Folk', 3),
        ('International', 4),
        ('Jazz', 5),
        ('Latin', 6),
        ('New-Age', 7),
        ('Pop_Rock', 8),
        ('Rap', 9),
        ('Reggae', 10),
        ('RnB', 11),
        ('Vocal', 12),
    )
    data = pd.DataFrame(columns=['msd_id', 'amg_num'])
    for (label, amg_num) in labels:
        genre = pd.read_csv(dataset_path / f'id_list_{label}.txt', names=('file_id', ))
        genre['amg_num'] = amg_num
        data = pd.concat([data, genre])
    return data


def load_cleansed_ids(dataset_path: str = 'lpd') -> pd.DataFrame:
    return pd.read_csv(Path(dataset_path) / 'cleansed_ids.txt', sep='    ', engine='python', names=('file_id', 'msd_id'))


def load_midi_info_v2(dataset_path: str = 'lpd') -> pd.DataFrame:
    data = pd.read_json(Path(dataset_path) / 'midi_info_v2.json').transpose().reset_index().rename(columns={'index': 'file_id'})
    return pd.merge(data, load_cleansed_ids(dataset_path), how='outer')


def load_masd_labels_cleansed(dataset_path: str = 'lpd') -> pd.DataFrame:
    return pd.read_csv(Path(dataset_path) / 'cleansed_ids.txt', sep='    ', engine='python', names=('file_id', 'msd_id'))


def load_lastfm(genre: str, dataset_path: str = 'lpd') -> pd.DataFrame:
    return pd.read_csv(Path(dataset_path) / 'lastfm' / f'id_list_{genre}.txt', names=('msd_id', ))


def load_file_names(dataset_path: str = 'lpd') -> pd.DataFrame:
    cleansed_directory = Path(dataset_path) / 'lpd_5' / 'lpd_5_cleansed'
    paths = tuple(cleansed_directory.rglob('*.npz'))
    names = tuple(path.stem for path in paths)
    return pd.DataFrame({
        'file_id': names,
        'file_path': paths
    })
