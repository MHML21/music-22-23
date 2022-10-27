from pathlib import Path

import pandas as pd


__all__ = ('load_amg', 'load_cleansed_ids', 'load_masd_labels_cleansed', 'load_midi_info_v2', 'load_all')


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


def load_all(dataset_path: str = 'lpd', reload: bool = False) -> pd.DataFrame:
    preloaded_path = Path(dataset_path) / 'preloaded.csv'
    if reload or not preloaded_path.exists():
        masd_labels = load_masd_labels_cleansed(dataset_path)
        cleansed_ids = load_cleansed_ids(dataset_path)
        data = pd.merge(masd_labels, cleansed_ids, on='msd_id', how='outer')
        midi_info = load_midi_info_v2(dataset_path)
        data = pd.merge(data, midi_info, on='file_id', how='outer')
        amg = load_amg(dataset_path)
        data = pd.merge(data, amg, on='msd_id', how='outer')
        data.to_csv(preloaded_path)
        return data
    else:
        return pd.read_csv(preloaded_path)


if __name__ == '__main__':
    ids = load_cleansed_ids()
    print(ids.columns)
    midi_info = load_midi_info_v2()
    print(midi_info.columns)
    data = pd.merge(midi_info, ids, how='outer')
    print(data.head())
