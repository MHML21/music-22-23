# encoding chords
ROOTS = {
    "C": 0,
    "C#": 1, "D♭": 1,
    "D": 2,
    "D#": 3, "E♭": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "G♭": 6,
    "G": 7,
    "G#": 8, "A♭": 8,
    "A": 9,
    "A#": 10, "B♭": 10,
    "B": 11
}
QUALITIES = {
    "": 0,
    "m": 1,
    "aug": 2,
    "dim": 3,
    "5": 4,
    "6": 5,
    "m6": 6,
    "7": 7, "maj7": 7,
    "7-5": 8,
    "7-9": 9,
    "7sus4": 10,
    "m7": 11,
    "m7-5": 12,
    "9": 13, "maj9": 13, "add9": 13,
    "m9": 14, "madd9": 14,
    "sus4": 15,
    "mM7": 16,
}
NQUALS = 17              # number of non-overlapping qualities
VOCAB_SIZE = NQUALS*12   # chordnum = NCATS*root + qualities, therefore from C=0 to BmM7=203



# model hyperparamets
HYPERPARAMS = {
    "input length": 8,
    "epochs": 150,
    "batch size": 128,
    "learning rate": 0.001,
    "validation ratio": 0.05,
    "embedding size": 8,
}