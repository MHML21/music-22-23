GENRE = 'classic'
TRACK = 'Strings'

NOTES = 84
RESOLUTION = 24
MEASURE = RESOLUTION * 4

HYPERPARAMS = {
    "pos weight": 80/4,          # ratio of negatives to positives (80 non-playing notes, 4 playing-notes)
    "input length": 2 * MEASURE, # 2 measures
    "output length": MEASURE,    # 1 measure
    "vocab size": NOTES,         # 84 notes (0: not playing, 1: playing)
    "epochs": 150,
    "batch size": 128,
    "learning rate": 0.001,
    "validation ratio": 0.05,
    "embedding size": 8,
}