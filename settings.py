GENRE = 'classic'
TRACK = 'Strings'

NOTES = 84
RESOLUTION = 24
BEAT = RESOLUTION * 4

HYPERPARAMS = {
    #"pos weight": 80/4,             # ratio of negatives to positives (80 non-playing notes, 4 playing-notes)
    "input length": 8 * BEAT,       # 2 measures
    "output length": 1,             # 1 for now, eventually we will try to output a whole measure (4 * BEAT)
    "vocab size": NOTES,            # 84 notes (0: not playing, 1: playing)
    "epochs": 100,
    "batch size": 128,
    "learning rate": 0.001,
    "validation ratio": 0.1,
    #"embedding size": 8,
}