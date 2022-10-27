import csv
from settings import ROOTS, QUALITIES, NQUALS, HYPERPARAMS



def chord_to_int(chord):
    """
    Converts a chord to a unique integer
    Ignores slash chords
    chordnum = n_qualities*root + category
    """
    chord = chord.split("/", 1)[0]
    if chord[1:2] in ("♭","#"):
        root = ROOTS[chord[:2]] # the root is the first 2 letters e.g. A#, D♭
        cat = QUALITIES[chord[2:]]
    else:
        root = ROOTS[chord[0]]  # the root is just the letter e.g. B, E
        cat = QUALITIES[chord[1:]]
    return NQUALS*root + cat



def int_to_chord(chordnum):
    """
    Converts chordnum to respective chord
    """
    root_i = chordnum // NQUALS
    qual_i = chordnum % NQUALS
    root = next(r for r, num in ROOTS.items() if num == root_i)     # get root name from root number
    qual = next(q for q, num in QUALITIES.items() if num == qual_i) # get quality name from quality number
    return root + qual



def split_song_chords(row):
    """
    Takes all the chords from a song, converts the chords into integers,
    then splits them into multiple rows each with length = input_length + 1
    """
    length = HYPERPARAMS["input length"] + 1
    clean_chords = [chord_to_int(c) for c in row if "N.C" not in c]
    return [clean_chords[i:i+length] for i in range(0,len(clean_chords)//length * length, length)]



def cleandata(fileout, filein):
    """
    Generates a list of chord integers and saves to fileout.csv
    """
    with open(f'{fileout}.csv', 'w', encoding='UTF8', newline='') as fout:
        writer = csv.writer(fout)
        
        # read raw data
        with open(f'{filein}.csv', 'r', encoding='UTF8') as fin:
            reader = csv.reader(fin)
            [writer.writerows(split_song_chords(row)) for row in reader]



if __name__ == '__main__':
    # takes raw scraped chords and prepares them to feed into neural network
    cleandata("data/cleandata", "data/rawdata")