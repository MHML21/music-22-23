import pretty_midi
import numpy as np
from pychord import Chord
from model import loadmodel
from settings import VOCAB_SIZE
from cleandata import int_to_chord, chord_to_int




def generate_midi(chords, filename="midis/quickplay", octave=4):
    """
    Takes a list of pychord.Chord and creates a midi
    """
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    for n, chord in enumerate(chords):
        for note_name in chord.components_with_pitch(root_pitch=octave):
            note_number = pretty_midi.note_name_to_number(note_name)
            note = pretty_midi.Note(velocity=100, pitch=note_number, start=n, end=n + 1)
            piano.notes.append(note)
    midi_data.instruments.append(piano)
    midi_data.write(f'{filename}.mid')



def generate_chords(chords, length, filename="midis/quickplay", octave=4):
    model = loadmodel()
    chordnums = np.reshape([chord_to_int(c) for c in chords], (1, 8, 1))
    # generate length number of chords
    for i in range(length):
        pred = model.predict(chordnums[:,-8:,:], verbose=0)         # one hot encode of predicted chord
        chord = np.random.choice(np.arange(VOCAB_SIZE), p=pred[0])  # pick a note using pred as probabability
        chordnums = np.append(chordnums, chord.reshape((1,1,1)), axis=1)
    chord_seq = [Chord(int_to_chord(c)) for c in chordnums[0,:,0]]
    generate_midi(chord_seq, filename, octave)
    print(f"Generated {filename}.mid")
    return chord_seq



if __name__ == '__main__':
    generate_chords(["C", "Bm7", "Em", "G", "C", "Bm7", "Em", "G"], 40)