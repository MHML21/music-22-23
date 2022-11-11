# Overview
MusicML is a machine learning model that attempts to synthesize music using an LSTM and ultimately a variational autoencoder.

# Model architecture
## Version 1
Version 1 of the model uses an LSTM that takes in a sequence of notes and outputs a boolean column vector of the notes being played. It compares the output to the next notes being played using a binary cross entropy loss (multi-label classification), allowing it to output multiple notes simultaneously. 

## Version 2
Version 2 of the model pre-processes the input data to obtain cleaner outputs. It also returns a sequence of notes instead of a vector of the note being played. 

## Version 3
Version 3 uses a custom loss function to hopefully generate more natural sounding music. 

## Version 4
Version 4 uses a variational auto encoder instead of an LSTM.

# Todo List
- [x] Load data
- [x] Create a basic LSTM with no pre-processing on input data (V1)
- [x] Process data to only allow one note to play at a time
- [ ] Modify LSTM to work with the prosessed data and output one note at a time
- [ ] Modify LSTM to output a sequence of single notes (V2)
- [ ] Create custom loss function that better captures musicality (V3)
- [ ] Move on from using an LSTM to a variational auto encoder (V4)