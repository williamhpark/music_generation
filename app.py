from flask import Flask, render_template, request

import numpy as np
import pandas as pd
import tensorflow as tf

import glob
import pathlib

app = Flask(__name__)

# music_model = tf.keras.models.load_model("music_model.h5")


# def midi_to_notes(midi_file: str) -> pd.DataFrame:
#     """Extracts the notes from a MIDI file and transfer the information to a dataframe

#     We will use three variables to represent a note when training the model: pitch, step and duration.
#     The pitch is the perceptual quality of the sound as a MIDI note number.
#     The step is the time elapsed from the previous note or start of the track.
#     The duration is how long the note will be playing in seconds and is the difference between the note end and note start times.

#     Args:
#         midi_file (str): Path to input MIDI file

#     Returns:
#         pd.DataFrame: Output dataframe
#     """

#     pm = pretty_midi.PrettyMIDI(midi_file)
#     instrument = pm.instruments[0]
#     notes = collections.defaultdict(list)

#     # Sort the notes by start time
#     sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
#     prev_start = sorted_notes[0].start

#     for note in sorted_notes:
#         start = note.start
#         end = note.end
#         notes["pitch"].append(note.pitch)
#         notes["start"].append(start)
#         notes["end"].append(end)
#         notes["step"].append(start - prev_start)
#         notes["duration"].append(end - start)
#         prev_start = start

#     return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


# def notes_to_midi(
#     notes: pd.DataFrame,
#     out_file: str,
#     instrument_name: str,
#     velocity: int = 100,  # note loudness
# ) -> pretty_midi.PrettyMIDI:
#     """Generates a MIDI file from a dataframe of notes

#     Args:
#         notes (pd.DataFrame): _description_
#         out_file (str): _description_
#         instrument_name (str): _description_
#         velocity (int, optional): _description_. Defaults to 100.

#     Returns:
#         pretty_midi.PrettyMIDI: _description_
#     """

#     pm = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(
#         program=pretty_midi.instrument_name_to_program(instrument_name)
#     )

#     prev_start = 0
#     for _, note in notes.iterrows():
#         start = float(prev_start + note["step"])
#         end = float(start + note["duration"])
#         note = pretty_midi.Note(
#             velocity=velocity,
#             pitch=int(note["pitch"]),
#             start=start,
#             end=end,
#         )
#         instrument.notes.append(note)
#         prev_start = start

#     pm.instruments.append(instrument)
#     pm.write(out_file)

#     return pm


# def predict_next_note(
#     notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0
# ) -> int:
#     """Generates information for the next predicted note

#     Args:
#         notes (np.ndarray): Initial sequence of notes
#         model (tf.keras.Model): Model used to make the prediction
#         temperature (float, optional): Used to control the randomness of notes generated. Defaults to 1.0.

#     Returns:
#         int: _description_
#     """

#     assert temperature > 0

#     # Add batch dimension
#     inputs = tf.expand_dims(notes, 0)

#     predictions = model.predict(inputs)
#     pitch_logits = predictions["pitch"]
#     step = predictions["step"]
#     duration = predictions["duration"]

#     pitch_logits /= temperature
#     pitch = tf.random.categorical(pitch_logits, num_samples=1)
#     pitch = tf.squeeze(pitch, axis=-1)
#     duration = tf.squeeze(duration, axis=-1)
#     step = tf.squeeze(step, axis=-1)

#     # `step` and `duration` values should be non-negative
#     step = tf.maximum(0, step)
#     duration = tf.maximum(0, duration)

#     return int(pitch), float(step), float(duration)

def generate_predictions(file:pd.DataFrame, model:tf.keras.Model, seq_length:int=25, vocab_size:int=128, temperature:float=2.0, num_prediction:int=120):
    key_order = ['pitch', 'step', 'duration']
    
    raw_notes = midi_to_notes(file)
    
    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training
    # sequences
    input_notes = (
        sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
    pitch, step, duration = predict_next_note(input_notes, model, temperature)
    start = prev_start + step
    end = start + duration
    input_note = (pitch, step, duration)
    generated_notes.append((*input_note, start, end))
    input_notes = np.delete(input_notes, 0, axis=0)
    input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
    prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    return generated_notes


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        generate = request.form["submit-btn"]

        if generate == 'Generate':
            generate_predictions()
            print('Generate')

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
