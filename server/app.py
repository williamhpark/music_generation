from flask import Flask, render_template, request

import numpy as np
import os
import pandas as pd
import tensorflow as tf

from midi2audio import FluidSynth
from random import randrange

from utils.audio import maestro_filenames, midi_to_notes, notes_to_midi
from utils.loss import mse_with_positive_pressure

app = Flask(__name__)

# Load the trained mdoel
music_model = tf.keras.models.load_model(
    "./models/music_model.h5",
    custom_objects={"mse_with_positive_pressure": mse_with_positive_pressure},
)

# Files from the Maestro dataset used to train and test the model
filenames = maestro_filenames()


def predict_next_note(
    notes: np.ndarray, model: tf.keras.Model, temperature: float = 1.0
) -> int:
    """Generates information for the next predicted note

    Args:
        notes (np.ndarray): Initial sequence of notes
        model (tf.keras.Model): Model used to make the prediction
        temperature (float, optional): Used to control the randomness of notes generated. Defaults to 1.0.

    Returns:
        int: _description_
    """

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions["pitch"]
    step = predictions["step"]
    duration = predictions["duration"]

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def generate_predictions(
    file: pd.DataFrame,
    model: tf.keras.Model,
    seq_length: int = 25,
    vocab_size: int = 128,
    temperature: float = 2.0,
    num_predictions: int = 120,
) -> pd.DataFrame:
    """_summary_

    Args:
        file (pd.DataFrame): _description_
        model (tf.keras.Model): _description_
        seq_length (int, optional): _description_. Defaults to 25.
        vocab_size (int, optional): _description_. Defaults to 128.
        temperature (float, optional): _description_. Defaults to 2.0.
        num_predictions (int, optional): _description_. Defaults to 120.

    Returns:
        pd.DataFrame: _description_
    """

    key_order = ["pitch", "step", "duration"]

    raw_notes = midi_to_notes(file)

    sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

    # The initial sequence of notes; pitch is normalized similar to training sequences
    input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])

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
        generated_notes, columns=(*key_order, "start", "end")
    )

    return generated_notes


def synthesize_midi(
    input_filename: str, output_filename: str, sample_rate: int = 16000
):
    fs = FluidSynth(sample_rate=sample_rate)

    current_dir = os.getcwd()
    fs.midi_to_audio(
        "{current_dir}/server/static/{input_filename}.mid".format(
            current_dir=current_dir, input_filename=input_filename
        ),
        "{current_dir}/server/static/{output_filename}.wav".format(
            current_dir=current_dir, output_filename=output_filename
        ),
    )


@app.route("/", methods=["GET", "POST"])
def predict():
    output_filename = None
    
    if request.method == "POST":

        generate = request.form["submit-btn"]

        if generate == "Generate":
            # Generate predictions based off of a random MIDI sample in the dataset
            filenames_index = randrange(len(filenames))
            generated_notes = generate_predictions(
                filenames[filenames_index], music_model
            )

            # Convert the predictions to a MIDI file and export as a MIDI
            generated_filename = "midi_{filenames_index}".format(
                filenames_index=filenames_index
            )
            notes_to_midi(
                generated_notes,
                generated_filename,
            )

            # Synthesize the generated MIDI file and export as a WAV file
            output_filename = generated_filename + "_exported"
            synthesize_midi(generated_filename, output_filename)

    return render_template("index.html", output_filename=output_filename)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
