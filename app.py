from flask import Flask, render_template, request

import glob
import numpy as np
import pandas as pd
import pathlib
from random import randrange
import tensorflow as tf

from models.model import midi_to_notes, notes_to_midi, mse_with_positive_pressure

app = Flask(__name__)


music_model = tf.keras.models.load_model(
    "music_model.h5",
    custom_objects={"mse_with_positive_pressure": mse_with_positive_pressure},
)

# Number of MIDI files
data_dir = pathlib.Path("data/maestro-v3.0.0")
filenames = glob.glob(str(data_dir / "**/*.mid*"))


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
):
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


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        generate = request.form["submit-btn"]

        if generate == "Generate":
            filenames_index = randrange(len(filenames))
            predictions = generate_predictions(filenames[filenames_index], music_model)
            print("Generate")
            print(filenames_index)
            print(predictions)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(port=3000, debug=True)
