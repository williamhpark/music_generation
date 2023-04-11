import collections
import glob
import numpy as np
import os
import pandas as pd
import pathlib
import pretty_midi
import shutil

# Returns a list of the MIDI files from the Maestro dataset
def maestro_filenames() -> list[str]:
    data_dir = pathlib.Path("data/maestro-v3.0.0")
    filenames = glob.glob(str(data_dir / "**/*.mid*"))

    return filenames


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """Extracts the notes from a MIDI file and transfers the information to a dataframe

    We will use three variables to represent a note when training the model: pitch, step and duration.
    The pitch is the perceptual quality of the sound as a MIDI note number.
    The step is the time elapsed from the previous note or start of the track.
    The duration is how long the note will be playing in seconds and is the difference between the note end and note start times.

    Args:
        midi_file (str): Path to input MIDI file

    Returns:
        pd.DataFrame: Output dataframe
    """

    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes["pitch"].append(note.pitch)
        notes["start"].append(start)
        notes["end"].append(end)
        notes["step"].append(start - prev_start)
        notes["duration"].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    instrument_name: str = "Acoustic Grand Piano",
    velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
    """Generates a MIDI file from a dataframe of notes

    Args:
        notes (pd.DataFrame): _description_
        out_file (str): _description_
        instrument_name (str): _description_
        velocity (int, optional): _description_. Defaults to 100.

    Returns:
        pretty_midi.PrettyMIDI: _description_
    """

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(instrument_name)
    )

    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note["step"])
        end = float(start + note["duration"])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note["pitch"]),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write("{out_file}.mid".format(out_file=out_file))

    # Move the generated file to the server/static folder
    current_dir = os.getcwd()
    src_path = "{current_dir}/{out_file}.mid".format(
        current_dir=current_dir, out_file=out_file
    )
    dest_path = "{current_dir}/server/static/{out_file}.mid".format(
        current_dir=current_dir, out_file=out_file
    )
    shutil.move(src_path, dest_path)
