import numpy as np
import pandas as pd
import tensorflow as tf

from utils.audio import maestro_filenames, midi_to_notes
from utils.loss import mse_with_positive_pressure


class MusicGenerationRNN:
    def __init__(
        self,
        filenames: str,
        seed: int = 42,
        sampling_rate: int = 16000,
        num_files: int = 5,
        seq_length: int = 25,
        vocab_size: int = 128,
        batch_size: int = 64,
        learning_rate: float = 0.005,
        epochs: int = 50,
    ):
        self.filenames = filenames  # Filenames for MIDI files to process
        self.seed = seed  # Random seed value
        self.sampling_rate = sampling_rate  # Sampling rate for audio playback
        self.num_files = num_files  # Number of MIDI files to process
        self.seq_length = seq_length  # Length of audio sequence
        self.vocab_size = vocab_size  # Size of vocabulary (defaults to 128, number of pitches supported by `pretty_midi`)
        self.batch_size = batch_size  # Size of batches for training
        self.learning_rate = learning_rate  # Learning rate of neural network
        self.epochs = epochs  # Number of epochs during training

        self.train_ds = None

        # Set random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def create_training_set(self):
        # Normalize note pitch
        def scale_pitch(x):
            x = x / [self.vocab_size, 1.0, 1.0]
            return x

        # Split the labels
        def split_labels(sequences):
            inputs = sequences[:-1]
            labels_dense = sequences[-1]
            labels = {key: labels_dense[i] for i, key in enumerate(key_order)}

            return scale_pitch(inputs), labels

        def create_sequences(
            dataset: tf.data.Dataset, seq_length: int
        ) -> tf.data.Dataset:
            """Creates the training dataset so that the input features are a sequence of notes and the label is the next note

            Args:
                dataset (tf.data.Dataset): Input dataset of notes
                seq_length (int): _description_

            Returns:
                tf.data.Dataset: Output dataset
            """

            seq_length = seq_length + 1

            # Take 1 extra for the labels
            windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

            # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
            flatten = lambda x: x.batch(seq_length, drop_remainder=True)
            sequences = windows.flat_map(flatten)

            return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

        # Generate the training dataset by extracting notes from the MIDI files
        all_notes = []
        for f in self.filenames[: self.num_files]:
            notes = midi_to_notes(f)
            all_notes.append(notes)

        all_notes = pd.concat(all_notes)

        n_notes = len(all_notes)

        # Convert the data to tf.data.Dataset format
        key_order = ["pitch", "step", "duration"]
        train_notes = np.stack([all_notes[key] for key in key_order], axis=1)
        notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

        seq_ds = create_sequences(notes_ds, self.seq_length)

        # Batch the dataset and configure it for better performance
        buffer_size = n_notes - self.seq_length  # the number of items in the dataset
        train_ds = (
            seq_ds.shuffle(buffer_size)
            .batch(self.batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        self.train_ds = train_ds

    def create_model(self) -> tf.keras.Model:
        input_shape = (self.seq_length, 3)

        inputs = tf.keras.Input(input_shape)
        x = tf.keras.layers.LSTM(128)(inputs)

        outputs = {
            "pitch": tf.keras.layers.Dense(128, name="pitch")(x),
            "step": tf.keras.layers.Dense(1, name="step")(x),
            "duration": tf.keras.layers.Dense(1, name="duration")(x),
        }

        model = tf.keras.Model(inputs, outputs)

        loss = {
            "pitch": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "step": mse_with_positive_pressure,
            "duration": mse_with_positive_pressure,
        }

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            loss=loss,
            loss_weights={
                "pitch": 0.05,
                "step": 1.0,
                "duration": 1.0,
            },
            optimizer=optimizer,
        )

        return model

    def train(self, model: tf.keras.Model):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath="./training_checkpoints/ckpt_{epoch}",
                save_weights_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, verbose=1, restore_best_weights=True
            ),
        ]

        model.fit(
            self.train_ds,
            epochs=self.epochs,
            callbacks=callbacks,
        )

    def evaluate(self, model: tf.keras.Model):
        model.evaluate(self.train_ds, return_dict=True)

    def save(self, model: tf.keras.Model, filename: str):
        model.save(filename)


filenames = maestro_filenames()

model_wrapper = MusicGenerationRNN(filenames)
# Create the training set
model_wrapper.create_training_set()
# Create and compile the model
model = model_wrapper.create_model()
# Train the model using the training set
model_wrapper.train(model)
# Evaluate the model
model_wrapper.evaluate(model)
# Save the model
model_wrapper.save(model, "music_model.h5")
