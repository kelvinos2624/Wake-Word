import tensorflow as tf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob

def extract_mfcc(file_path, sr = 16000, n_mfcc = 13):
    audio, _ = librosa.load(file_path, sr = sr)
    mfccs = librosa.feature.mfcc(y = audio, sr = sr, n_mfcc = n_mfcc)
    return mfccs.T

def pad_or_truncate_data(mfcc, max_len = 40):
    if len(mfcc) > max_len:
        return mfcc[:max_len]
    elif len(mfcc) < max_len:
        pad_width = max_len - len(mfcc)
        return np.pad(mfcc, ((0, pad_width), (0, 0)), mode = "constant")
    return mfcc

labels = ["yes", "no", "stop", "go", "_background_noise_"]

dataset_folder = os.curdir + "/datasets/speech_commands"
file_paths = []
for label in os.listdir(dataset_folder):
    if label not in labels:
        continue
    folder_path = os.path.join(dataset_folder, label)
    if os.path.isdir(folder_path):
        for file in glob.glob(os.path.join(folder_path, "*.wav")):
            file_paths.append(file)

print(file_paths)

X = np.array([pad_or_truncate_data(extract_mfcc(file_path)) for file_path in file_paths])
X = X.reshape(X.shape[0], 40, 13, 1)

label_map = {"yes":0, "no":1, "stop":2, "go":3, "_background_noise_":4}
y = np.array([label_map[os.path.basename(os.path.dirname(file_path))] for file_path in file_paths])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (40, 13, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation = "softmax")
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_val, y_val))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        yield [input_value]

converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()

with open("voice_command_cnn_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path = "voice_command_cnn_model_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Size of the tflite model: {os.path.getsize('voice_command_cnn_model_quantized.tflite')} bytes")