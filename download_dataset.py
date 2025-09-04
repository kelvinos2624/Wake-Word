import tensorflow as tf

dataset_path = tf.keras.utils.get_file('speech_commands', 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz', extract=True)

dataset_folder = dataset_path[:-7]

print(f"Dataset extracted to: {dataset_folder}")