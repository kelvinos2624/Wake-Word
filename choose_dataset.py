import os

dataset_folder = "/datasets/speech_commands"
primary_labels = ["yes", "no"]
audio_files = []
audio_labels = []

for label in primary_labels:
    folder_path = os.path.join(os.curdir, dataset_folder, label)
    print(folder_path)
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                audio_files.append(os.path.join(folder_path, file))
                audio_labels.append(label)

print(f"Number of audio files: {len(audio_files)}")