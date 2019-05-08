from pydub import AudioSegment
import os

fragment_size = 5000
music_directories = ["./data/audio_pt1", './data/audio_pt2', './data/audio_pt3', './data/audio_pt4', './data/audio_pt5']
count = 0
for directory in music_directories:
    print('Splitting songs in', directory)
    for file_name in os.listdir(directory):
        print(f"Progress: filename: {file_name}")
        song_name = f'{directory}/{file_name}'
        song = AudioSegment.from_mp3(song_name)
        fragment_number = 1
        start = 0
        while (start + fragment_size) < len(song):
            fragment = song[start:(start + fragment_size)]
            fragment_name = f"{file_name.replace('.mp3', '')}-{fragment_number}"
            fragment.set_frame_rate(44100)
            fragment.export(f"/mnt/data/xavier/audio_wav/{fragment_name}.wav", format="wav")
            fragment_number += 1
            start += fragment_size
        print(f"Progress: processed song# {count} filename: {file_name}")
        count += 1

