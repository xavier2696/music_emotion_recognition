import os
from pydub import AudioSegment

def split_audio_files(metadata):
    print('Splitting songs')
    cont = 0
    for file_name in os.listdir("../data/karaoke/mp3"):
        print(f"Progress: filename: {file_name}")
        song_name = f'../data/karaoke/mp3/{file_name}'
        song = AudioSegment.from_mp3(song_name)
        fragment_number = 1
        for pair in metadata[file_name.replace('.mp3', '')]:
            fragment = song[pair[0]:pair[1]]
            fragment_name = f"{file_name.replace('.mp3', '')}-{fragment_number}"
            fragment.set_frame_rate(44100)
            fragment.export(f"../data/karaoke/song_fragments/{fragment_name}", format="mp3")
            fragment_number += 1
        print(f"Progress: processed song# {cont} filename: {file_name}")
        cont += 1

metadata = {}
for file_name in os.listdir("../data/karaoke/metadata"):
    with open(f"../data/karaoke/metadata/{file_name}") as fp:
        time_annotations = []
        for cnt, line in enumerate(fp):
            if len(line) > 1 and line[0] == '[' and line[1].isdigit():
                begin_number = ""
                cont = 1
                while cont < len(line) and line[cont].isdigit():
                    begin_number += line[cont]
                    cont += 1
                end_number = ""
                duration = ""
                cont = len(line) - 1
                while cont > 0 and line[cont] != '<':
                    cont -= 1
                cont += 1
                while cont < len(line) and line[cont].isdigit():
                    end_number += line[cont]
                    cont += 1
                cont += 1
                while cont < len(line) and line[cont].isdigit():
                    duration += line[cont]
                    cont += 1
                begin = int(begin_number)
                end = begin + int(end_number) + int(duration)
                time_annotations.append((begin, end))

        metadata[file_name.replace('.txt', '')] = time_annotations

print(metadata)
split_audio_files(metadata)


