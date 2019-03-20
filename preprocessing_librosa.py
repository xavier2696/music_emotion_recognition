import librosa
import os
import pandas as pd

class AudioDataSet:

    def __init__(self, load_csv = False, s_frequency = 44100, threshold = 2):
        self.s_frequency = s_frequency
        self.threshold = threshold
        if load_csv:
            self.annotations = pd.read_csv('data/annotations.csv')
        else:
            self.annotations = self.load_audio_anotations()
            self.song_data = self.load_audio_files()
            #self.get_audio_features(self.song_data['53-1'])

    def load_audio_anotations(self):
        v_anotations = pd.read_csv('data/annotations/valence.csv')
        v_anotations = v_anotations.dropna(axis='columns')
        a_anotations = pd.read_csv('data/annotations/arousal.csv')
        a_anotations = a_anotations.dropna(axis='columns')
        #get valence anotations for fragments
        song_names = []
        valence_values = []
        for index, row in v_anotations.iterrows():
            start_time = 15000
            columns_grouped_total = (self.threshold * 1000)/500
            column_sum = 0
            current_columns_grouped = 0
            current_fragment = 1
            while start_time <= 44500:
                column_sum += row[f'sample_{start_time}ms']
                current_columns_grouped += 1
                start_time += 500
                if current_columns_grouped == columns_grouped_total:
                    song_names.append(f'{int(row.song_id)}-{current_fragment}')
                    valence_values.append(column_sum/columns_grouped_total) #average, can be changed for max/min
                    column_sum = 0
                    current_columns_grouped = 0
                    current_fragment += 1
        #get arousal anotations for fragments
        arousal_values = []
        for index, row in a_anotations.iterrows():
            start_time = 15000
            columns_grouped_total = (self.threshold * 1000) / 500
            column_sum = 0
            current_columns_grouped = 0
            while start_time <= 44500:
                column_sum += row[f'sample_{start_time}ms']
                current_columns_grouped += 1
                start_time += 500
                if current_columns_grouped == columns_grouped_total:
                    arousal_values.append(column_sum / columns_grouped_total)  # average, can be changed for max/min
                    column_sum = 0
                    current_columns_grouped = 0
        valence_fragments = pd.DataFrame({'song_id': song_names, 'valence': valence_values, 'arousal': arousal_values})
        valence_fragments.to_csv('data/annotations.csv')
        return valence_fragments


    def load_audio_files(self):
        songs = {}
        for file_name in os.listdir('data/MEMD_audio/'):
            song_name = f'data/MEMD_audio/{file_name}'
            song, sr = librosa.load(song_name, sr=self.s_frequency)
            start = 15 * self.s_frequency  # annotations start from second 15
            fragment_number = 1
            step = self.threshold * self.s_frequency
            while (start + step < len(song)):
                songs[f"{file_name.replace('.mp3', '')}-{fragment_number}"] = song[start:(start + step)]
                start += step
                fragment_number += 1
            #if len(songs) > 16:
                #break
        return songs

    def get_audio_features(self, song):
        #zero crossing rate
        #gets zero crossing rate for each second
        zcr = librosa.feature.zero_crossing_rate(song, frame_length=self.s_frequency + 1, hop_length=self.s_frequency + 1)
        print(zcr)


a = AudioDataSet()

#get duration of song in milliseconds
#from mutagen.mp3 import MP3
#audio = MP3("MEMD_audio/2.mp3")
#print(audio.info.length * 44100)

#show amplitude/time graph
#import matplotlib.pyplot as plt
#import librosa.display
#plt.figure(figsize=(14, 5))
#librosa.display.waveplot(x, sr=sr)
#plt.show()

#show spectogram
#X = librosa.stft(x)
#Xdb = librosa.amplitude_to_db(abs(X))
#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#plt.colorbar()
#plt.show()

