import os
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

class AudioDataSet:

    def __init__(self, load_training_data=False, s_frequency=44100, threshold=5, directory='data',
                 f_output_file_name='data/features.csv'):
        self.s_frequency = s_frequency
        self.threshold = threshold
        self.directory = directory
        self.annotations = pd.DataFrame()
        self.song_features = pd.DataFrame()
        self.dataset = pd.DataFrame()
        if load_training_data:
            if os.path.exists(f'{self.directory}/annotations.csv'):
                print("Loaded annotations")
                self.annotations = pd.read_csv(f'{self.directory}/annotations.csv')
            else:
                self.annotations = self.load_audio_anotations()
            if os.path.exists(f'{f_output_file_name}'):
                print("Loaded features")
                self.song_features = pd.read_csv(f'{f_output_file_name}')
            else:
                self.song_features = self.load_audio_files()
            if os.path.exists(f'{self.directory}/dataset.csv'):
                print("Loaded dataset")
                self.dataset = pd.read_csv(f'{self.directory}/dataset.csv')
            else:
                self.dataset = self.combine_features_anotations()

            #divide into training and testing
            self.x = self.dataset.drop('song_name', axis=1)
            self.x = self.x.drop('valence', axis=1)
            self.x = self.x.drop('arousal', axis=1)
            self.y = self.dataset[['valence', 'arousal']]
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,
                                                                                    self.y,
                                                                                    test_size=0.2,
                                                                                    random_state=26)
        else:
            if os.path.exists(f'{f_output_file_name}'):
                self.song_features = pd.read_csv(f'{f_output_file_name}')
            else:
                self.song_features = self.load_audio_files(output_file_name=f_output_file_name)
            self.x = self.song_features.drop('song_name', axis=1)


    def load_audio_anotations(self):
        print("Loading audio annotations")
        v_anotations = pd.read_csv('data/annotations/valence.csv')
        v_anotations = v_anotations.dropna(axis='columns')
        a_anotations = pd.read_csv('data/annotations/arousal.csv')
        a_anotations = a_anotations.dropna(axis='columns')
        # get valence anotations for fragments
        song_names = []
        valence_values = []
        for index, row in v_anotations.iterrows():
            start_time = 15000
            columns_grouped_total = (self.threshold * 1000)/500
            column_sum = 0
            current_columns_grouped = 0
            current_fragment = 1
            #while start_time <= 44500:
            while f'sample_{start_time}ms' in row and row[f'sample_{start_time}ms'] != '':
                column_sum += row[f'sample_{start_time}ms']
                current_columns_grouped += 1
                start_time += 500
                if current_columns_grouped == columns_grouped_total:
                    song_names.append(f'{int(row.song_id)}-{current_fragment}')
                    valence_values.append(column_sum/columns_grouped_total) # average, can be changed for max/min
                    column_sum = 0
                    current_columns_grouped = 0
                    current_fragment += 1
        # get arousal anotations for fragments
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
        valence_fragments = pd.DataFrame({'song_name': song_names, 'valence': valence_values, 'arousal': arousal_values})
        valence_fragments.to_csv('data/annotations.csv',  index=False)
        return valence_fragments

    def clean_name(self, row):
        return row['name'].replace('\'', '').replace('_0', '')

    def load_audio_files(self, input_file_name='data/features_final.csv', output_file_name='data/features.csv'):
        if not os.path.exists(input_file_name):
            self.generate_open_smile_csv()
            self.transform_open_smile_csv_to_df_csv()
        song_features = pd.read_csv(input_file_name)
        song_features['song_name'] = song_features.apply(lambda row: self.clean_name(row), axis=1)
        song_features = song_features.drop(['name', 'frameTime', 'emotion'], axis=1)
        song_features.to_csv(output_file_name)
        return song_features

    def generate_open_smile_csv(output_file_name='data/features_os.csv'):
        print('Creating audio features')
        count = 0
        for file_name in os.listdir('/mnt/data/xavier/audio_wav'):
            os.system(
                f"./SMILExtract -C emobase.conf -I /mnt/data/xavier/audio_wav/{file_name} -O {output_file_name} -1 {file_name.replace('.wav', '')}")
            print(f"Progress: processed song# {count} filename: {file_name}")
            count += 1

    def transform_open_smile_csv_to_df_csv(input_file_name='data/features_os.csv',
                                           output_file_name='data/features_final.csv'):
        with open(input_file_name) as csv_file:
            o_file = open(f'{output_file_name}', 'w', newline='')
            with o_file:
                writer = csv.writer(o_file)
                header_columns = []
                data_start_index = -1
                for cnt, line in enumerate(csv_file):
                    if '@attribute' in line:
                        header_columns.append(line.split(' ')[1])
                    if '@data' in line:
                        header_string = ",".join(header_columns)
                        print(header_string)
                        writer.writerow(header_string.split(','))
                        data_start_index = cnt + 1
                    if cnt > data_start_index and line != '' and data_start_index > 0:
                        writer.writerow(line.split(','))

    def combine_features_anotations(self):
        print('Creating dataset by combining features and annotations')
        dataset = self.song_features.merge(self.annotations, how='inner', on='song_name')
        dataset.to_csv(f"{self.directory}/dataset.csv",  index=False)
        return dataset



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

