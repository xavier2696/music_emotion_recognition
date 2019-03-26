import pandas as pd
from preprocessing_librosa import AudioDataSet

d1 = AudioDataSet(load_training_data=True, directory='data/audio_pt1',
                  f_output_file_name='features1.csv')
d2 = AudioDataSet(load_training_data=True, directory='data/audio_pt2',
                  f_output_file_name='features2.csv')
d3 = AudioDataSet(load_training_data=True, directory='data/audio_pt3',
                  f_output_file_name='features3.csv')
d4 = AudioDataSet(load_training_data=True, directory='data/audio_pt4',
                  f_output_file_name='features4.csv')

print('Combining feature dataframes')
combined_df = pd.concat([d1.song_features, d2.song_features, d3.song_features, d4.song_features])
combined_df.to_csv('data/features.csv', index=False)
