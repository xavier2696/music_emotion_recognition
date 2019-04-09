import urllib.request
import pandas as pd

songs = pd.read_csv("../data/karaoke/music2.csv", header=None, names=['name', 'lyric', 'path'], sep=';')
for index, row in songs.iterrows():
    print(row['path'])
    urllib.request.urlretrieve(f"{row['path']}", f"../data/karaoke/mp3/{row['name']}.mp3")
    urllib.request.urlretrieve(f"{row['lyric']}", f"../data/karaoke/metadata/{row['name']}.txt")
