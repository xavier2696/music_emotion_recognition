from NNModel import NeuralNet
from LSTMModel import LSTM
from preprocessing_librosa import AudioDataSet
import torch
import pandas as pd

lstm = True
gpu_number = 1

complete_dataset = AudioDataSet(load_training_data=False, directory='../data/karaoke/song_fragments',
                                f_output_file_name='../data/unlabeled_features.csv')
print('Dataset length:', len(complete_dataset.dataset))

device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
print(device)
x = torch.from_numpy(complete_dataset.x.values)
x = x.type('torch.FloatTensor').to(device)

n_in = x.shape[1]
n_out = 1

# v_model = NeuralNet(n_in, n_out)
v_model = LSTM(n_in, 256, output_dim=n_out, batch_size=1)
v_model.load_state_dict(torch.load('../models/valence_model.pt'))
v_model = v_model.to(device)
v_model.eval()

a_model = LSTM(n_in, 256, output_dim=n_out, batch_size=1)
a_model.load_state_dict(torch.load('../models/arousal_model.pt'))
a_model = a_model.to(device)
a_model.eval()

with torch.no_grad():
    if lstm:
        v_predicted = []
        for i in range(0, x.shape[0]):
            v_predicted.append([v_model(x[i:(i+1)])])
        v_predicted = torch.tensor(v_predicted)
        v_predicted = v_predicted.type('torch.FloatTensor').to(device)
    else:
        v_predicted = v_model(x)
    v_predicted = [item[0].item() for item in v_predicted]

    if lstm:
        a_predicted = []
        for i in range(0, x.shape[0]):
            a_predicted.append([a_model(x[i:(i+1)])])
        a_predicted = torch.tensor(a_predicted)
        a_predicted = a_predicted.type('torch.FloatTensor').to(device)
    else:
        a_predicted = a_model(x)
    a_predicted = [item[0].item() for item in a_predicted]

    dataframe = pd.DataFrame({'id': complete_dataset.song_features.song_name,
                              'valence': v_predicted,
                              'arousal': a_predicted})
    dataframe.to_csv('../data/karaoke/predictions.csv', index=False)
