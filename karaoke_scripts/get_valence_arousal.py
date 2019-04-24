from NNModel import NeuralNet
from LSTMModel import LSTM
from LSTMModel2 import SimpleRNN
from LSTMModel3 import LSTMModel
from preprocessing_librosa import AudioDataSet
import torch
import pandas as pd

lstm = True
gpu_number = 1

#discrete emotions for valence-arousal
    #positive valence & positive arousal
        #|valence| >= |arousal| = happy
        #|valence| < |arousal| = excited
    #negative valence & positive arousal
        #|valence| >= |arousal| =  angry
        #|valence| < |arousal| = afraid
    #negative valence & negative arousal
        #|valence| >= |arousal| = depressed
        #|valence| < |arousal| = sad
    #positive valence & negative arousal
        #|valence| >= |arousal| = calm
        #|valence| < |arousal| = content
def discrete_emotion(row):
    if row.valence >= 0 and row.arousal >= 0:
        if abs(row.valence) >= abs(row.arousal):
            return "happy"
        else:
            return "excited"
    if row.valence < 0 and row.arousal >= 0:
        if abs(row.valence) >= abs(row.arousal):
            return "angry"
        else:
            return "afraid"
    if row.valence < 0 and row.arousal < 0:
        if abs(row.valence) >= abs(row.arousal):
            return "depressed"
        else:
            return "sad"
    if row.valence >= 0 and row.arousal < 0:
        if abs(row.valence) >= abs(row.arousal):
            return "calm"
        else:
            return "content"

complete_dataset = AudioDataSet(load_training_data=False, directory='../data/karaoke/song_fragments',
                                f_output_file_name='../data/unlabeled_features.csv')
print('Dataset length:', len(complete_dataset.x))

device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu")
print(device)
x = torch.from_numpy(complete_dataset.x.values)
x = x.type('torch.FloatTensor').to(device)

n_in = x.shape[1]
n_out = 1

v_model = NeuralNet(n_in, n_out)
if lstm:
    #v_model = LSTM(n_in, 256, output_dim=n_out, batch_size=1)
    #v_model = SimpleRNN(n_in)
    v_model = LSTMModel(n_in, 128, 1, n_out)
v_model.load_state_dict(torch.load('../models/valence_model.pt'))
v_model = v_model.to(device)
v_model.eval()

a_model = NeuralNet(n_in, n_out)
if lstm:
    #a_model = LSTM(n_in, 256, output_dim=n_out, batch_size=1)
    #a_model = SimpleRNN(n_in)
    a_model = LSTMModel(n_in, 128, 1, n_out)
a_model.load_state_dict(torch.load('../models/arousal_model.pt'))
a_model = a_model.to(device)
a_model.eval()

with torch.no_grad():
    if lstm:
        v_predicted = []
        for i in range(0, x.shape[0]):
            inp = x[i:(i + 1)]
            inp = inp.view(-1, 1, n_in)
            v_predicted.append(v_model(inp).item())
            #v_predicted.append([v_model(x[i:(i+1)])[0][0].item()])
        v_predicted = torch.tensor(v_predicted)
        v_predicted = v_predicted.type('torch.FloatTensor').to(device)
    else:
        v_predicted = v_model(x)
    #v_predicted = [item[0].item() for item in v_predicted]
    v_predicted = [item.item() for item in v_predicted]

    if lstm:
        a_predicted = []
        for i in range(0, x.shape[0]):
            inp = x[i:(i + 1)]
            inp = inp.view(-1, 1, n_in)
            print(inp)
            a_predicted.append(a_model(inp).item())
            #a_predicted.append([a_model(x[i:(i+1)])[0][0].item()])
        a_predicted = torch.tensor(a_predicted)
        a_predicted = a_predicted.type('torch.FloatTensor').to(device)
    else:
        a_predicted = a_model(x)
    #a_predicted = [item[0].item() for item in a_predicted]
    a_predicted = [item.item() for item in a_predicted]

    va_df = pd.DataFrame({'fragment_id': complete_dataset.song_features.song_name,
                              'valence': v_predicted,
                              'arousal': a_predicted})
    va_df['emotion'] = va_df.apply(lambda row: discrete_emotion(row), axis=1)
    va_df.to_csv('../data/karaoke/predictions.csv', index=False)


    #create dataframe with sentences with it's respective emotion

    sentences_df = pd.read_csv('../data/karaoke/sentences.csv')

    complete_df = va_df.merge(sentences_df, how='inner', on='fragment_id')

    complete_df = complete_df.sort_values('fragment_id')
    complete_df.to_csv('../data/karaoke/sentence_emotions.csv', index=False)



