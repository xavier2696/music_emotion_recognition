from NNModel import NeuralNet
from preprocessing_librosa import AudioDataSet
import torch

def train_model(model, x_train, y_train, file_name, iterations):
    batch_size = 64
    iterations = iterations
    learning_rate = 1e-4
    # loss
    criterion = torch.nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    for t in range(iterations):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x_train)

        # Compute and print loss
        loss = torch.sqrt(criterion(y_pred, y_train)) # rmse
        if t % 100 == 0:
            print("Iteration:", t, "Loss:", loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'{file_name}.pt')

# main code
complete_dataset = AudioDataSet(load_training_data=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
x_train = torch.from_numpy(complete_dataset.x_train.values)
x_train = x_train.type('torch.FloatTensor').to(device)
y_v_train = torch.from_numpy(complete_dataset.y_train.valence.values)
y_v_train = y_v_train.type('torch.FloatTensor').to(device)
y_a_train = torch.from_numpy(complete_dataset.y_train.arousal.values)
y_a_train = y_a_train.type('torch.FloatTensor').to(device)

x_test = torch.from_numpy(complete_dataset.x_test.values)
x_test = x_test.type('torch.FloatTensor').to(device)
y_v_test = torch.from_numpy(complete_dataset.y_test.valence.values)
y_v_test = y_v_test.type('torch.FloatTensor').to(device)
y_a_test = torch.from_numpy(complete_dataset.y_test.arousal.values)
y_a_test = y_a_test.type('torch.FloatTensor').to(device)

n_in = x_train.shape[1]
n_out = 1

v_model = NeuralNet(n_in, n_out)
#load model
v_model.load_state_dict(torch.load('valence_model.pt'))
v_model = v_model.to(device)
v_model.eval()
#train model
#v_model = v_model.to(device)
#train_model(v_model, x_train, y_v_train, 'valence_model', 5000)

a_model = NeuralNet(n_in, n_out)
#load model
a_model.load_state_dict(torch.load('arousal_model.pt'))
a_model = a_model.to(device)
a_model.eval()
#train model
#a_model = a_model.to(device)
#train_model(a_model, x_train, y_a_train, 'arousal_model', 5000)

#see results
with torch.no_grad():
    loss = torch.nn.MSELoss()
    v_predicted = v_model(x_test)
    v_rmse = torch.sqrt(loss(v_predicted, y_v_test))
    print('Valence RMSE:', v_rmse.item())
    a_predicted = a_model(x_test)
    a_rmse = torch.sqrt(loss(a_predicted, y_a_test))
    print('Arousal RMSE:', a_rmse.item())


