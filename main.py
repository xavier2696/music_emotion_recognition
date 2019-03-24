from NNModel import NeuralNet
from preprocessing_librosa import AudioDataSet
import torch

complete_dataset = AudioDataSet(load_training_data=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
x = torch.from_numpy(complete_dataset.x.values)
x = x.type('torch.FloatTensor').to(device)
#print(x)
y = torch.from_numpy(complete_dataset.dataset[['arousal', 'valence']].values) #both at once
#y = torch.from_numpy(complete_dataset.arousal.values)
y = y.type('torch.FloatTensor').to(device).to(device)
#print(y)

print(x.shape)
#print(y.shape)
n_in = x.shape[1]
n_out = 2
batch_size = 64
epochs = 20
learning_rate = 1e-4

model = NeuralNet(n_in, n_out)
model = model.to(device)
print(model)

#loss
criterion = torch.nn.MSELoss()
#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training
for t in range(epochs):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = torch.sqrt(criterion(y_pred, y)) #rmse
    print("Iteration:", t, "Loss:", loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(y[0:5])
with torch.no_grad():
    outputs = model(x[0:5])
    print(outputs)