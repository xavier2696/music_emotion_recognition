from NNModel import NeuralNet
from preprocessing_librosa import AudioDataSet
import torch
import pandas as pd

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
        if (t + 1) % 100 == 0 or t == 0:
            print("Iteration:", t + 1, "Loss:", loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'models/{file_name}.pt')

def show_results(prediction_df, ground_truth_df):
    correct_first_quadrant = 0
    total_first_quadrant = 0
    correct_second_quadrant = 0
    total_second_quadrant = 0
    correct_third_quadrant = 0
    total_third_quadrant = 0
    correct_fourth_quadrant = 0
    total_fourth_quadrant = 0
    for i in range(0, len(ground_truth_df.valence)):
        if ground_truth_df.valence.values[i] >= 0 and ground_truth_df.arousal.values[i] >= 0:
            total_first_quadrant += 1
            if prediction_df.valence.values[i] >= 0 and prediction_df.arousal.values[i] >= 0:
                correct_first_quadrant += 1
        elif ground_truth_df.valence.values[i] >= 0 and ground_truth_df.arousal.values[i] < 0:
            total_fourth_quadrant += 1
            if prediction_df.valence.values[i] >= 0 and prediction_df.arousal.values[i] < 0:
                correct_fourth_quadrant += 1
        elif ground_truth_df.valence.values[i] < 0 and ground_truth_df.arousal.values[i] < 0:
            total_third_quadrant += 1
            if prediction_df.valence.values[i] < 0 and prediction_df.arousal.values[i] < 0:
                correct_third_quadrant += 1
        elif ground_truth_df.valence.values[i] < 0 and ground_truth_df.arousal.values[i] >= 0:
            total_second_quadrant += 1
            if prediction_df.valence.values[i] < 0 and prediction_df.arousal.values[i] >= 0:
                correct_second_quadrant += 1
    print('First quadrant:', correct_first_quadrant, '/', total_first_quadrant)
    print('Second quadrant:', correct_second_quadrant, '/', total_second_quadrant)
    print('Third quadrant:', correct_third_quadrant, '/', total_third_quadrant)
    print('Fourth quadrant:', correct_fourth_quadrant, '/', total_fourth_quadrant)

# main code
complete_dataset = AudioDataSet(load_training_data=True)
print('Dataset length:', len(complete_dataset.dataset))

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

training = True
v_model = NeuralNet(n_in, n_out)
if training:
    v_model = v_model.to(device)
    train_model(v_model, x_train, y_v_train, 'valence_model', 5000)
else:
    v_model.load_state_dict(torch.load('models/valence_model.pt'))
    v_model = v_model.to(device)
    v_model.eval()

training = True
a_model = NeuralNet(n_in, n_out)
if training:
    a_model = a_model.to(device)
    train_model(a_model, x_train, y_a_train, 'arousal_model', 5000)
else:
    a_model.load_state_dict(torch.load('models/arousal_model.pt'))
    a_model = a_model.to(device)
    a_model.eval()

#see results
with torch.no_grad():
    loss = torch.nn.MSELoss()

    v_predicted_train = v_model(x_train)
    v_rmse_train = torch.sqrt(loss(v_predicted_train, y_v_train))
    print('\nTrain Results:')
    print('Valence RMSE:', v_rmse_train.item())
    a_predicted_train = a_model(x_train)
    a_rmse_train = torch.sqrt(loss(a_predicted_train, y_a_train))
    print('Arousal RMSE:', a_rmse_train.item())
    v_predicted_train = [x[0].item() for x in v_predicted_train]
    a_predicted_train = [x[0].item() for x in a_predicted_train]
    prediction_df_train = pd.DataFrame({'valence': v_predicted_train, 'arousal': a_predicted_train})
    ground_truth_df_train = pd.DataFrame(
        {'valence': complete_dataset.y_train.valence, 'arousal': complete_dataset.y_train.arousal})
    show_results(prediction_df_train, ground_truth_df_train)

    v_predicted = v_model(x_test)
    v_rmse = torch.sqrt(loss(v_predicted, y_v_test))
    print('\nTest Results:')
    print('Valence RMSE:', v_rmse.item())
    a_predicted = a_model(x_test)
    a_rmse = torch.sqrt(loss(a_predicted, y_a_test))
    print('Arousal RMSE:', a_rmse.item())
    v_predicted = [x[0].item() for x in v_predicted]
    a_predicted = [x[0].item() for x in a_predicted]
    prediction_df = pd.DataFrame({'valence': v_predicted, 'arousal': a_predicted})
    ground_truth_df = pd.DataFrame(
        {'valence': complete_dataset.y_test.valence, 'arousal': complete_dataset.y_test.arousal})
    show_results(prediction_df, ground_truth_df)

