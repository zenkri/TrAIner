import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils import data
from features_extraction import json2data
import json

class Dataset(data.Dataset):
    def __init__(self, data, labels, seq_len):
        self.labels = labels
        self.data = data
        self.seq_len = seq_len


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        idx = min(index, len(self.data) - self.seq_len - 1)

        x = self.data[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len]

        return torch.tensor(x), torch.tensor(y)

def get_data():
    # import the json file
    dir_to_file = '/home/oussama/Desktop/TrAIner/data/pass.json'
    jsondata = []
    with open(dir_to_file) as jsonFile:
        try:
            jsondata = json.load(jsonFile)
            train_data, labels = json2data(jsondata)

            for col in range(train_data.shape[1]):
                train_data[:, col] = (train_data[:, col] - np.mean(train_data[:, col])) / np.std(train_data[:, col])

            idxs = np.arange(0, len(train_data))
            random.shuffle(idxs)

            train_data = train_data[idxs]
            labels = labels[idxs]

            test_len = int(0.3 * len(train_data))

            train_signals = train_data[:-test_len]
            test_signals = train_data[-test_len:]
            train_labels_idx = labels[:-test_len]
            test_labels_idx = labels[-test_len:]

            train_labels = np.zeros((0, 2))
            test_labels = np.zeros((0, 2))

            for label in train_labels_idx:
                one_hot = np.zeros((1, 2)).astype(int)[0]
                idx = label[0] > 0
                one_hot[int(idx)] = 1
                train_labels = np.vstack((train_labels, one_hot))

            for label in test_labels_idx:
                one_hot = np.zeros((1, 2)).astype(int)[0]
                one_hot[int(label[0] > 0)] = 1
                test_labels = np.vstack((test_labels, one_hot))


        except Exception as e:
            print(e)

    return train_signals.astype(float), train_labels.astype(float), test_signals.astype(float), test_labels.astype(float)


# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, seq_len, hidden_dim, batch_size, output_dim=1,num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.dropout = nn.Dropout(0.4)


        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.linear2 = nn.Linear(int(self.hidden_dim / 2), output_dim)


    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer

        lstm_out, self.hidden = self.lstm(input)
        hidden_state = self.hidden
        #print(hidden_state[0].shape)
        #lstm_out = lstm_out[:, -1]
        lstm_out = self.dropout(hidden_state[0])
        lstm_out = self.dropout(lstm_out[-1])
        # Only take the output from the final timetep
        #state = lstm_out[-1].view(self.hidden_dim)
        #print(state.shape)
        y_pred = F.leaky_relu(self.linear1(lstm_out))
        y_pred = self.linear2(y_pred)
        return y_pred

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.05 * 0.1 ** (0.5 * (epoch // 100) + 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_sample(X, y, batch_size, sample_len):
    X_output = np.zeros((batch_size, sample_len, X.shape[1]))
    y_output = np.zeros((batch_size, sample_len, y.shape[1]))
    for ii in range(batch_size):
        start_idx = random.randint(0, len(X) - sample_len)
        X_output[ii] = X[start_idx:start_idx + sample_len, :]
        y_output[ii] = y[start_idx:start_idx + sample_len, :]

    return X_output, y_output



if __name__ == "__main__":

    # Parameters
    params = {'batch_size': 512,
              'shuffle': True,
              'num_workers': 6
              }

    device = torch.device("cuda:0")

    seqence_length = 40

    train_data, train_labels, val_data, val_labels = get_data()

    training_set = Dataset(train_data, train_labels, seqence_length)
    training_generator = data.DataLoader(training_set, **params)

    val_set = Dataset(val_data, val_labels, seqence_length)
    val_generator = data.DataLoader(val_set, **params)

    lstm_input_size = train_data.shape[1]
    h1 = 32
    num_train = 512
    output_dim = 2
    num_layers = 1
    num_epochs = 400
    learning_rate = 0.001

    model = LSTM(lstm_input_size, 1, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

    model = model.float().to(device)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #####################
    # Train model
    #####################

    hist = np.zeros(num_epochs)

    for t in range(num_epochs):

        adjust_learning_rate(optimiser, t)

        for X_train, y_train in iter(training_generator):

            X_train, y_train = X_train.float().to(device), y_train.float().to(device)

            # Clear stored gradient
            model.zero_grad()

            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            model.hidden = model.init_hidden()

            # Forward pass
            y_pred = model(X_train)

            loss = loss_fn(F.softmax(y_pred), y_train)

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()

        X_val, y_val = next(iter(val_generator))

        X_val, y_val = X_val.float().to(device), y_val.float().to(device)

        model.eval()

        #model.hidden = model.init_hidden

        y_val_pred = model(X_val)

        model.train()

        val_loss = loss_fn(F.softmax(y_val_pred), y_val)

        print("Epoch ", t, "Train MSE: ", loss.item(), "Validation MSE: ", val_loss.item())
        hist[t] = loss.item()

    total_loss = 0
    tatal_values = 0

    for X_val, y_val in val_generator:

        X_val, y_val = X_val.float().to(device), y_val.float().to(device)

        model.eval()

        model.hidden = model.init_hidden

        y_val_pred = model(X_val)

        pos_max = np.argmax(y_val_pred.cpu().detach().numpy(), axis=1).astype(int)

        pospos = np.argmax(y_val.cpu().detach().numpy(), axis=1).astype(int)


        total_loss += sum(pos_max == pospos)
        tatal_values += len(pos_max)

    print(total_loss / tatal_values)

    for i in range(len(pos_max)):
        print(pos_max[i], pospos[i])

