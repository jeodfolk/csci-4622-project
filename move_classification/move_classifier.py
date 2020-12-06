import torchvision.models as models
import torch
import logging
from torch import nn
from PIL import Image
from time import time
from os import path
import pickle

class MoveClassifier(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        with open(path.join(path.dirname(__file__), 'class_dicts/idx_to_move.pkl'), 'rb') as f:
            self.idx_to_move = pickle.load(f)
        self.device = device
        self.cnn = models.squeezenet1_0(pretrained=True).to(device)

        for param in self.cnn.parameters():
            param.requires_grad = False

        last_layer_dim = 1024
        self.cnn.classifier[1] = nn.Conv2d(512, last_layer_dim, kernel_size=(1,1), stride=(1,1))
        self.cnn.num_classes = last_layer_dim

        self.lstm = nn.LSTM(input_size=last_layer_dim, hidden_size=512, num_layers=3)
        self.linear = nn.Linear(512, num_classes)
    
    def forward(self, X):
        lstm_in = []
        for i in range(X.shape[0]):
            lstm_in.append(self.cnn(X[0]))
        lstm_in_tensor = torch.stack(lstm_in).to(self.device)
        lstm_out, _ = self.lstm(lstm_in_tensor)
        linear_in = lstm_out.narrow(1, lstm_out.shape[1]-1, 1) # take only the output from the final image
        linear_in = torch.squeeze(linear_in)
        y_hat = self.linear(linear_in)
        return y_hat

    def save(self, filepath):
        logging.info("Saving model to {}".format(filepath))
        torch.save(self.state_dict(), filepath)
        logging.info("Model saved successfully.")

    def load(self, filepath):
        logging.info("Loading model from {}".format(filepath))
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        logging.info("Model loaded successfully.")
