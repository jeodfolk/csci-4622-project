from move_classification.move_classifier import MoveClassifier
from torch import nn
from os import path
import torch
import pickle

class MoveClassifierTrainer:
    def __init__(self, device):
        with open(path.join(path.dirname(__file__), 'class_dicts/idx_to_move.pkl'), 'rb') as f:
            self.idx_to_move = pickle.load(f)
        self.device = device
        self.model = MoveClassifier(len(self.idx_to_move), device).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def preprocess(self, sample):
        return self.preprocessor(sample).to(self.device)

    def train_step(self, sample):
        self.optim.zero_grad()
        X, y = sample
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = self.model(X)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optim.step()
        return loss.item()

    def evaluate_accuracy(self, sample):
        X, y = sample
        X = X.to(self.device)
        y = y.to(self.device)
        y_hat = torch.argmax(self.model(X), dim=1)
        correct = torch.sum(y_hat == y)
        return correct