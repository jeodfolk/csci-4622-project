from move_classification.move_classifier import MoveClassifier
from torch import nn
from os import path
import torch
import pickle

class MoveClassifierTrainer:
    def __init__(self, device):
        self.device = device
        with open(path.join(path.dirname(__file__), 'class_dicts/idx_to_move.pkl'), 'rb') as f:
            self.idx_to_move = pickle.load(f)
        self.model = MoveClassifier(len(self.idx_to_move), device).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.00005)

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
        model_out = self.model(X)
        y_hat = None
        if len(model_out.shape) > 1:
            y_hat = torch.argmax(model_out, dim=1)
        else:
            y_hat = torch.argmax(model_out)
        correct = torch.sum(y_hat == y)

        label_total = {}
        for i in range(len(y)):
            if y[i].item() in label_total:
                label_total[y[i].item()] += 1
            else:
                label_total[y[i].item()] = 1
        label_correct = {}
        for i in range(len(y_hat)):
            if y_hat[i].item() == y[i].item():
                if y_hat[i].item() in label_correct:
                    label_correct[y_hat[i].item()] += 1
                else:
                    label_correct[y_hat[i].item()] = 1

        return correct, label_total, label_correct