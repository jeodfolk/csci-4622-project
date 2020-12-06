from move_classification.move_classifier_trainer import MoveClassifierTrainer
from move_classification.move_dataset import MoveDataset
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import torch
import datetime
import time
import matplotlib.pyplot as plt
import logging

def time_since(since):
    now = time.time()
    s = now - since
    m = s // 60
    s -= m * 60
    return '%dm %ds' % (m, s)

batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MoveDataset()
train_size = int(0.9*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
print('dataset loaded')
trainer = MoveClassifierTrainer(device)
print('trainer loaded')

# logging setup
dt_now = datetime.datetime.now()
dirname = os.path.dirname(__file__)

logging.basicConfig(level=logging.INFO,
    filename="{}/logs/{}".format(dirname, dt_now.strftime('%Y_%m_%d_%H_%M_%S.log')),
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logging.info('Training start')

print(len(train_dataset))
print(len(train_dataloader))

start = time.time()
epochs = 40

all_losses = []
total_loss = 0

iter_n = 1
n_iters = len(train_dataloader)*epochs
for epoch in range(epochs):
    for batch_ind, sample in enumerate(train_dataloader):
        loss = trainer.train_step(sample)
        total_loss += loss

        if iter_n % 1 == 0:
            logging.info('%s (%d %.2f%%) %.4f' % (time_since(start), iter_n, iter_n/n_iters*100, loss))

        if iter_n % 1 == 0:
            all_losses.append(total_loss / 100)
            total_loss = 0

            plt.figure()
            plt.plot(all_losses)
            savepath = "plots/{}".format(dt_now.strftime('%Y_%m_%d_%H_%M_%S'))
            savepath = os.path.join(dirname, savepath)
            plt.savefig(savepath)
            plt.close()

        if iter_n % 3 == 0:
            savepath = "saved_models/{}".format(dt_now.strftime('%Y_%m_%d_%H_%M_%S'))
            savepath = os.path.join(dirname, savepath)
            trainer.model.save(savepath)

        iter_n += 1

trainer.model.eval()
train_acc = 0
for batch_ind, sample in enumerate(test_dataloader):
    train_acc += trainer.evaluate_accuracy(sample)
final_train_acc = train_acc / len(test_dataset)

logging.info('Test accuracy: %f' % (final_train_acc.item()))