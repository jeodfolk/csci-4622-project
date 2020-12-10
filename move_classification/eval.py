from move_classification.move_classifier_trainer import MoveClassifierTrainer
from move_classification.move_classifier import MoveClassifier
from move_classification.move_dataset import MoveDataset
from torch.utils.data import DataLoader
from os import path
from PIL import Image
import torchvision.transforms as transforms
import torch
import glob
import csv

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = MoveClassifierTrainer(device)
trainer.model.load(path.join(path.dirname(__file__), 'saved_models/2020_12_06_21_42_06'))

dataset = MoveDataset()
dataloader = DataLoader(dataset, batch_size=256, num_workers=8)

label_correct = {}
label_total = {}

trainer.model.eval()
train_acc = 0
for batch_ind, sample in enumerate(dataloader):
    print(batch_ind)
    _, label = sample

    correct, l_total, l_correct = trainer.evaluate_accuracy(sample)
    train_acc += correct

    for item in l_total:
        if item in label_total:
            label_total[item] += l_total[item]
        else:
            label_total[item] = l_total[item]
    for item in l_correct:
        if item in label_correct:
            label_correct[item] += l_correct[item]
        else:
            label_correct[item] = l_correct[item]

final_train_acc = train_acc / len(dataset)
print(final_train_acc)

with open('eval.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['move index', 'correct', 'total'])
    for key in label_total:
        spamwriter.writerow([key, label_correct.get(key, 0), label_total[key]])