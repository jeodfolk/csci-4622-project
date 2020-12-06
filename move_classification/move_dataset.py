from torch.utils.data import Dataset
from os import path
from PIL import Image
import torchvision.transforms as transforms
import glob
import torch
import pickle

class MoveDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.move_data = glob.glob(path.join(path.dirname(__file__), 'move_data/*/*'))
        self.preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open(path.join(path.dirname(__file__), 'class_dicts/move_to_idx.pkl'), 'rb') as f:
            self.move_to_idx = pickle.load(f)


    def process(self, sample):
        frame_paths = glob.glob(path.join(sample, '*.png'))
        frames = [self.preprocess(Image.open(f).convert('RGB')) for f in frame_paths]
        X = torch.stack(frames)

        y = path.basename(path.dirname(sample)) # class is name of parent directory
        y = self.move_to_idx[y]

        return X, y

    def __len__(self):
        return len(self.move_data)
    
    def __getitem__(self, index):
        return self.process(self.move_data[index])
