from move_classification.move_classifier import MoveClassifier
from os import path
from PIL import Image
import torchvision.transforms as transforms
import torch
import glob

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MoveClassifier(69, device).to(device)
model.load(path.join(path.dirname(__file__), 'saved_models/2020_12_06_21_42_06'))

while True:
    frame_paths = glob.glob(path.join(path.dirname(__file__), 'move_test_data/*'))
    frames = [preprocess(Image.open(f).convert('RGB')) for f in frame_paths]
    X = torch.stack(frames)
    X = torch.unsqueeze(X, 0)
    X = X.to(device)

    model.eval()
    y_hat = model(X)
    pred_move = model.idx_to_move[torch.argmax(y_hat).item()]
    print(pred_move)