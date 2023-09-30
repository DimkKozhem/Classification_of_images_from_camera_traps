#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

class AnimalClassification:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])        

    def predict(self, image_files, verbose=False):
        empty = []
        bad = []
        model.to(device)
        for filename in image_files:
            # Загружаем изображение
            try:
                    image = Image.open(filename)
                    images = self.data_transform(image)
            except:
                    bad.append(filename)
                    continue
            images = images.unsqueeze(0).to(device)
            with torch.no_grad():
                    outputs = model(images)
                    predicted = torch.argmax(outputs).to('cpu').item()               

            if verbose:
                    print(filename)

            if predicted == 1:
                    empty.append(filename)
            else:
                    bad.append(filename)
            
        return bad, empty



class Efficientnet_b4(nn.Module):
    def __init__(self, num_classes=2):
        super(Efficientnet_b4, self).__init__()
        self.efficientnet = models.efficientnet_b4 ()
        self.linear_layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.linear_layer(x)
        return x



if __name__ == '__main__':
    # Определяем список файлов для детекции
    DATASET_PATH_ANIMAL = r'I:\val\1'
    path_files = os.path.abspath(DATASET_PATH_ANIMAL)
    files = [entry.path for entry in os.scandir(path_files) if entry.is_file()]

    # Загружаем модель
    model = Efficientnet_b4()
    model.load_state_dict(torch.load('efficient_weights_2class.pth'))
    model.eval()

    # Если есть GPU то устанавливаем device = 'cuda'
    device = f"cuda" if torch.cuda.is_available() else "cpu"

    animal_сlassification = AnimalClassification(model, device)
    # Делаем прогноз
    bad, empty = animal_сlassification.predict(files, verbose=False)

    print(len(bad), len(empty))


# In[ ]:




