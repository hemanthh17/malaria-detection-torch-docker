import io
import os
from scripts.config import Config as cfg 
import numpy as np 
import torch 
import torch.nn as nn
from PIL import Image
import timm
import torchvision as tv
class MalariaDetector(nn.Module):
    def __init__(self):
        super(MalariaDetector, self).__init__()
        self.base_model = timm.create_model(cfg.trans_model, pretrained=False)
        self.base_model.head = nn.Linear(self.base_model.head.in_features, 2)

    def forward(self, imgs):
        return self.base_model(imgs)

def predict_malaria(img):
    img_data=img.copy()
    model_wt= torch.load(str(cfg.model_pth+'/vit_model.pth'))
    model=MalariaDetector()
    model.load_state_dict(model_wt)
    model.eval()
    transform=tv.transforms.Compose([
        tv.transforms.Resize(cfg.img_shape),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.5349, 0.4215, 0.4339),std=(0.3397, 0.2707, 0.2738))
    ])
    img_t=transform(img_data)
    img_t=img_t.unsqueeze(0)
    print(img_t.shape)
    out=torch.argmax(model(img_t.float())[0]).item()
    print('Done')
    print(out)
    if out==0:
        return "Uninfected"
    return "Infected"
if __name__ == "__main__":
    print('This is a Helper script!! execute the Flask script outside')
