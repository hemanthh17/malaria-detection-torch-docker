from config import Config as cfg
from tqdm import tqdm 
from glob import glob 
from PIL import Image

import numpy as np 
import pandas as pd
import torchvision as tv
import torch
import torch.nn as nn 
import torch.optim as optim 
import timm
from torch.utils.data import DataLoader, Dataset

from sklearn import metrics

train_data= pd.DataFrame({'path':[pth for pth in glob(str(cfg.train_pth+'/*/*'))],
                    'label':[0 if lab.split("\\")[1].lower()=='uninfected' else 1 for lab in glob(str(cfg.train_pth+'/*/*'))]})
test_data= pd.DataFrame({'path':[pth for pth in glob(str(cfg.test_pth+'/*/*'))],
                    'label':[0 if lab.split("\\")[1].lower()=='uninfected' else 1 for lab in glob(str(cfg.test_pth+'/*/*'))]})
print(train_data.head())

if cfg.data_stat:
    trans= tv.transforms.Compose([
        tv.transforms.Resize(cfg.img_shape),
        tv.transforms.ToTensor()
    ])
    img_li=[trans(Image.open(img)) for img in train_data.path.values]
    label=[lab for lab in train_data.label.values]

    img_stack= torch.stack(img_li,dim=1)
    img_stack= img_stack.permute(1,0,2,3)
    print("Stack Dimension:",img_stack.shape)
    print("Mean of the images per channel:",torch.mean(img_stack,dim=(0,2,3)))
    print("Standard Deviation of the images per channel:",torch.std(img_stack,dim=(0,2,3)))


class MalariaDataset(Dataset):
    def __init__(self,data,transform=None):
        self.data=data
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        img_path= self.data.loc[idx,"path"] 
        label= self.data.loc[idx,"label"] 
        if self.transform:
            img=self.transform(Image.open(img_path))
        return {
            "img":img,
            "lab":torch.tensor(label)
        }
train_trans= tv.transforms.Compose([
        tv.transforms.Resize(cfg.img_shape),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.5349, 0.4215, 0.4339),std=(0.3397, 0.2707, 0.2738))
    ])
test_trans=tv.transforms.Compose([
        tv.transforms.Resize(cfg.img_shape),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.5349, 0.4215, 0.4339),std=(0.3397, 0.2707, 0.2738))
    ])

train_dataset=MalariaDataset(train_data,train_trans)
test_dataset=MalariaDataset(test_data,test_trans)
train_loader= DataLoader(train_dataset,batch_size=cfg.bs,shuffle=True)
test_loader= DataLoader(test_dataset,batch_size=cfg.bs)



class MalariaDetector(nn.Module):
    def __init__(self):
        super(MalariaDetector,self).__init__()
        self.base_model= timm.create_model(cfg.trans_model,pretrained=True)
        self.base_model.head=nn.Linear(self.base_model.head.in_features,2)
    def forward(self,imgs):
        return self.base_model(imgs)

model=MalariaDetector()
loss_fn= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr=cfg.lr)
n_epochs=10 

model.train()
for epoch in tqdm(range(n_epochs)):
    cons_loss=0
    for d in train_loader:
        img,label=d['img'],d['lab']
        out=model(img)
        loss= loss_fn(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cons_loss+=loss.item()

    print(f"Epoch: {epoch+1}, Loss: {cons_loss/cfg.bs}")

model.eval()
preds=[]
targets=[]
with torch.no_grad():
    for d in test_loader:
        img,label=d["img"],d["lab"]
        out=model(img)
        _,predicted=torch.max(out,dim=1)
        preds.extend(predicted.tolist())
        targets.extend(label.tolist())
    print("F1 score is:", metrics.f1_score(targets,preds))

torch.save(model.state_dict(),'C:/Users/Hemanth/Desktop/Data Analytics analyticvidya/pytorch malaria detection-docker/models/vit_model.pth')




        





        

