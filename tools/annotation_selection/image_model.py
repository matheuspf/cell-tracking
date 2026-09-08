"""Image architecture shared by source fitting and annotation-free inference."""
import numpy as np
import torch
from torch import nn


class PatchModel(nn.Module):
    def __init__(self,tabular=0):
        super().__init__();self.tabular=tabular
        self.encoder=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                   nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                   nn.Conv2d(32,48,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
        self.head=nn.Sequential(nn.Linear(48+tabular,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,image,tab):
        x=self.encoder(image)
        if self.tabular:x=torch.cat([x,tab],dim=1)
        return self.head(x)[:,0]


@torch.no_grad()
def predict_image(checkpoint,patches,features,device='cuda'):
    model=PatchModel(features.shape[1] if checkpoint['plus_tabular'] else 0).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    mean=np.asarray(checkpoint['tab_mean'],np.float32);std=np.asarray(checkpoint['tab_std'],np.float32)
    out=[]
    for start in range(0,len(features),512):
        im=np.array(patches[start:start+512],np.float32)/255.
        im=(im-checkpoint['image_mean'])/checkpoint['image_std']
        tab=(features[start:start+512]-mean)/std
        logits=model(torch.from_numpy(im).to(device),torch.from_numpy(tab).to(device))
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0,np.float32)
