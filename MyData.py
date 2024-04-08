import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self,root_dir,label_dir,transforms):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path_list = os.listdir(self.path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_name = self.img_path_list[index]
        ima_name_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = self.transforms(Image.open(ima_name_path).convert('RGB'))
        label = self.label_dir
        return img,label
    
    def __len__(self):
        return len(self.img_path_list)
    


