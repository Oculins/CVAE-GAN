import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import transforms
import torch
import numpy as np

# modify for transformation for vit
# modfify wider crop-person images

def age_vector(age, num_cls):
    assert num_cls in [7]

    agev = np.zeros(num_cls)

    if num_cls == 7:
        if age < 45:
            agev[0] = 1
        elif age < 50:
            agev[1] = 1
        elif age < 55:
            agev[2] = 1
        elif age < 60:
            agev[3] = 1
        elif age < 65:
            agev[4] = 1
        elif age < 70:
            agev[5] = 1
        else:
            agev[6] = 1
    
    return agev

class DataSet(Dataset):
    def __init__(self,
                ann_file,
                img_size,
                num_cls,
                speedup,
                ):
        self.ann_file = ann_file
        self.num_cls = num_cls
        self.img_size = img_size
        self.speedup = speedup
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
            ] 
        )
        self.anns = []
        self.load_anns()    

    
    def load_anns(self):

        self.anns = json.load(open(self.ann_file, "r"))
        if self.speedup:
            self.anns = self.anns[:int(len(self.anns) / 1000)]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        idx = idx % len(self)
        ann = self.anns[idx]
        img = Image.open(ann["img_path"]).convert("RGB")
        img = self.transform(img)
        label = age_vector(ann["target"], self.num_cls)

        message = {
            "img_path": ann["img_path"],
            "label": torch.from_numpy(label).float(),
            "img": img, 
            "age": torch.Tensor([ann["target"] / 100]).float(),
        }

        return message

