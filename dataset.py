from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class ImageCaptionDataset(Dataset):
    def __init__(self, img_paths, captions, transform):
        super(ImageCaptionDataset, self).__init__()
        self.transform = transform
        self.img_paths = []
        self.captions = []
        for i in range(len(img_paths)):
            for caption in captions[i]:
                self.captions.append((caption,captions[i]))
                self.img_paths.append(img_paths[i])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return torch.FloatTensor(img), torch.tensor(self.captions[index][0]), torch.tensor(self.captions[index][1])

    def __len__(self):
        return len(self.captions)
