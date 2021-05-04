from torch.utils.data import Dataset
from PIL import Image
import torch


class ImageCaptionDataset(Dataset):
    def __init__(self, img_paths, captions, transform):
        super(ImageCaptionDataset, self).__init__()
        self.transform = transform
        self.img_paths = img_paths
        self.captions = captions

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return torch.FloatTensor(img), torch.tensor(random.choice(self.captions[index])), torch.tensor(self.captions[index])

    def __len__(self):
        return len(self.captions)
