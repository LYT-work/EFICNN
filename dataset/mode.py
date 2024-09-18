import os
from torch.utils.data import Dataset
from dataset.transform import *


class Dataset(Dataset):
    def __init__(self, name, root, mode):
        self.name = name
        self.root = root
        self.mode = mode

        if mode == 'train':
            with open('splits/%s/train.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'val':
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'test':
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]

        if self.mode == 'train':
            imgA = Image.open(os.path.join(self.root, 'train', 'A', id)).convert('RGB')
            imgB = Image.open(os.path.join(self.root, 'train', 'B', id)).convert('RGB')

            mask = np.array(Image.open(os.path.join(self.root, 'train', 'label', id)))
            mask = mask / 255
            mask = Image.fromarray(mask.astype(np.uint8))

            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)

            return imgA, imgB, mask

        if self.mode == 'val':
            imgA = Image.open(os.path.join(self.root, 'val', 'A', id)).convert('RGB')
            imgB = Image.open(os.path.join(self.root, 'val', 'B', id)).convert('RGB')

            mask = np.array(Image.open(os.path.join(self.root, 'val', 'label', id)))
            mask = mask / 255
            mask = Image.fromarray(mask.astype(np.uint8))

            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id

        if self.mode == 'test':
            imgA = Image.open(os.path.join(self.root, 'test', 'A', id)).convert('RGB')
            imgB = Image.open(os.path.join(self.root, 'test', 'B', id)).convert('RGB')

            mask = np.array(Image.open(os.path.join(self.root, 'test', 'label', id)))
            mask = mask / 255
            mask = Image.fromarray(mask.astype(np.uint8))

            imgA, mask = normalize(imgA, mask)
            imgB = normalize(imgB)
            return imgA, imgB, mask, id

    def __len__(self):
        return len(self.ids)

