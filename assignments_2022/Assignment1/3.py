import numpy as np
import torch
import os
import pandas as pd
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PokemonDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # Path to the csv file with annotations.
        self.root_dir = root_dir  # Directory with all the images.
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1] + ".jpg")
        image = cv2.imread(img_path)
        y_label = torch.tensor(self.annotations['label'][index])

        if self.transform:
            image = self.transform(image)

        return image, y_label


class PokemonDataLoader(DataLoader):
    def __init__(self, dataset: PokemonDataset, batch_size: int = 10, collate_fn=None, shuffle: bool = True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        self.len = len(dataset)

    def __len__(self):
        return self.len


def collate_batch(batch):
    images, labels = [], []
    for (_image, _label) in batch:
        # np.append(labels, _label)
        labels.append(_label)
        images.append(cv2.resize(_image, dsize=(int(_image.shape[0] * .8), int(_image.shape[1] * .8)),
                                 interpolation=cv2.INTER_CUBIC))
    images = np.array([i for i in images])
    images_torch = torch.from_numpy(images)
    labels_torch = torch.tensor(labels, dtype=torch.int64)
    return images_torch, labels_torch


ds = PokemonDataset("./pokemon.csv", "/home/carbon/Documents/CodeWorkspace/NNTI/1/pokemon_images")
dl = PokemonDataLoader(ds, collate_fn=collate_batch)
img, label = next(iter(dl))
print(img)
print(next(iter(dl)))
