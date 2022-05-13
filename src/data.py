import numpy as np
import pandas as pd
import torch.utils.data as data
from monai.transforms import (Compose, RandAffine, RandHistogramShift, Resize,
                              ScaleIntensity, ToTensor)
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import WeightedRandomSampler

from src.utils import load_img


class kneeMRIDataset(data.Dataset):
    def __init__(self, df, transform=None, augement=None):
        self.X = df.drop(columns="aclDiagnosis")
        self.y = df["aclDiagnosis"]
        self.transform = transform
        self.augement = augement

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        row = self.X.iloc[idx]
        filename = row["volumeFilename"]

        img = load_img(f"data/scans/{filename}").astype('float32')

        roi = img[row.roiZ: row.roiZ+row.roiDepth, row.roiY: row.roiY +
                  row.roiWidth, row.roiX: row.roiX+row.roiHeight]

        roi = np.expand_dims(roi, axis=0)

        if self.augement:
            roi = self.augement(roi)

        if self.transform:
            roi = self.transform(roi)

        return [roi, self.y.iloc[idx]]


def prepare_data(sampling_frac=1.0, num_workers=None):
    df = load_df(sampling_frac)

    subsets = split_data(df)

    datasets = prepare_datasets(subsets)

    dataloaders = prepare_dataloaders(datasets, num_workers)

    return dataloaders


def load_df(sampling_frac):
    df = pd.read_csv("data/metadata.csv")

    df["aclDiagnosis"] = (df["aclDiagnosis"] > 0).astype('float32')

    if sampling_frac < 1.0:
        df = df.sample(frac=sampling_frac, random_state=1)

    return df


def split_data(df):
    df_train, df_test = train_test_split(
        df, test_size=0.3, random_state=1, shuffle=True, stratify=df["aclDiagnosis"])
    df_valid, df_test = train_test_split(
        df_test, test_size=0.5, random_state=1, shuffle=True, stratify=df_test["aclDiagnosis"])

    print("Train/Valid/Test dataset length:", len(df_train),
          len(df_valid), len(df_test))

    return df_train, df_valid, df_test


def prepare_datasets(subsets):
    transformations = Compose([
                              Resize((3, 90, 90))],
                              ScaleIntensity(),
                              ToTensor()
                              )

    augmentations = Compose([
                            RandHistogramShift(
                                prob=0.5, num_control_points=80),
                            RandAffine(
                                prob=0.5,
                                translate_range=(1, 10, 10),
                                rotate_range=(0.17, 0.17, 0.17),
                                scale_range=(0.05, 0.05, 0.05),
                                padding_mode='border')
                            ])

    df_train, df_valid, df_test = subsets

    train_dataset = kneeMRIDataset(
        df_train, transform=transformations, augement=augmentations)
    valid_dataset = kneeMRIDataset(
        df_valid, transform=transformations, augement=None)
    test_dataset = kneeMRIDataset(
        df_test, transform=transformations, augement=None)

    return train_dataset, valid_dataset, test_dataset


def prepare_dataloaders(datasets, num_workers):
    train_dataset, valid_dataset, test_dataset = datasets

    train_sampler = prepare_weighted_sampler(train_dataset)

    train_dl = data.DataLoader(
        train_dataset, batch_size=64, pin_memory=True, drop_last=True, num_workers=num_workers, sampler=train_sampler)
    valid_dl = data.DataLoader(
        valid_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_dl = data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_dl, valid_dl, test_dl


def prepare_weighted_sampler(dataset):
    y = dataset.y.astype('uint8')

    counts = np.bincount(y)

    labels_weights = 1. / counts

    weights = labels_weights[y]

    return WeightedRandomSampler(weights, len(weights), replacement=True)
