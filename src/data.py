import numpy as np
import pandas as pd
import torch.utils.data as data
from monai.transforms import (CenterSpatialCrop, Compose, GaussianSmooth,
                              Rand3DElastic, RandAffine, RandHistogramShift,
                              Resize, ScaleIntensity, ToTensor)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import WeightedRandomSampler

from src.utils import *


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

        z_m, y_m, x_m = img.shape

        roi = img[
            max(0, row.roiZ-2): min(z_m, row.roiZ+row.roiDepth+2),
            max(0, row.roiY-5): min(y_m, row.roiY+row.roiWidth+5),
            max(0, row.roiX-5): min(x_m, row.roiX+row.roiHeight+5),
        ]

        roi = np.expand_dims(roi, axis=0)

        if self.augement:
            roi = self.augement(roi)

        if self.transform:
            roi = self.transform(roi)

        return [roi, self.y.iloc[idx]]


def prepare_data(batch_size=64, sampling_frac=1.0, num_workers=None):
    df = load_df(sampling_frac)

    subsets = split_data(df)

    datasets = prepare_datasets(subsets)

    dataloaders = prepare_dataloaders(datasets, batch_size, num_workers)

    class_weights = prepare_class_weights(df["aclDiagnosis"])

    return dataloaders, class_weights


def load_df(sampling_frac):
    df = pd.read_csv("data/metadata.csv")

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
    transformations = Compose(
        [
            Resize((16, 96, 96)),
            GaussianSmooth(sigma=0.2),
            ScaleIntensity(minv=-1.0, maxv=1.0),
            ToTensor()
        ]
    )

    augmentations = Compose(
        [
            RandHistogramShift(
                prob=0.1, num_control_points=(800, 1000)),
            RandAffine(
                prob=0.1,
                translate_range=(3, 10, 10),
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode='border'),
            Rand3DElastic(prob=0.1, sigma_range=(
                3, 7), magnitude_range=(10, 50))
        ]
    )

    df_train, df_valid, df_test = subsets

    train_dataset = kneeMRIDataset(
        df_train, transform=transformations, augement=augmentations)
    valid_dataset = kneeMRIDataset(
        df_valid, transform=transformations, augement=None)
    test_dataset = kneeMRIDataset(
        df_test, transform=transformations, augement=None)

    return train_dataset, valid_dataset, test_dataset


def prepare_dataloaders(datasets, batch_size, num_workers):
    train_dataset, valid_dataset, test_dataset = datasets

    train_sampler = prepare_weighted_sampler(train_dataset)

    train_dl = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=num_workers, sampler=train_sampler)
    valid_dl = data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    test_dl = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_dl, valid_dl, test_dl


def prepare_weighted_sampler(dataset):
    y = dataset.y.astype('uint8')

    counts = np.bincount(y)

    labels_weights = 1. / counts

    weights = labels_weights[y]

    return WeightedRandomSampler(weights, len(weights), replacement=True)


def prepare_class_weights(y):
    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = torch.FloatTensor(class_weights)
    return class_weights
