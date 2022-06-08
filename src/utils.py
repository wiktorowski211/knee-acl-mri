import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_img(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def show_img(*imgs, cmap='gray'):
    imgs = list(imgs)

    fig = plt.figure(figsize=(10, 10))

    for index, img in enumerate(imgs):
        fig.add_subplot(1, len(imgs), index+1)
        plt.imshow(img[img.shape[0]//2, :, :], cmap=cmap)

    plt.show()


def save_img(*imgs, filename='img.png', cmap='gray'):
    imgs = list(imgs)

    fig = plt.figure(figsize=(4, 4))

    for index, img in enumerate(imgs):
        fig.add_subplot(1, len(imgs), index+1)
        plt.imshow(img, cmap=cmap)

    plt.savefig(filename)


def show_planes(img, cmap='gray'):
    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(1, 3, 1)
    plt.imshow(img[img.shape[0]//2, :, :], cmap=cmap)

    fig.add_subplot(1, 3, 2)
    plt.imshow(img[:, img.shape[1]//2, :], cmap=cmap)

    fig.add_subplot(1, 3, 3)
    plt.imshow(img[:, :, img.shape[2]//2], cmap=cmap)

    plt.show()


def describe_img(img):
    print(img.min(), img.max(), img.mean(), img.shape)


def show_overlay(img1, img2, alpha=0.5):
    img1_slice = img1[img1.shape[0]//2, :, :]
    img2_slice = img2[img2.shape[0]//2, :, :]

    plt.figure(figsize=(10, 10))
    plt.imshow(img1_slice, cmap='gray')
    plt.imshow(img2_slice, cmap='hot', vmin=0.0, vmax=1.0, alpha=alpha)
    plt.show()


def compare_histogram(img1, img2, bins=100):
    plt.hist(img1.flatten(), bins=bins, alpha=0.5, label='x')
    plt.hist(img2.flatten(), bins=bins, alpha=0.5, label='y')
    plt.show()


def compare_diff(img1, img2):
    x = np.absolute(np.subtract(img1.astype(np.int16), img2.astype(np.int16)))
    show_img(img1, x, img2)


def save_checkpoint(checkpoint_path, model, optimizer, val_loss, val_score, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        'score': val_score,
    }, checkpoint_path)
    print("Checkpoint saved!")


def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def plot_gradients(model, epoch, writer):
    for subnet_name, subnet in model.named_children():
        for name, module in subnet.named_children():
            parameters = module.parameters()
            norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
            writer.add_scalar(f'Gradients/{subnet_name}_{name}', norm, epoch)
