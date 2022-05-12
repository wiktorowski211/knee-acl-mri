import pickle

import matplotlib.pyplot as plt
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


def show_overlay(img1, img2, alpha=0.5):
    img1_slice = img1[img1.shape[0]//2, :, :]
    img2_slice = img2[img2.shape[0]//2, :, :]

    plt.figure(figsize=(10, 10))
    plt.imshow(img1_slice, cmap='gray')
    plt.imshow(img2_slice, cmap='hot', vmin=0.0, vmax=1.0, alpha=alpha)
    plt.show()


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
