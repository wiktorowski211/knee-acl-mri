import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.tensorboard import SummaryWriter

from src.data import prepare_data
from src.metrics import *
from src.model import *
from src.utils import *


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)
    return device


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model, dataloader, optimizer, criterion, epoch, num_epochs, writer, device):
    model.train()

    y_preds = np.array([])
    y_actuals = np.array([])

    losses = []

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred, sample = model(X)

        if i == 0:
            save_img(sample.detach().cpu().numpy(), filename=f"x_prim_{epoch}.png")

        loss = criterion(y_pred, y)
        loss.backward()

        optimizer.step()

        y_pred_np = y_pred.argmax(dim=1).cpu().detach().numpy().flatten()
        y_np = y.cpu().numpy().flatten()

        score = f1_score(y_np, y_pred_np, average='weighted')

        y_preds = np.append(y_preds, y_pred_np)
        y_actuals = np.append(y_actuals, y_np)

        step = epoch * len(dataloader) + i

        print(
            f"TRAIN Epoch: {epoch+1}/{num_epochs}, step: {step}, batch: {i+1}/{len(dataloader)}, score: {score}, loss: {loss}")

        writer.add_scalar("Train/loss", loss.item(), step)
        writer.add_scalar("Train/score/roc_auc", score, step)

        losses.append(loss.item())

    plot_gradients(model, epoch=epoch, writer=writer)

    epoch_loss = np.mean(losses)
    epoch_score = f1_score(y_actuals, y_preds, average='weighted')

    print(confusion_matrix(y_actuals, y_preds))

    writer.add_scalar("Train/epoch_loss", epoch_loss, epoch)
    writer.add_scalar("Train/epoch_score/roc_auc", epoch_score, epoch)

    return epoch_loss, epoch_score


def evaluate(model, dataloader, criterion, device):
    model.eval()

    losses = []

    y_preds = np.array([])
    y_actuals = np.array([])

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred, _ = model(X)

            loss = criterion(y_pred, y)

            y_pred_np = y_pred.argmax(dim=1).cpu().detach().numpy().flatten()
            y_np = y.cpu().numpy().flatten()

            y_preds = np.append(y_preds, y_pred_np)
            y_actuals = np.append(y_actuals, y_np)

            losses.append(loss.item())

    epoch_loss = np.mean(losses)
    epoch_score = f1_score(y_actuals, y_preds, average='weighted')

    print(confusion_matrix(y_actuals, y_preds))

    return epoch_loss, epoch_score


def evaluate_on_valid(model, dataloader, criterion, scheduler, epoch, num_epochs, writer, device):
    epoch_loss, epoch_score = evaluate(model, dataloader, criterion, device)

    scheduler.step(epoch_loss)

    print(
        f"\n\nVAL Epoch: {epoch+1}/{num_epochs}, score: {epoch_score}, loss: {epoch_loss}\n\n")

    writer.add_scalar("Val/epoch_loss", epoch_loss, epoch)
    writer.add_scalar("Val/epoch_score/roc_auc", epoch_score, epoch)

    return epoch_loss, epoch_score


def evaluate_on_test(model, dataloader, criterion, device, checkpoint_path=None):
    if checkpoint_path:
        model = load_checkpoint(checkpoint_path, model)

    epoch_loss, epoch_score = evaluate(model, dataloader, criterion, device)

    print(f"\n\nTEST Score: {epoch_score}\n\n")

    return epoch_loss, epoch_score


def run(args):
    num_epochs = args.epochs
    writer = SummaryWriter()

    seed_everything(0)

    device = get_device()

    (train_dl, valid_dl, test_dl), class_weights = prepare_data(batch_size=args.batch_size, sampling_frac=args.data_sampling_frac, num_workers=args.dataloader_num_workers)

    model = AclNet(architecture=args.architecture).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),reduction='mean')
    criterion = nn.CrossEntropyLoss()

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience, factor=.5, verbose=True)

    best_score = float(0)

    for epoch in range(num_epochs):
        _, _ = train(model, train_dl, optimizer,
                     criterion, epoch, num_epochs, writer, device)

        val_loss, val_score = evaluate_on_valid(
            model, valid_dl, criterion, scheduler, epoch, num_epochs, writer, device)

        if val_score > best_score:
            print(f"Improvement! Previous: {best_score}, new: {val_score}")
            best_score = val_score

            save_checkpoint("checkpoint.pt", model, optimizer,
                            val_loss, val_score, epoch)

    evaluate_on_test(model, test_dl, criterion, device,
                     checkpoint_path="checkpoint.pt")

    writer.flush()
    writer.close()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--architecture', type=str, default='resnet10',
                        choices=['resnet10', 'resnet34', 'resnet50', "efficientnet-b0"])
    parser.add_argument('--data_sampling_frac', type=float, default=1.0)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    args = parser.parse_args()
    print("Using params:", args)
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
