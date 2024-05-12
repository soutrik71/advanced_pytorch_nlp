import torch
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def plot_loss_accuracy(
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    labels,
    colors,
    loss_legend_loc="upper center",
    acc_legend_loc="upper left",
    legend_font=5,
    fig_size=(16, 10),
    sub_plot1=(1, 2, 1),
    sub_plot2=(1, 2, 2),
):

    plt.rcParams["figure.figsize"] = fig_size

    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])

    for i in range(len(train_loss)):
        x_train = range(len(train_loss[i]))
        x_val = range(len(val_loss[i]))

        min_train_loss = np.array(train_loss[i]).min()

        min_val_loss = np.array(val_loss[i]).min()

        plt.plot(
            x_train,
            train_loss[i],
            linestyle="-",
            color="tab:{}".format(colors[i]),
            label="TRAIN ({0:.4}): {1}".format(min_train_loss, labels[i]),
        )
        plt.plot(
            x_val,
            val_loss[i],
            linestyle="--",
            color="tab:{}".format(colors[i]),
            label="VALID ({0:.4}): {1}".format(min_val_loss, labels[i]),
        )

    plt.xlabel("epoch no.")
    plt.ylabel("loss")
    plt.legend(loc=loss_legend_loc, prop={"size": legend_font})
    plt.title("Training and Validation Loss")

    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])

    for i in range(len(train_acc)):
        x_train = range(len(train_acc[i]))
        x_val = range(len(val_acc[i]))

        max_train_acc = np.array(train_acc[i]).max()

        max_val_acc = np.array(val_acc[i]).max()

        plt.plot(
            x_train,
            train_acc[i],
            linestyle="-",
            color="tab:{}".format(colors[i]),
            label="TRAIN ({0:.4}): {1}".format(max_train_acc, labels[i]),
        )
        plt.plot(
            x_val,
            val_acc[i],
            linestyle="--",
            color="tab:{}".format(colors[i]),
            label="VALID ({0:.4}): {1}".format(max_val_acc, labels[i]),
        )

    plt.xlabel("epoch no.")
    plt.ylabel("accuracy")
    plt.legend(loc=acc_legend_loc, prop={"size": legend_font})
    plt.title("Training and Validation Accuracy")

    plt.show()


def train_module(
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metric,
    train_losses: list,
    train_metrics: list,
):

    # setting model to train mode
    model.train()
    pbar = tqdm(train_dataloader)

    # batch metrics
    train_loss = 0
    train_metric = 0
    processed_batch = 0

    for idx, (data, label) in enumerate(pbar):
        # setting up device
        data = data.to(device)
        label = label.to(device)

        # forward pass output
        preds = model(data)

        # calc loss
        loss = criterion(preds, label)
        train_loss += loss.item()
        # print(f"training loss for batch {idx} is {loss}")

        # backpropagation
        optimizer.zero_grad()  # flush out  existing grads
        loss.backward()  # back prop of weights wrt loss
        optimizer.step()  # optimizer step -> minima

        # metric calc
        preds = torch.argmax(preds, dim=1)
        # print(f"preds:: {preds}")
        metric.update(preds, label)
        train_metric += metric.compute().detach().item()

        # updating batch count
        processed_batch += 1

        pbar.set_description(
            f"Avg Train Loss: {train_loss/processed_batch} Avg Train Metric: {train_metric/processed_batch}"
        )

    # It's typically called after the epoch completes
    metric.reset()
    # updating epoch metrics
    train_losses.append(train_loss / processed_batch)
    train_metrics.append(train_metric / processed_batch)

    return train_losses, train_metrics


def test_module(
    model: torch.nn.Module,
    device: torch.device,
    test_dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    metric,
    test_losses,
    test_metrics,
):
    # setting model to eval mode
    model.eval()
    pbar = tqdm(test_dataloader)

    # batch metrics
    test_loss = 0
    test_metric = 0
    processed_batch = 0

    with torch.inference_mode():
        for idx, (data, label) in enumerate(pbar):
            data, label = data.to(device), label.to(device)
            # predictions
            preds = model(data)
            # print(preds.shape)
            # print(label.shape)

            # loss calc
            loss = criterion(preds, label)
            test_loss += loss.item()

            # metric calc
            preds = torch.argmax(preds, dim=1)
            metric.update(preds, label)
            test_metric += metric.compute().detach().item()

            # updating batch count
            processed_batch += 1

            pbar.set_description(
                f"Avg Test Loss: {test_loss/processed_batch} Avg Test Metric: {test_metric/processed_batch}"
            )

        # It's typically called after the epoch completes
        metric.reset()
        # updating epoch metrics
        test_losses.append(test_loss / processed_batch)
        test_metrics.append(test_metric / processed_batch)

    return test_losses, test_metrics


def MakePredictions(model, loader):
    Y_shuffled, Y_preds = [], []
    for X, Y in loader:
        preds = model(X)
        Y_preds.append(preds)
        Y_shuffled.append(Y)

    Y_preds, Y_shuffled = torch.cat(Y_preds), torch.cat(Y_shuffled)
    return (
        Y_shuffled.detach().cpu().numpy(),
        F.softmax(Y_preds, dim=-1).argmax(dim=-1).detach().cpu().numpy(),
    )
