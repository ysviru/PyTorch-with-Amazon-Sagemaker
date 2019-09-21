import os
import argparse
import logging
import sys

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

from torch.nn.init import xavier_uniform_ as xavier_uniform

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class DefaultHyperParameters:
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    TEST_BATCH_SIZE = 1


class SageMakerEnviron:
    SM_MODEL_DIR = "SM_MODEL_DIR"  # Defines the path where we store the model.
    SM_CHANNEL_TRAINING = "SM_CHANNEL_TRAINING"  # Defines the path where we store the data.
    SM_NUM_GPUS = "SM_NUM_GPUS"  # Defines the number of gpus on the system.


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(1337)

        self.fc1 = nn.Linear(15, 10)
        xavier_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(10, 3)
        xavier_uniform(self.fc2.weight)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(x)


def __get_train_data_loader(batch_size, training_dir):
    X_train_df = pd.read_csv(training_dir + "/X_train.csv")
    y_train_df = pd.read_csv(training_dir + "/y_train.csv")
    # The LongTensor() for "target" vector y_train_df is needed for the particular loss function criterion we use,
    # i.e., nn.CrossEntropyLoss() inside the train(args) function.
    train = TensorDataset(torch.Tensor(np.array(X_train_df)), torch.LongTensor(np.array(y_train_df)))
    return DataLoader(train, batch_size=int(batch_size), shuffle=True)


def __get_test_data_loader(test_batch_size, training_dir):
    X_validation_df = pd.read_csv(training_dir + "/X_validation.csv")
    y_validation_df = pd.read_csv(training_dir + "/y_validation.csv")
    # The LongTensor() for "target" vector y_validation_df is needed for the particular loss function criterion we use,
    # i.e., nn.CrossEntropyLoss() inside the train(args) function.
    validation = TensorDataset(torch.Tensor(np.array(X_validation_df)), torch.LongTensor(np.array(y_validation_df)))
    return DataLoader(validation, batch_size=int(test_batch_size))


def train(args):
    use_cuda = args.num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = __get_train_data_loader(args.batch_size, args.data_dir)
    test_loader = __get_test_data_loader(args.test_batch_size, args.data_dir)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        loss_ = []
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss_.append(loss.item())
            loss.backward()
            optimizer.step()
        print('\nTrain epoch: {}, Loss: {:.6f}'.format(
            epoch, np.mean(loss_)))
        test(model, test_loader, device, mode="Validation")
    save_model(model, args.model_dir)


def test(model, test_loader, device, mode="Test"):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.view(-1)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(pred)

    print("{}: accuracy={}".format(mode, ((correct * 100.0) / total)))
    test_loss /= len(test_loader.dataset)
    print("{}: loss = {}".format(mode, test_loss))
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def save_model(model, model_dir):
    path = os.path.join(model_dir, 'model.pth')
    torch.save(model.cpu().state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper parameters.
    parser.add_argument("--num-epochs", type=int, default=DefaultHyperParameters.NUM_EPOCHS, metavar="N",
                        help="number of epochs (default: {})".format(DefaultHyperParameters.NUM_EPOCHS))
    parser.add_argument("--learning-rate", type=float, default=DefaultHyperParameters.LEARNING_RATE, metavar="LR",
                        help="learning rate (default: {})".format(DefaultHyperParameters.LEARNING_RATE))
    parser.add_argument("--batch-size", type=float, default=DefaultHyperParameters.BATCH_SIZE, metavar="BSZ",
                        help="batch size (default: {})".format(DefaultHyperParameters.BATCH_SIZE))
    parser.add_argument("--test-batch-size", type=float, default=DefaultHyperParameters.TEST_BATCH_SIZE, metavar="TBSZ",
                        help="test batch size (default: {})".format(DefaultHyperParameters.TEST_BATCH_SIZE))

    parser.add_argument("--model-dir", type=str, default=os.environ[SageMakerEnviron.SM_MODEL_DIR])
    parser.add_argument("--data-dir", type=str, default=os.environ[SageMakerEnviron.SM_CHANNEL_TRAINING])
    parser.add_argument("--num-gpus", type=int, default=os.environ[SageMakerEnviron.SM_NUM_GPUS])

    train(parser.parse_args())
