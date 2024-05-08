import os
import shutil
import random

import torch
import time
import sys
import numpy as np
import torch.utils.data

from model import mGCN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from libs.pytorchtools import EarlyStopping


start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def readData(path):
    dataset = []
    files = []
    for file in os.listdir(path):
        if file[-4:] == '.npz':
            graph = np.load(os.path.join(path, file), allow_pickle=True)
            files.append(file[:-4])
            nodeAttr = torch.tensor(graph['nodeAttr'], dtype=torch.float)
            nodeAttr = nodeAttr.squeeze(1)
            edgeIndex = torch.tensor(graph['edgeIndex'], dtype=torch.long)
            edgeAttr = torch.tensor(graph['edgeAttr'], dtype=torch.float)

            
            label = torch.tensor(graph['label'], dtype=torch.long)
            data = Data(x=nodeAttr, edge_index=edgeIndex, edge_attr=edgeAttr, y=label)
            dataset.append(data)
    return dataset, files



def splitData(path, trainPath, testPath):
    if os.path.exists(trainPath):
        shutil.rmtree(trainPath)
    if os.path.exists(testPath):
        shutil.rmtree(testPath)
    os.makedirs(trainPath)
    os.makedirs(testPath)

    files_pos = []
    files_neg = []
    for commit in os.listdir(path):
        filePath = os.path.join(path, commit, 'out.npz')
        graph = np.load(filePath, allow_pickle=True)
        if graph['label'] == 1:
            files_pos.append(filePath)
        else:
            files_neg.append(filePath)

    random.shuffle(files_pos)
    random.shuffle(files_neg)

    train = files_pos[:int(0.8 * len(files_pos))]
    train += files_neg[:int(0.8 * len(files_neg))]
    test = files_pos[int(0.8 * len(files_pos)):]
    test += files_neg[int(0.8 * len(files_neg)):]

    for file in train:
        fromPath = file.replace('\\', '/')
        toPath = fromPath.replace('graph', 'train').replace('/out', '')
        shutil.copy(fromPath, toPath)
    for file in test:
        fromPath = file.replace('\\', '/')
        toPath = fromPath.replace('graph', 'test').replace('/out', '')
        shutil.copy(fromPath, toPath)



def train(model, trainLoader, optimizer,  criterion):
    model.to(device)
    model.train()

    lossAll = 0
    for data in trainLoader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model.forward(data)
        loss = criterion(out, data.y)
        loss.backward()
        lossAll += loss.item() * len(data.y)
        optimizer.step()
    lossAll /= len(trainLoader.dataset)
    return model, lossAll


def test(model, testLoader, criterion):
    model.to(device)
    model.eval()

    correct = 0
    lossAll = 0
    preds = []
    labels = []
    for data in testLoader:
        data.to(device)
        out = model.forward(data)
        predicition = out.argmax(dim=1)
        loss = criterion(out, data.y)
        lossAll += loss.item() * len(data.y)
        correct += int((predicition == data.y).sum())
        preds.extend(predicition.int().tolist())
        labels.extend(data.y.int().tolist())
    acc = correct / len(testLoader.dataset)
    lossAll /= len(testLoader.dataset)

    preds = np.array(preds)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    F1 = f1_score(labels, preds)
    return acc, lossAll, pre, recall, F1 


def main():
    BATCHSIZE = 32
    EPOCHS = 100
    node_feature_dim = 1024
    num_classes = 2
    learning_rate = 0.001
    weight_decay = 5e-4
    

    model = mGCN(node_feature_dim, num_classes)


    optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


    criterion = torch.nn.CrossEntropyLoss()


    path = 'data/graph'
    trainPath = 'data/train'
    testPath = 'data/test'

    splitData(path, trainPath, testPath)
    dataset, _ = readData(trainPath)
    train_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)
    dataset, _ = readData(testPath)
    test_loader = DataLoader(dataset=dataset, batch_size=BATCHSIZE, shuffle=True)


    trainAcc, trainLoss = [], []
    testAcc, testLoss = [], []


    for epoch in range(EPOCHS):
        model, train_loss = train(model, train_loader, optimizer, criterion)
        scheduler.step()
        train_acc, train_loss, _, _, _ = test(model, train_loader, criterion)
        test_acc, test_loss, test_pre, test_rec, test_F1 = test(model, test_loader, criterion)
        print(f'Epoch: {epoch + 1:}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}, Test Pre: {test_pre:.4f}, Test Rec: {test_rec:.4f}, Test F1: {test_F1:.4f}, Test Loss: {test_loss:.4f}')

        trainAcc.append(train_acc)
        trainLoss.append(train_loss)
        testAcc.append(test_acc)
        testLoss.append(test_loss)



if __name__ == '__main__':
    main()


