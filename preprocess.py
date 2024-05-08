import os
import re
import shutil
import sys
import time
import numpy as np
import pandas as pd
import javalang
import torch
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from transformers import AutoModel, AutoTokenizer, AutoConfig

from gensim.models import Word2Vec
import numpy as np
from libs.Tokenize import tokenize


def splitDataset(root):
    path = root + 'graph/'
    outPath = root + 'data/'
    for commit in os.listdir(path):
        filePath = path + commit + '/out.npz'
        if not os.path.exists(filePath):
            continue
        shutil.copy2(filePath, outPath + commit + '.npz')
        print(commit + ' is copying...')


def typeTokenize(typeOriginal):
    type = ""
    cnt = 0
    for i in range(len(typeOriginal)):
        if typeOriginal[i] == ' ':
            cnt += 1
            if cnt > 1:
                type += typeOriginal[i:]
                break
        if typeOriginal[i].isupper():
            type += ' ' + typeOriginal[i].lower()
        else:
            type += typeOriginal[i]

    type = type.lower()
    type = list(javalang.tokenizer.tokenize(type))
    type = [token.value for token in type]
    return type


def word2vec():
    path = 'data/graph/'
    List = []
    for commit in os.listdir(path):
        filePath = path + commit + '/out.log'
        if not os.path.exists(filePath):
            continue
        nodes, _, _ = importGraph(filePath)
        for node in nodes:
            code = tokenize(node[5])
            if code not in List:
                List.append(code)
            type = typeTokenize(node[4])
            if type not in List:
                List.append(type)
    print(len(List))

    with open('logs/list.txt', 'w', encoding='utf-8') as f:
        for item in List:
            f.write("%s\n" % item)
    print('正在训练...')
    model = Word2Vec(List, min_count=1, vector_size=64)
    model.save('./models/word2vec/whole.model')



def procNode(line):
    '''
    :param line: (24, -1, 0, '88', 'RETURN', 'return new ArrayList<String>();')
    :return: [24, -1, 0, 88, RETURN, return new ArrayList<String>();]
    '''
    
    content = line[1:-2]

    comma = [m.start() for m in re.finditer(',', content)]
    nodeID = content[:comma[0]]
    version = content[comma[0] + 2:comma[1]]
    nodeDegree = content[comma[1] + 2:comma[2]]
    nodeLine = content[comma[2] + 3:comma[3] - 1]
    nodeType = content[comma[3] + 3:comma[4] - 1]
    nodeContent = content[comma[4] + 3:-1]
    node = np.array([nodeID, version, nodeDegree, nodeLine, nodeType, nodeContent], dtype=object)
    return node


def procEdge(line):
    '''
    :param line:(22, 18, 'DDG', -1)
    :return:[22,18,DDG,-1]
    '''

    line = line.rstrip('\n')
    content = line[1:-1]

    comma = [m.start() for m in re.finditer(',', content)]
    nodeID1 = content[:comma[0]]
    nodeID2 = content[comma[0] + 2:comma[1]]
    edgeType = content[comma[1] + 3:comma[2] - 1]
    version = content[comma[2] + 2:]

    edge = np.array([nodeID1, nodeID2, edgeType, version], dtype=object)
    return edge


def parseEdge(edges):
    '''
    AST:3
    CDG:5
    DDG:7
    CFG:9
    pre:1
    ctx:3
    pst:2
    :param edges:[[22,18,DDG,-1],...]
    :return:nodeDict - {22:0,18:1,...}
            edgeIndex -  [[22,18],...]
            edgeAttr - [[7,1],...]
    '''
    nodeFrom = [edge[0] for edge in edges]
    nodeTo = [edge[1] for edge in edges]
    nodeType = [edge[2] for edge in edges]
    nodeVersion = [edge[3] for edge in edges]

    nodeSet = nodeFrom + nodeTo
    nodeSet = {}.fromkeys(nodeSet)
    nodeSet = list(nodeSet.keys())
    nodeDict = {node: i for i, node in enumerate(nodeSet)}

    nodeFromIndex = [nodeDict[i] for i in nodeFrom]
    nodeToIndex = [nodeDict[i] for i in nodeTo]

    edgeIndex = []
    edgeIndex = np.array([nodeFromIndex, nodeToIndex])

    edgeAttr = []
    for edge in edges:
        attr = [0, 0, 0, 0, 0, 0]
        if '-1' == edge[3]:
            attr[0] = 1
        elif '1' == edge[3]:
            attr[1] = 1
        else:
            attr[0] = 1
            attr[1] = 1

        if 'A' in edge[2]:
            attr[2] = 1
        if 'C' in edge[2]:
            attr[3] = 1
        if 'D' in edge[2]:
            attr[4] = 1
        if 'F' in edge[2]:
            attr[5] = 1
        edgeAttr.append(attr)
    edgeAttr = np.array(edgeAttr)
    return nodeDict, edgeIndex, edgeAttr    



def parseNodeBert(nodes, nodeDict, model, tokenizer):
    '''
    :param nodes: [24, -1, 0, 88, RETURN, return new ArrayList<String>();]
    :param nodeDict: {22:0,18:1,...}
    :return:
    '''
    
    nodeID = [node[0] for node in nodes]
    nodeOrder = [nodeID.index(node) for node in nodeDict]
    for i in range(len(nodeID)):
        if i not in nodeOrder:
            nodeOrder.append(i)
    newNodes = [nodes[order] for order in nodeOrder]

    nodeAttr = []
    for node in newNodes:
        type = node[4]
        code = node[5]

        codeTokens = tokenizer(code)
        codeTokens = tokenizer.convert_tokens_to_ids(codeTokens)

        typeTokens = typeTokenize(type)
        typeTokens = tokenizer.convert_tokens_to_ids(typeTokens)


        if len(codeTokens) > 20:
            codeTokens = codeTokens[:20]
        codeTokensTensor = torch.tensor([codeTokens])
        typeTokensTensor = torch.tensor([typeTokens])
        if codeTokensTensor.shape == torch.Size([1, 0]):
            codeTokensTensor = torch.tensor([[0]])
        with torch.no_grad():
            output1 = model(codeTokensTensor)
            output2 = model(typeTokensTensor)

        codeEmbedding = output1.last_hidden_state.squeeze()
        if len(codeEmbedding.shape) == 1:
            codeEmbedding = codeEmbedding.unsqueeze(0)
        if codeEmbedding.shape[0] > 12:
            codeEmbedding = codeEmbedding[:12]
        elif codeEmbedding.shape[0] < 12:
            codeEmbedding = torch.cat((codeEmbedding, torch.zeros(12 - len(codeEmbedding), 128)), dim=0)

        typeEmbedding = output2.last_hidden_state.squeeze()
        if len(typeEmbedding.shape) == 1:
            typeEmbedding = typeEmbedding.unsqueeze(0)
        if typeEmbedding.shape[0] < 4:
            typeEmbedding = torch.cat((typeEmbedding, torch.zeros(4 - len(typeEmbedding), 128)), dim=0)


        codeEmbedding = codeEmbedding.reshape(1, -1)
        typeEmbedding = typeEmbedding.reshape(1, -1)

        nodeEmbedding = np.c_[typeEmbedding, codeEmbedding]

        nodeAttr.append(nodeEmbedding)
    return nodeAttr


def parseNodeW2V(nodes, nodeDict, model):
    '''
    :param nodes: [24, -1, 0, 88, RETURN, return new ArrayList<String>();]
    :param nodeDict: {22:0,18:1,...}
    :return:
    '''
    
    nodeID = [node[0] for node in nodes]
    nodeOrder = [nodeID.index(node) for node in nodeDict]
    for i in range(len(nodeID)):
        if i not in nodeOrder:
            nodeOrder.append(i)
    newNodes = [nodes[order] for order in nodeOrder]

    nodeAttr = []
    for node in newNodes:
        type = node[4]
        code = node[5]

        typeTokens = typeTokenize(type)
        codeTokens = tokenize(code)
        try:
            typeEmbedding = model.wv[typeTokens]
            codeEmbedding = model.wv[codeTokens]
        except:
            print('error')
            typeEmbedding = np.zeros((1, 64))
            codeEmbedding = np.zeros((1, 64))

        if typeEmbedding.shape[0] < 4:
            typeEmbedding = np.concatenate((typeEmbedding, np.zeros((4 - typeEmbedding.shape[0], 64))), axis=0)
        if codeEmbedding.shape[0] < 12:
            codeEmbedding = np.concatenate((codeEmbedding, np.zeros((12 - codeEmbedding.shape[0], 64))), axis=0)
        elif codeEmbedding.shape[0] > 12:
            codeEmbedding = codeEmbedding[:12]


        nodeEmbedding = np.concatenate((typeEmbedding, codeEmbedding), axis=0)

        nodeEmbedding = nodeEmbedding.reshape(1, -1)
        nodeAttr.append(nodeEmbedding)
    return nodeAttr


def importGraph(path):
    f = open(path, encoding='utf-8', errors='ignore')
    lines = f.readlines()
    f.close()
    mergeNodes = []
    mergeEdges = []
    flag = 0
    label = int(lines[0][6])
    for line in lines[1:]:
        if not line.startswith('('):
            flag += 1
            continue
        if flag == 1:
            node = procNode(line)
            mergeNodes.append(node)
        else:
            edge = procEdge(line)
            mergeEdges.append(edge)
    return mergeNodes, mergeEdges, label


def preprocess(m):
    path = 'data/graph'
    if m == 'codebert':

        model_path = 'microsoft/codebert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path, hidden_size=128, num_hidden_layers=4, num_attention_heads=4)
        model = AutoModel.from_config(config)

    elif m == 'word2vec':
        word2vec()
        model = Word2Vec.load('./models/word2vec/whole.model')

    for commit in os.listdir(path):
        graphPath = path + '/' + commit + '/out.log'
        saveName = './data/graph/' + commit + '/' + 'out.npz'
        if os.path.exists(saveName):
            os.remove(saveName)
        print(commit + ' is processing...')
        nodes, edges, label = importGraph(graphPath)
        nodeDict, edgeIndex, edgeAttr = parseEdge(edges)
        if m == 'codebert':
            # codeBert
            nodeAttr = parseNodeBert(nodes, nodeDict, model, tokenizer)
        elif m == 'word2vec':
            # word2vec
            nodeAttr = parseNodeW2V(nodes, nodeDict, model)
        label = np.array([label])
        np.savez(saveName, nodeAttr=nodeAttr, edgeIndex=edgeIndex, edgeAttr=edgeAttr, label=label)
        print(commit + ' is done!')




if __name__ == '__main__':
    m = 'word2vec'
    preprocess(m)

