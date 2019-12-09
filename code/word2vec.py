#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time    :2019/12/9  17:43
#!@File    :  .py

import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss,nn
import random
import sys
import time

# 获取数据，如果训练效果不好多半是数据的问题
with open("dataset.txt","r") as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]

# 简化数据
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda  x: x[1]>=5,counter.items()))

# 将词映射到整数索引
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]# 注意，此时数据集已全为索引
num_tokens = sum([len(st) for st in dataset])
print(num_tokens) # 486841个单词，感觉数据后面还得处理

# 二次采样
def discard(idx):
    return random.uniform(0,1) < 1 - math.sqrt(1e-4/counter[idx_to_token[idx]]*num_tokens)
subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print(sum([len(st) for st in subsampled_dataset]))# 274641

# 提取中心词和背景词
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2: # 至少两个单词才能组成中心词和背景词
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i-window_size),min(len(st),center_i+1+window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset,5)

# 负采样
#
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    # 每个中心词的噪声词， 负采样候选词，一般会多采些，
    population = list(range(len(sampling_weights)))# 所有的词的索引
    for contexts in all_contexts:# 每个中心词的背景词循环
        negatives = []
        while len(negatives) < len(contexts) * K:# 每个中心词有背景词的K倍个噪声词
            if i == len(neg_candidates):
                # 生成噪声词序列，每次序列都会改变
                i, neg_candidates = 0, random.choices(population, sampling_weights,k=int(1e5))
            neg, i = neg_candidates[i], i+1
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]** 0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

# 随机小批量读取数据
def batchify(data):
    max_len = max(len(c) + len(n) for _,c,n in data)
    centers, contexts_negatives, masks, labels = [],[],[],[]
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context+negative+[0]*(max_len-cur_len)]
        masks += [[1] * cur_len + [0] *(max_len-cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1,1)), nd.array(contexts_negatives),nd.array(masks),nd.array(labels))

batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers,all_contexts,all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size,shuffle=True,batchify_fn=batchify, num_workers=num_workers)

for batch in data_iter:
    for name, data in zip(['centers','contexts_negative','masks','labels'],batch):
        print(name, 'shape:',data.shape)
    break

# 跳字模型
embed = nn.Embedding(input_dim=20,output_dim=4)
embed.initialize()
embed.weight
x = nd.array([[1,2,3],[4,5,6]])
embed(x)

# 小批量乘法
X = nd.ones((2,1,4))
Y = nd.ones((2,4,6))
nd.batch_dot(X,Y).shape

def