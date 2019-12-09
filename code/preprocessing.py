#!/usr/bin/env python
#!-*-coding:utf-8 -*-
#!@Time    :2019/12/9  15:08
#!@File    :  .py
# 将数据预处理并保存文件避免重复生成
from nltk.corpus import stopwords
import os

data_file = 'dataset.txt'

path = "../data/"
files = os.listdir(path)
lines = []

# 读取数据
for file in files:
    filename = path + file
    with open(filename,"r") as f:
        dataset = f.read()
        dataset = dataset.lower()# 小写化
        dataset = dataset.replace("\n",'').split('.')# 分句
        dataset = [st.split() for st in dataset] # 分词
        for data in dataset:
            # 去掉停用词
            lines.append([w for w in data if w not in stopwords.words('english')])

with open(data_file,"w")as f:
    for line in lines:
        for word in line:
            f.write(word)
            f.write(' ')
        f.write('\n')
    f.close()