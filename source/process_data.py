import time
import pickle
import json
import re
import random
import sys
import numpy as np
from config import config
from numpy import array
from numpy import shape
from utility import *



def word2label(word):       # 将单个单词转换为状态
    if word == "":
        return ""
    if len(word) == 1:
        return "S"
    else:
        return "B" + "M" * (len(word) - 2) + "E"


def calc_seg_transp(labels):    # 计算转换概率
    tags = config['segment']['tags']
    transc = {i:{j:0 for j in tags} for i in tags}
    for i in range(len(labels)-1):
        transc[labels[i]][labels[i+1]] += 1
    
    transp = {}
    for i in tags:
        total = sum([transc[i][j] for j in tags])
        transp[i] = {j:transc[i][j]/total for j in tags}
    return transp

# def generate_word_dict():   # 计算转换概率
#     words = set()
#     for path in config['segment']['word_dict']:
#         file = load_file(path)
#         re.split('[\W\sa-zA-Z0-9_]', file)

def to_categorical(no, count):  # 将数字转换成一个向量
    res = np.zeros(count, dtype=np.int32)
    res[no] = 1
    return res
    pass

def load_dictwords():           # 加载训练词典中的词
    path = config['segment']['dictionary']
    files = [path+i for i in os.listdir(path)]
    words = []
    for d in files:
        words.extend(load_file(d).split())
    return words


def process_segment_data():
    # 读取原始训练数据路径，并读入文件内容，多个文件以换行连接
    paths = config['segment']["raw_data"]
    file = '\n'.join([load_file(path) for path in paths])

    # 按空白符分割为词
    words = [s for s in file.split() if s]

    # words = [s for s in re.split('[\W\sa-zA-Z0-9_]',file) if s]
    # longwords = [s for s in words if len(s)>=4]
    # # for i in range(2):
    # random.shuffle(longwords)
    # words.extend(longwords)

    # dicwords = load_dictwords()

    # for i in range(10):
    #     random.shuffle(dicwords)
    #     words.extend(dicwords)

    # 将每个词转换为标签
    labels = [word2label(w) for w in words]

    # 将每个词与标签拼接为一个整串
    words = ''.join(words)
    labels = ''.join(labels)

    # 从配置中读取标签
    tags = config['segment']['tags']

    training_length = len(labels)
    sequence_length = config['segment']['sequence_length']
    padding_length = (sequence_length - training_length % sequence_length) % sequence_length
    tags_count = len(tags)
    
    words += '。'*padding_length
    labels += 'S'*padding_length

    # 将字典排序后再编号，保证同一语料中每次运行每个字的编号相同
    dicts = sorted(list(set(words)))
    dicts = {dicts[i]:i+1 for i in range(len(dicts))}

    # 将标签转为索引数字
    tag2num = {s:tags.index(s) for s in tags}

    # 切片步长
    step = sequence_length//2

    X = np.array([np.array([dicts[c] for c in list(words[i:i+sequence_length])],dtype=np.int32) for i in range(0,len(words)-step,step)])
    y = np.array([np.array([to_categorical(tag2num[i], tags_count) for i in list(labels[i:i+sequence_length])],dtype=np.int32) for i in range(0,len(words)-step,step)])

    print('shape(X): ',shape(X),' shape(y): ', shape(y))
    return X, y, dicts

def calc_label_transp(samples, tags):   # 计算转移概率，实际上并不需要得到概率值，只是统计是否可以从一个状态到达另一个状态
    transp = {s:{j:0 for j in tags} for s in tags}
    for i in range(1,len(samples)):
        transp[samples[i-1][1]][samples[i][1]] += 1

    for s in tags:
        sm = sum([i[1] for i in transp[s].items()])
        for j in transp[s].items():
            transp[s][j[0]] /= sm if sm else 1
    with open(config['label']['transp'],'w',encoding='utf-8') as fp:
        json.dump(transp,fp)

def process_label_data():
    # 读取原始训练数据路径，并读入文件内容
    paths = config['label']["raw_data"]
    file = '\n'.join([load_file(path) for path in paths])

    words = []
    for line in file.split('\n'):
        items = re.findall('([^[\s]+)/([\w]+)',line)
        words.extend([(i,j.lower()) for (i,j) in items])

    tags = config['label']['tags']
    training_length = len(words)
    sequence_length = config['label']['sequence_length']
    padding_length = (sequence_length - training_length % sequence_length) % sequence_length
    tags_count = len(tags)
    
    words.extend([('#','x')] * padding_length)
    dicts = sorted(list(set([i[0] for i in words])))
    dicts = {dicts[i]:i+1 for i in range(len(dicts))}
    tag2num = {s:tags.index(s) for s in tags}

    step = sequence_length
    sample_count = len(words)//sequence_length

    
    X = np.array([dicts[j[0]] for j in words]).reshape(sample_count, sequence_length)
    y = np.array([to_categorical(tag2num[j[1].lower()], tags_count) for j in words]).reshape(sample_count, sequence_length, tags_count)

    print('shape(X): ',shape(X),' shape(y): ', shape(y))
    return X, y, dicts

def process_label():
    print('process_label')
    X, y, dicts = process_label_data()

    # 保存加工后的数据
    with open(config['label']['processed_data'], 'wb') as fp:
        pickle.dump((X, y), fp)
    with open(config['label']['dicts'], 'wb') as fp:
        pickle.dump(dicts, fp)

def process_segment():
    print('process_segment')
    X, y, dicts = process_segment_data()

    # 保存加工后的数据
    with open(config['segment']['processed_data'], 'wb') as fp:
        pickle.dump((X, y), fp)
    with open(config['segment']['dicts'], 'wb') as fp:
        pickle.dump(dicts, fp)

def main():
    if 'segment' in sys.argv:
        process_segment()
    if 'label' in sys.argv:    
        process_label()

    print('finished')

if __name__ == "__main__":
    start = time.clock()
    main()

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
   