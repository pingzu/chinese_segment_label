# coding=utf-8
import re
import pickle
import json
import time
import os
import numpy as np
from keras import layers
from keras import models
from config import config
import matplotlib.pyplot as plt
# from create_model import create_model

def save_pickle(path, data):
    with open(path, 'wb') as fp:
        pickle.dump(data, fp)

def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)

def save_json(path, data):
    with open(path, 'w') as fp:
        json.dump(data, fp)

def load_json(path):
    with open(path, "r") as fp:
        return json.load(fp)


def save_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)


def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def create_model(max_features, sequence_length, word_dimension, hidden_units, label_count, compile=True):
    '''
    max_features    ：输入整数序列的最大值
    sequence_length ：输入序列的规整化长度
    word_dimension  ：将整数序列映射的目标向量维度
    hidden_units    ：BiLSTM 层隐藏结点数
    label_count     ：标签数量
    '''
    model = models.Sequential()

    # Embedding：将整数序列映射为词向量
    model.add(layers.Embedding(max_features+100, word_dimension, input_length=sequence_length, mask_zero=True))

    # Bidirectional LSTM：对序列数据进行处理
    model.add(layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True), merge_mode="sum"))

    # TimeDistributed：对上一层每个时间步的输出进行一次 Dense 操作，输出预测结果
    model.add(layers.TimeDistributed(layers.Dense(label_count, activation="softmax")))

    if compile:
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def create_model_by_kind(kind, compile=True):
    sequence_length = config[kind]['sequence_length']
    word_dimension  = config[kind]['word_dimension']
    hidden_units    = config[kind]['hidden_units']
    label_count     = len(config[kind]['tags'])
    max_features    = len(load_pickle(config[kind]['dicts']))
    
    model = create_model(max_features,sequence_length,word_dimension,hidden_units,label_count,compile=compile)

    return model


# def create_train_model(sequence_length, word_dimension, hidden_units, compile=False):
#     max_features = len(load_pickle(config['segment']['dicts']))
#     label_count = len(config['segment']['tags'])

#     model = models.Sequential()
#     model.add(layers.Embedding(max_features+100, word_dimension, input_length=sequence_length, mask_zero=True))
#     model.add(layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True), merge_mode="sum"))
#     model.add(layers.TimeDistributed(layers.Dense(label_count, activation="softmax")))

#     if not compile:
#         model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     return model

# def create_train_model(sequence_length, word_dimension, hidden_units, max_features, label_count):
#     model = models.Sequential()
#     model.add(layers.Embedding(max_features+100, word_dimension, input_length=sequence_length, mask_zero=True))
#     model.add(layers.Bidirectional(layers.LSTM(hidden_units, return_sequences=True), merge_mode="sum"))
#     model.add(layers.TimeDistributed(layers.Dense(label_count, activation="softmax")))

#     model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#     return model


def load_model(kind):
    dicts   = load_pickle(config[kind]['dicts'])
    tags    = config[kind]["tags"]
    transp  = load_json(config[kind]["transp"])
    model   = create_model_by_kind(kind, compile=False)
    model.load_weights(config[kind]["model"])

    return model, dicts, tags, transp


def find_lcseque(s1, s2):   #source: https://blog.csdn.net/wateryouyo/article/details/50917812 
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = "ok"
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = "left"
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = "up"
    (p1, p2) = (len(s1), len(s2))

    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == "ok":  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == "left":  # 根据标记，向左找下一个
            p2 -= 1
        if c == "up":  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return s


def word2label(word):  # 将单个单词转换为状态
    if word == "":
        return ""
    if len(word) == 1:
        return "S"
    else:
        return "B" + "M" * (len(word) - 2) + "E"



def compare_seg():
    act = load_file(config['segment']["output"])
    std = load_file(config['segment']["stand"])
    actuls = act.split("\n")
    stands = std.split("\n")

    actl =''.join([word2label(i) for i in re.split("[\W]", act) if i])
    stdl =''.join([word2label(i) for i in re.split("[\W]", std) if i])

    if len(actl) != len(stdl):
        print('not same long label')
    else:
        mc = [i for i in range(len(actl)) if actl[i] == stdl[i]]
        print('label match: ', len(mc) / len(actl))
    
    
    if len(stands) != len(actuls):
        print('not same long',len(stands),len(actuls))
        pass
        # stand = [i for i in re.split('[\W|a-zA-Z0-9]', proto) if i]
        # actul = [i for i in re.split('[\W|a-zA-Z0-9]', actul) if i]
        # res = find_lcseque(stand, actul)
        # print(len(stand), len(actul), len(res), len(res)/len(stand))
    else:
        print(len(stands), len(actuls))
        res = []
        sts = []
        acs = []
        proc = 0
        total = len(stands)
        for i in range(len(stands)):
            proc += 1
            print('\rcompare:  {:%}'.format(proc/total), end='')
            stands[i] = [i for i in re.split("[\W]", stands[i]) if i]
            actuls[i] = [i for i in re.split("[\W]", actuls[i]) if i]
            sts.extend(stands[i])
            acs.extend(actuls[i])
            res.extend(find_lcseque(stands[i], actuls[i]))
        print()
        print(len(sts), len(acs), len(res), len(res) / len(sts))



def viterbi(weights, transp, states):
    routes = [dict()]
    for layer in range(1, len(weights)):
        routes.append(dict())
        for curr in states:
            prob, prev = max(
                [
                    (
                        weights[layer - 1][prev]
                        + transp[prev][curr]
                        + weights[layer][curr],
                        prev,
                    )
                    for prev in states
                ]
            )
            weights[layer][curr] = prob
            routes[layer][curr] = prev

    prob, prev = max([(weights[-1][state], state) for state in states])
    res = [prev]
    for layer in range(len(weights) - 1, 0, -1):
        prev = routes[layer][prev]
        res = [prev] + res
    return res

def plot_history():
    temp = config['setting']['temp']
    img = config['setting']['img']
    files = [i for i in os.listdir(temp) if i.startswith('history')]
    for f in files:
        history = load_pickle(temp+f)

        # epochs = range(1, len(loss) + 1)

        # loss = history['loss']
        # val_loss = history['val_loss']
        # plt.plot(epochs, loss, 'bo', label='Training loss')
        # plt.plot(epochs, val_loss, 'b', label='Validation loss')
        # plt.title('loss: '+f)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig('{}{}-{}'.format(img,'loss',f[8:]))
        plt.clf()   # clear figure

        acc = history['acc']
        val_acc = history['val_acc']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc: '+str(max(acc))[:5])
        plt.plot(epochs, val_acc, 'b', label='Validation acc: '+str(max(val_acc))[:5])

        # max_indx=np.argmax(val_acc)
        # show_max=str(val_acc[max_indx])[:5]
        # plt.annotate(show_max,xytext=(max_indx,val_acc[max_indx]),xy=(max_indx,val_acc[max_indx]))
        
        plt.title('accuracy: '+f)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}{}-{}'.format(img,'acc',f[8:]))
        # plt.show()
        # exit()

# plot_history()

def compare_label():
    output = load_file(config['label']['output'])
    stand = load_file(config['label']['stand'])
    # output = load_file(config['segment']['output'])
    # stand = load_file(config['segment']['stand'])

    out = re.findall('([\S]+)/([\w]+)',output)
    std = re.findall('([\S]+)/([\w]+)',stand)
    
    if len(out) != len(std):
        print('not same long',len(out),len(std))
        for i in range(0,len(out),10):
            print(out[i:i+10])
            print(std[i:i+10])
        exit()

    total = len(std)
    diff = [(i,std[i],out[i])for i in range(len(out)) if out[i][1] != std[i][1]]

    print(len(diff),len(std),'{:%}'.format(1.0-len(diff)/len(std)))

    data = '\n{:<5}\t{:<15}\t{:<15}'.format('no','stand','output')
    for i,j,k in diff:
        data += '\n{:<5}\t{:<15}\t{:<15}'.format(i,str(j),str(k))
    save_file(config['label']['diff'],data)

