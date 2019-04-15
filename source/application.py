# coding=utf-8
import pickle
import re
import pickle
import json
import time
import sys
import numpy as np
from keras import layers
from keras import models
from config import config
from utility import *

seg_model,seg_dicts,seg_states,seg_transp = load_model('segment')
if 'label' in sys.argv:
    lab_model,lab_dicts,lab_states,lab_transp = load_model('label')

def segment_nn_word(sent):
    # 加载模型
    model, dicts, states, transp = seg_model, seg_dicts, seg_states, seg_transp

    # 对句子进行切片规整化
    sequence_length = config['segment']["sequence_length"]
    snippet = [sent[i:i+sequence_length] for i in range(0,len(sent),sequence_length)]

    # 将句子转换为整数序列，输入模型中得到预测概率矩阵
    results = []
    for snip in snippet:
        sequence = [(dicts[i] if i in dicts else 0) for i in snip] + [0] * (sequence_length - len(snip))
        results.append(model.predict(np.array([sequence]))[0][: len(sent)])

    # 将分片的预测结果重新组合到一起
    r = results[0]
    for v in results[1:]:
        r = np.concatenate((r,v),axis=0)

    # 进一步处理
    for s in ['M','E']:      #词首不可能为 M E
        r[0][states.index(s)] = 1e-100
    for s in ['M','B']:      #词尾不可能为 M B
        r[-1][states.index(s)] = 1e-100

    # 取对数
    r = np.log(r)

    # 应用 viterbi 算法
    weights = [dict(zip(config['segment']["tags"], i)) for i in r]
    t = viterbi(weights, transp, states)

    # d = ['S','B','M','E']
    # t = [d[np.argmax(r[i])] for i in range(len(sent))]

    # 分词
    words = []
    for i in range(len(sent)):
        if t[i] in ["S", "B"] or words == []:
            words.append(sent[i])
        else:
            words[-1] += sent[i]
    return words



def segment_re(unprocessed):
    processed = unprocessed
    patterns = [
        # ("[\\s]+", '/'),
        ('(?:[○一二三四五六七八九十０１２３４５６７８９0-9]+[年月日]{1,})','t'),
        ("[0-9０１２３４５６７８９.]+[%％‰]?",'m'),
        ("[\\W][—]{0,2}",'w'),
        ("[a-zA-Zａ-ｚＡ-Ｚ]+",'nx')
    ]

    for pattern, label in patterns:
        unprocessed = processed
        processed = []
        for item in unprocessed:
            if not item[0]:			# 跳过空字符
                pass
            elif item[1] == False:	# 已切割完的词直接加入队列中
                processed.append(item)
            else:
                sentence = item[0]
                subitem = []
                splited = re.findall(pattern,sentence) # 匹配项，无需继续切割
                unsplited = re.split(pattern,sentence) # 介于匹配项之间的未分割词
                for i in range(len(splited)):
                    subitem.append((unsplited[i], True, ''))
                    subitem.append((splited[i], False, label)) # 为匹配项标注记性
                subitem.append((unsplited[-1],True,''))
                processed.extend(subitem)
    return processed

def segment_nn(unprocessed):
    processed = []
    proc = 0
    total = len(unprocessed)
    for item in unprocessed:
        proc += 1
        print('\rsegment_nn:  {:%}'.format(proc/total), end='')
        if not item[0]:
            pass
        elif item[1] == False:
            processed.append(item)
        else:
            sentence = item[0]
            splited = segment_nn_word(sentence)
            subitem = [(word, False, '') for word in splited]
            processed.extend(subitem)
    print()
    return processed

def segment(input):
    input = [(input, True, '')]

    input = segment_re(input)

    # input = segment_phrase(input)

    input = segment_nn(input)

    if 'label' not in sys.argv:
        return [i[0] for i in input] 
    else:
        input = label_nn(input)
        return ['{}/{}'.format(i,k) for (i,j,k) in input]


def label_nn_word(sent):
    model, dicts, states, transp = lab_model, lab_dicts, lab_states, lab_transp

    # 对词序列进行切片规整化
    sequence_length = config['label']["sequence_length"]
    snippet = [sent[i:i+sequence_length] for i in range(0,len(sent),sequence_length)]

    # 将词序列转换为整数序列，输入模型中得到预测概率矩阵
    results = []
    for snip in snippet:
        sequence = [(dicts[i[0]] if i[0] in dicts else 0) for i in snip] + [0] * (
            sequence_length - len(snip)
        )
        results.append(model.predict(np.array([sequence]))[0][: len(sent)])
    r = results[0]
    for v in results[1:]:
        r = np.concatenate((r,v),axis=0)

    # 已确定标注则将其余概率置为0，标注来源于正则切分确定的词性
    for i in range(len(sent)):      
        if sent[i][2]:
            for j in states:
                if j != sent[i][2]:
                    r[i][states.index(j)] = 1e-100 
    for j in range(len(sent)):
        if re.search("[\\w]",sent[j][0]):
            r[j][states.index('w')] = 1e-100 

    # fp = open('data/segment/temp.txt','w',encoding='utf-8')
    # for k in range(len(sent)):
    #     if sent[k][0] not in dicts:
    #         fp.write('{},{}\n'.format(sent[k][0], ['{}:{}'.format(states[i],r[k][i]) for i in range(len(states))]))

    r = np.log(r)
    weights = [dict(zip(config['label']["tags"], i)) for i in r]
    t = viterbi(weights, transp, states)
    for i in range(len(sent)):
        sent[i] = (sent[i][0],sent[i][1],t[i])
    return sent


def label_nn(unprocessed):
    if len(unprocessed) == 0:
        return unprocessed

    return label_nn_word(unprocessed)
    # # 将每两个不标注词之间的序列作为一个整体进行标注，不会超过一行的文本
    # # '/' 标识为不需要标注的词，如切割出的换行符等，获取这些词的索引
    # j = 0
    # unlabel = [i for i in range(len(unprocessed)) if unprocessed[i][2] == '/']  
    # if len(unlabel) == 0 or unlabel[-1] != len(unprocessed)-1:
    #     unlabel.append(len(unprocessed)-1)

    # processed = label_nn_word(unprocessed[:unlabel[0]])

    # # 合并标注词队列与不标注词队列
    # for i in range(1,len(unlabel)):
    #     last = unlabel[i-1]
    #     curr = unlabel[i]
    #     processed.append(unprocessed[last])
    #     processed.extend(label_nn_word(unprocessed[last+1:curr]))
    # print(len(unprocessed),len(processed))
    # return processed



def label(input):
    input = [(i, False, '') for i in input]
    # print(input)

    input = label_nn(input)
    # print(input)

    res =  ['{}/{}'.format(i,k) if i else '' for (i,j,k) in input]
    # print(res)
    # exit()

    return(res)

def test_segment():
    lines = load_file(config['segment']["input"])
    lines = lines.split('\n')

    splited = ['  '.join(segment(line)) for line in lines]

    result = '\n'.join(splited)
    # input = '自然语言处理是研究计算机处理人类语言的一门技术'
    # print(output)

    save_file(config['segment']["output"], result)
    # compare_seg()

def test_label():
    # compare_label(); exit()
    lines = load_file(config['label']["input"])
    lines = lines.split('\n')

    labeled = [" ".join(label(line.split())) for line in lines]

    result = '\n'.join(labeled)

    save_file(config['label']["output"], result)
    compare_label()
    pass


def main():
    if 'segment' in sys.argv:
        test_segment()
    elif 'label' in sys.argv:
        test_label()
    else:
        test_segment()




if __name__ == "__main__":
    start = time.clock()
    main()

    elapsed = time.clock() - start
    print("Time used:", elapsed)

