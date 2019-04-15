# coding=utf-8
import re
import time
import pickle
import json
import re
import random
from keras.utils import np_utils
import numpy as np
from numpy import array
from config import config

path = 'data/label_people.txt'

path = r"E:\zhup\Desktop\NLP\train_data.json\train_data.json"
op = 'label_train.txt'
o = open(op,'w',encoding='utf-8')
tags = set()
with open(path,'r',encoding='utf-8') as f:
    line = f.readline()
    while line:
        data = json.loads(line)
        # sent = ['{}/{}'.format(d['word'],d['pos']) for d in data['postag']]
        tags |= set([d['pos'] for d in data['postag']])
        sent = ['{}/{}'.format(d['word'],d['pos']) for d in data['postag']]
        # o.write(' '.join(sent))
        # o.write('\n')
        line = f.readline()
print(tags)
# close(o)
exit()  

def process_label_data():
    # path = config['label']["raw_data"][0]
    path = 'data/label_people.txt'

    samples = []
    lineno = 0
    longwords = []
    with open(path,'r',encoding='utf-8') as fp:
        line = fp.readline()
        lineno = 1
        # while line:
        while line and lineno <10:
            longwords.extend([(''.join(re.findall('([\w]+)/[\w]+',i)) ,j) for (i,j) in  re.findall('(\[.*?\])([\w]+)',line)])
            samples.extend(re.findall('([^[\s]+)/([\w]+)',line))
            line = fp.readline()
            if lineno % 10000 == 0:
                print('processed {} lines'.format(lineno))
            lineno += 1
            
    # print(samples)
    # exit()
    # with open('data/temp.dat','wb') as fp:
    #     pickle.dump(samples,fp)
    # exit()
    # print(samples)
    states = config['label']['states']

    training_length = len(samples)
    sequence_length = config['label']['sequence_length']
    padding_length = (sequence_length - training_length % sequence_length) % sequence_length
    states_count = len(states)
    
    samples.extend([('#','/')*padding_length])

    words = sorted(list(set([i[0] for i in samples])))
    dicts = {words[i]:i+1 for i in range(len(words))}
    tags = {s:states.index(s) for s in states}

    # print(words[:100])
    step = sequence_length

    X = np.array([np.array([dicts[j[0]] for j in samples[i:i+step]],dtype=np.int32) for i in range(0,len(samples),step)])
    y = np.array([np.array([np_utils.to_categorical(tags[j[1]], states_count) for j in samples[i:i+step]],dtype=np.int32) for i in range(0,len(samples),step)])
    return X, y, dicts
process_label_data()
exit()
# path = 'data/label.txt'

# inp = open(path,'r',encoding='utf-8')
# f = '''  [不] 够/a 顺手/a ，/w 阵地战/n 射门/v  [不] /d 果断/a ，/w 防守/v 上/nd 漏洞/n 较/d 多/a ，/w 被/p 对方/n ８/m 号/n 孟/a 优胜/v 、/w １１/m 号/n 金百炼/nh 的/u 内线/n 偷袭/v 和/c 外围/n 强打/v 连连/d 得手/v 。/w 
#  距/v 全场/n 结束/v 只/d 差/v 一/m 分钟/q 时/nt ，/w 解放/v 军队/n  [不] 惜/v 使/v 队员/n 被/p 罚/v 黄牌/n 和/c 小/a 罚/v 出场/v ，/w 用尽/v 一切/r 手段/n 死死/d 缠住/v 天津/ns 队员/n ，/w 阻止/v 进攻/v ，/w 拖延/v 时间/n 以/p 保住/v ０/m ./w ５/m 的/u 优势/n 。/w 
#  决定/v 指出/v ，/w １９８７/m 年/nt １１/m 月/nt ２５/m 日/nt 辽宁/ns 队/n 对/p 山东/ns 队/n 一/m 场/q 比赛/v 中/nd 出现/v  [不] /d 正常/a 现象/n ，/w 场上/nl 表现/n 消极/a ，/w 赛风/n  [不] /d 正/a ，/w 引起/v 群众/n  [不] 满/a 。/w 
#  对/p 此/r ，/w 仲裁委员会/ni 提出/v 报告/n ，/w 竞委会/ni 多方/n 收集/v 反映/v 和/c 意见/n ，/w 认为/v 两/m 队/n 在/p 这/r 场/q 比赛/v 中/nd 违背/v 了/u 社会主义/n 体育/n 道德/n ，/w 是/vl 与/c "/w 公正/a 竞赛/v ，/w 团结/v 拼搏/v "/w 的/u 竞赛/v 原则/n  [不] 符/v 的/u 。/w 
#  对于/p 这/r 一/m 处理/v 决定/v ，/w 在场/v 记者/n 议论纷纷/i ，/w 各/r 抒/v 已/d 见/v ，/w 争论 [不] 休/i 。/w 
#  埃及/ns 教练/n 穆罕默德.古哈里/nh 也/d 在/p 下半时/nt 调兵遣将/i ，/w faw/ws 决心/n  [与] /c 爱军/ns 周旋/v 到底/d 。/w 
#  辽宁/ns 山东/ns 二/m 队/n 赛风/n  [不] /d 正/a '''

# line =  inp.readline()
# for line in f.split('\n'):
#     s = re.findall('([\S]+)/([\w]+)',line)
#     print(s)
# exit()

# def word2label(word):  # 将单个单词转换为状态
#     # states = 
#     if word == "":
#         return ""
#     if len(word) == 1:
#         return "S"
#     else:
#         return "B" + "M" * (len(word) - 2) + "E"


# def cmp(inp, stp):
#     print(len(inp.split('\n')))
#     print(len(stp.split('\n')))
#     inp = re.split(r'[\W\s0-9a-zA-Z_]',inp)
#     stp = re.split(r'[\W\s0-9a-zA-Z_]',stp)
#     # print(inp)
#     # print(stp)
#     iw = ''.join([i for i in inp if i!=''])
#     sw = ''.join([i for i in stp if i!=''])
#     print(len(iw),len(sw))
#     inp = [word2label(i) for i in inp if i]
#     stp = [word2label(i) for i in stp if i]
#     a = ''.join(inp)
#     b = ''.join(stp)
#     if len(a) != len(b):
#         print('not same long',len(a),len(b))
#         exit()
#     cnt = 0
#     total = len(b)
#     for i in range(len(a)):
#         cnt += a[i] == b[i]
#         if a[i] != b[i]:
#             print(i,iw[i-5:i+5],a[i-5:i+5],b[i-5:i+5])
#     print('match: {:%}'.format(cnt/total))

# from config import config
# with open(config['segment']['output'],'r',encoding='utf-8') as f:
#     inp = f.read()

# with open(config['segment']['stand'],'r',encoding='utf-8') as f:
#     stp = f.read()
    
# cmp(inp,stp)
# def segment_re(unprocessed):
#     processed = unprocessed
#     patterns = ["[\\W\\s]+", "[0-9一二三四五六七八九十]+[年月日]", "[a-zA-Z0-9_]+"]
#     # patterns = ["[\\W\\s]+", "[0-9一二三四五六七八九十]+[年月日]", "[a-zA-Z0-9_"]
#     for pattern in patterns:
#         unprocessed = processed
#         processed = []
#         print(len(unprocessed), len(processed))
#         for item in unprocessed:
#             if len(item[0]) == 0:
#                 pass
#             elif item[1] == False:
#                 processed.append(item)
#             else:
#                 sentence = item[0]
#                 subitem = []
#                 print(sentence,len(sentence))
#                 splited = re.findall(pattern,sentence)
#                 unsplited = re.split(pattern,sentence)
#                 for i in range(len(splited)):
#                     subitem.append((unsplited[i], True, "/"))
#                     subitem.append((splited[i], False, "/"))
#                 subitem.append((unsplited[-1],True,'/'))
#                 processed.extend(subitem)
#                 print("len", len(processed), len(subitem))
#         print(len(unprocessed), len(processed))
#     return processed


# s = """
# 共同创造美好的新世纪——二○○一年新年贺词
# （二○○○年十二月三十一日）（附图片1张）
# 女士们，先生们，同志们，朋友们：
# 2001年新年钟声即将敲响。人类社会前进的航船就要驶入21世纪的新航程。中国人民进入了向现代化建设第三步战略目标迈进的新征程。
# 过去的一年，是我国社会主义改革开放和现代化建设进程中具有标志意义的一年。在中国共产党的领导下，全国各族人民团结奋斗，国民经济继续保持较快的发展势头，经济结构的战略性调整顺利部署实施。西部大开发取得良好开端。精神文明建设和民主法制建设进一步加强。我们在过去几年取得成绩的基础上，胜利完成了第九个五年计划。我国已进入了全面建设小康社会，加快社会主义现代化建设的新的发展阶段。
# 面对新世纪，世界各国人民的共同愿望是：继续发展人类以往创造的一切文明成果，克服20世纪困扰着人类的战争和贫困问题，推进和平与发展的崇高事业，创造一个美好的世界。
# 我们希望，新世纪成为各国人民共享和平的世纪。在20世纪里，世界饱受各种战争和冲突的苦难。
# """
# u = (s, True, "/")
# r = segment_re([u])
# print(r)
# w = [i[0] for i in r]
# print(' '.join(w))
