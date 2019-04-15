

####基于双向LSTM神经网络实现中文分词及词性标注

采用4-tag标注，神经网络框架为Keras

####程序结构及使用说明

程序的目录结构如下：

```
.
|-- config.py			# 配置文件
|-- data				# 数据文件
|   |-- label			# 标注相关数据
|   |   |-- dicts.pkl
|   |   |-- diff.txt
|   |   |-- history
|   |   |-- input.txt
|   |   |-- model
|   |   |-- output.txt
|   |   |-- stand.txt
|   |   `-- transp.json
|   `-- segment			# 分词相关数据
|-- process_data.py		# 处理数据相关函数
|-- application.py		# 主程序	
|-- training.py			# 训练模型
`-- utility.py			# 需要的一些工具函数
```

使用方式，直接运行相应的Python文件，通过命令行参数决定是分词还是标注：

```bash
python process_data.py segment			# 加工分词数据
python process_data.py label			# 加工标注数据
python process_data.py segment label	# 加工分词与标注数据
python training.py segment				# 训练分词模型
python training.py label				# 训练标注模型
python application.py label				# 测试标注
python application.py segment			# 测试分词
```

配置文件说明：

```python
config = {
    "segment": {
        # 语料路径
        "raw_data": ["data/segment/pku_training.utf8"],
        # 语料处理后的保存路径
        "processed_data": "data/segment/training_data.pkl",
        "dicts":"data/segment/dicts.pkl",
        "model": "data/segment/model",
        "history": "data/segment/history",
        # 测试文件路径
        "input": "data/segment/input.txt",
        # 测试结果输出路径
        "output": "data/segment/output.txt",
        # 标准文件路径
        "stand": "data/segment/stand.txt",
        "transp": "data/segment/transp.json",
        "tags": ["S", "B", "M", "E"],
        # 模型参数
        "epochs":20,
        "sequence_length": 16,
        "word_dimension": 256,
        "hidden_units": 64
    },
    "label": {
        ...
        ...
    },
    "setting": {
        "temp":"data/temp/",
        "img":"data/img/",
    },
}

```

####测试

由于分词与标注的数据集不同，分词的语料比词性标注的要多一些，因为训练的时候是两个模型使用不同的语料单独训练

- 分词训练语料：SIGHAN Bakeoff 2005 Peking University，训练集
- 分词测试语料：SIGHAN Bakeoff 2005 Peking University，测试集
- 标注训练语料：词性标注，人民日报@199801
- 标注测试语料：词性标注，人民日报@199801

使用SIGHAN Bakeoff 2005 的比对脚本比较输出结果与人工标注的结果，测试结果如下，召回率、精度、F值都达到了0.902，算是一个不错的结果

```
INSERTIONS:	0
DELETIONS:	0
SUBSTITUTIONS:	0
NCHANGE:	0
NTRUTH:	27
NTEST:	27
TRUE WORDS RECALL:	1.000
TEST WORDS PRECISION:	1.000
=== SUMMARY:
=== TOTAL INSERTIONS:	3028
=== TOTAL DELETIONS:	2990
=== TOTAL SUBSTITUTIONS:	7198
=== TOTAL NCHANGE:	13216
=== TOTAL TRUE WORD COUNT:	104372
=== TOTAL TEST WORD COUNT:	104410
=== TOTAL TRUE WORDS RECALL:	0.902	# 召回率
=== TOTAL TEST WORDS PRECISION:	0.902	# 精度
=== F MEASURE:	0.902					# F值
=== OOV Rate:	0.058
=== OOV Recall Rate:	0.556
=== IV Recall Rate:	0.924
###	output.txt	3028	2990	7198	13216	104372	104410	0.902	0.902	0.902	0.058	0.556	0.924
```

分词标注一体，单独标注测试时准确度还是比较高的，但与分词结合时标注准确率就大幅下降了，一个重要的原因是分词的错误会产生大量的非登录词进而影响了标注的准确性，标注的准确度不会高于分词的准确度。

```
在/p  十五大/j  精神/n  指引/v  下/f  胜利/v  前/f  进/v  ——/w  元旦/t  献辞/n
我们/r  即将/d  以/p  丰收/v  的/u  喜悦/an  送/v  走/v  牛年/t  ，/w  以/p  昂扬/a  的/u  斗志/n  迎来/v  虎年/t  。/w  我们/r  伟大/a  祖国/n  在/p  新/a  的/u  一年/t  ，/w  将/d  是/v  充满/v  生机/n  、/w  充满/v  希望/n  的/u  一年/t  。/w
刚刚/d  过去/v  的/u  一年/t  ，/w  大气磅礴/i  ，/w  波澜壮阔/i  。/w  在/p  这/r  一年/t  ，/w  以/p  江/nr  泽民/nr  同志/n  为/v  核心/n  的/u  党中央/nt  ，/w  继承/v  邓/nr  小平/nr  同志/n  的/u  遗志/n  ，/w  高举/v  邓小平理论/n  的/u  伟大/a  旗帜/n  ，/w  领导/v  全党/n  和/c  全国/n  各族/r  人民/n  坚定不移/i  地/u  沿着/p  建设/v  有/v  中国/ns  特色/n  社会主义/n  道路/n  阔步/d  前/f  进/v  ，/w  写/v  下/v  了/u  改革/v  开放/v  和/c  社会主义/n  现代化/vn  建设/vn  的/u  辉煌/a  篇章/n  。/w  顺利/a  地/u  恢复/v  对/p  香港/ns  行使/v  主权/n  ，/w  胜利/v  地/u  召开/v  党/n  的/u  第十五/m  次/q  全国/n  代表大会/n  ———/w  两/m  件/q  大事/n  办/v  得/u  圆满/a  成功/a  。/w  国民经济/n  稳中求进/l  ，/w  国家/n  经济/n  实力/n  进一步/d  增强/v  ，/w  人民/n  生活/vn  继续/v  改善/v  ，/w  对外/vn  经济/n  技术/n  交流/vn  日益/d  扩大/v  。/w  在/p  国际/n  金融/n  危机/n  的/u  风浪/n  波及/v  许多/m  国家/n  的/u  情况/n  下/f  ，/w  我国/n  保持/v  了/u  金融/n  形势/n  和/c  整个/b  经济/n  形势/n  的/u  稳定/an  发展/vn  。/w  社会主义/n  精神文明/n  建设/vn  和/c  民主/a  法制/n  建设/vn  取得/v  新/a  的/u  成绩/n  ，/w  各项/r  社会/n  事业/n  全面/ad  进步/v  。/w  外交/n  工作/vn  取得/v  可喜/a  的/u  突破/vn  ，/w  我国/n  的/u  国际/n  地位/n  和/c  国际/n  威望/n  进一步/d  提高/v  。/w  实践/v  使/v  亿万/m  人民/n  对/p  邓小平理论/n  更加/d  信/v  仰/v  ，/w  对/p  以/p  江/nr  泽民/nr  同志/n  为/v  核心/n  的/u  党中央/nt  更加/d  信赖/v  ，/w  对/p  伟大/a  祖国/n  的/u  光辉/n  前景/n  更加/d  充满/v  信心/n  。/w
```


